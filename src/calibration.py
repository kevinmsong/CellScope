"""Spatial calibration (§4).

Four modes, and no fifth: a scale bar clicked in the image, a scale bar
measured on a separate reference frame at the same magnification, direct entry
of the resolution, or explicitly uncalibrated. Nothing is ever inferred from
file metadata — the lab JPEGs this
tool targets carry only ``XResolution = 96 dpi``, a screen default that has no
relationship to the microscope.

In uncalibrated mode the scales stay ``None`` and downstream code emits
pixel-space columns only. There is deliberately no fallback value: a 1.0 or a
NaN standing in for µm/px could be mistaken for a measurement.
"""

from __future__ import annotations

import math
from typing import Any, Sequence

from .types import Calibration, ImageRecord

Point = Sequence[float]  # (x, y)


def display_to_original(point: Point, display_scale: float) -> tuple[float, float]:
    """Map a click on the downscaled preview back to original pixel coordinates.

    ``display_scale`` comes from :func:`src.image_io.make_display` and is
    ``original_size / display_size``. A silent error in this one conversion
    would scale every physical measurement in the export, which is why it is a
    named, tested function rather than an inline multiplication.
    """
    if display_scale <= 0:
        raise ValueError(f"display_scale must be positive, got {display_scale!r}.")
    x, y = float(point[0]), float(point[1])
    return (x * display_scale, y * display_scale)


def pixel_distance(p1: Point, p2: Point) -> float:
    """Euclidean distance in pixels; handles diagonal scale bars correctly."""
    dx = float(p2[0]) - float(p1[0])
    dy = float(p2[1]) - float(p1[1])
    return math.hypot(dx, dy)


def calculate_scale(p1: Point, p2: Point, known_length_um: float) -> float:
    """µm per pixel from a scale bar of known physical length (§4A).

    ``µm_per_pixel = known_length_um / measured_length_pixels``
    """
    if known_length_um is None or float(known_length_um) <= 0:
        raise ValueError(
            f"Scale-bar length must be a positive number of µm, got {known_length_um!r}."
        )
    measured = pixel_distance(p1, p2)
    if measured <= 0:
        raise ValueError(
            "The two scale-bar endpoints are the same pixel; click the two ends "
            "of the bar."
        )
    return float(known_length_um) / measured


def calibration_from_scale_bar(
    p1: Point, p2: Point, known_length_um: float
) -> Calibration:
    """Build an isotropic :class:`Calibration` from two clicked endpoints.

    A scale bar measures one physical direction, so it can only justify an
    isotropic scale. Anisotropic pixels must be entered explicitly via
    :func:`calibration_from_resolution`.
    """
    um_per_px = calculate_scale(p1, p2, known_length_um)
    return Calibration(
        method="scale_bar",
        um_per_px_x=um_per_px,
        um_per_px_y=um_per_px,
        params={
            "point1_px": [float(p1[0]), float(p1[1])],
            "point2_px": [float(p2[0]), float(p2[1])],
            "measured_length_px": pixel_distance(p1, p2),
            "known_length_um": float(known_length_um),
        },
    )


def calibration_from_resolution(
    um_per_px_x: float, um_per_px_y: float | None = None
) -> Calibration:
    """Direct entry of X/Y resolution (§4B). Supports anisotropic pixels."""
    if um_per_px_y is None:
        um_per_px_y = um_per_px_x
    for name, value in (("X", um_per_px_x), ("Y", um_per_px_y)):
        if value is None or float(value) <= 0:
            raise ValueError(f"{name} resolution must be a positive µm/px, got {value!r}.")
    return Calibration(
        method="manual_resolution",
        um_per_px_x=float(um_per_px_x),
        um_per_px_y=float(um_per_px_y),
        params={
            "entered_um_per_px_x": float(um_per_px_x),
            "entered_um_per_px_y": float(um_per_px_y),
        },
    )


def uncalibrated() -> Calibration:
    """Explicitly uncalibrated (§4C). Measurements stay in px and px²."""
    return Calibration(method="uncalibrated", um_per_px_x=None, um_per_px_y=None, params={})


# --------------------------------------------------------------------------- #
# Unit labelling — used for column suffixes and axis titles
# --------------------------------------------------------------------------- #


def length_unit(calibration: Calibration) -> str:
    return "um" if calibration.is_calibrated else "px"


def area_unit(calibration: Calibration) -> str:
    return "um2" if calibration.is_calibrated else "px2"


def length_label(calibration: Calibration) -> str:
    return "µm" if calibration.is_calibrated else "px"


def area_label(calibration: Calibration) -> str:
    return "µm²" if calibration.is_calibrated else "px²"


def describe(calibration: Calibration) -> dict[str, Any]:
    """Metadata block for ``metadata.json`` (§18)."""
    return calibration.describe()


# --------------------------------------------------------------------------- #
# Calibration from a separate reference image (§4A, extended)
# --------------------------------------------------------------------------- #


class ReferenceMismatch(ValueError):
    """Raised when a reference image cannot legitimately calibrate a sample."""


def check_reference_compatible(reference: ImageRecord, sample: ImageRecord) -> None:
    """Refuse to transfer a scale between images of different pixel dimensions.

    Microscope software often exports the scale bar in a separate frame at the
    same magnification. µm/px then transfers -- but only if both frames came off
    the same sensor at the same resolution. Matching pixel dimensions is a
    necessary condition and the only one we can actually verify, so it is
    enforced; the researcher remains responsible for the magnification matching.
    """
    if (reference.height, reference.width) != (sample.height, sample.width):
        raise ReferenceMismatch(
            f"Reference image {reference.filename} is "
            f"{reference.width}x{reference.height} px but the sample "
            f"{sample.filename} is {sample.width}x{sample.height} px. "
            "A scale measured on a differently sized frame does not transfer. "
            "Use a reference captured at the same magnification and resolution."
        )


def calibration_from_reference_bar(
    bar_length_px: float,
    known_length_um: float,
    reference: ImageRecord | None = None,
    sample: ImageRecord | None = None,
    detection: dict[str, Any] | None = None,
) -> Calibration:
    """Build a calibration from a scale bar measured on a reference image.

    ``bar_length_px`` is whatever the researcher confirmed -- the auto-detected
    width, or their own corrected value. The reference image's identity and
    hash are recorded so the calibration remains traceable to the exact frame it
    came from (§18).
    """
    if bar_length_px is None or float(bar_length_px) <= 0:
        raise ValueError(
            f"Scale-bar length must be a positive number of pixels, got {bar_length_px!r}."
        )
    if known_length_um is None or float(known_length_um) <= 0:
        raise ValueError(
            f"Scale-bar physical length must be positive, got {known_length_um!r}."
        )

    if reference is not None and sample is not None:
        check_reference_compatible(reference, sample)

    um_per_px = float(known_length_um) / float(bar_length_px)
    params: dict[str, Any] = {
        "bar_length_px": float(bar_length_px),
        "known_length_um": float(known_length_um),
    }
    if reference is not None:
        params["reference_filename"] = reference.filename
        params["reference_sha256"] = reference.sha256
        params["reference_dimensions"] = [reference.width, reference.height]
    if detection is not None:
        params["detection"] = detection

    return Calibration(
        method="reference_scale_bar",
        um_per_px_x=um_per_px,
        um_per_px_y=um_per_px,
        params=params,
    )
