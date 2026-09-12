"""Measurement correctness on synthetic geometry (§19).

These verify the *measurement* layer against analytic values. They say nothing
about segmentation accuracy, and §20 forbids claiming otherwise.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
from conftest import (
    ABS_TOL_CIRCULARITY,
    MIN_SOLIDITY_CONVEX,
    REL_TOL_AREA,
    REL_TOL_AXIS,
    REL_TOL_RATIO,
    assert_close,
    assert_feret,
    disk_labels,
    ellipse_labels,
    rectangle_labels,
)

from src.calibration import calibration_from_resolution, uncalibrated
from src.morphometry import (
    contour_perimeter,
    measure_cells,
    perimeter_estimator_for,
    touches_border,
)
from src.types import ClusterRecord, ObjectRecord


def measure_single(labels, calibration=None, status="resolved"):
    """Measure a one-object label image and return the single row."""
    calibration = calibration or uncalibrated()
    resolved = (1,) if status == "resolved" else ()
    unresolved = () if status == "resolved" else (1,)
    frame = measure_cells(
        labels,
        [ObjectRecord(1, 1, status)],
        [ClusterRecord(1, (1,), resolved, unresolved)],
        calibration,
    )
    assert len(frame) == 1
    return frame.iloc[0]


# --------------------------------------------------------------------------- #
# Circles
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("radius", [25, 50, 100])
def test_disk_geometry(radius):
    row = measure_single(disk_labels(radius=radius))

    assert_close(row["area_px2"], math.pi * radius**2, REL_TOL_AREA, "disk area")
    assert_close(row["major_axis_px"], 2 * radius, REL_TOL_AXIS, "disk major")
    assert_close(row["minor_axis_px"], 2 * radius, REL_TOL_AXIS, "disk minor")
    assert_close(row["equivalent_diameter_px"], 2 * radius, REL_TOL_AXIS, "disk equiv diam")
    assert_feret(row["feret_max_px"], 2 * radius, "disk feret")

    assert row["circularity"] == pytest.approx(1.0, abs=ABS_TOL_CIRCULARITY)
    assert row["aspect_ratio"] == pytest.approx(1.0, abs=0.02)
    assert row["eccentricity"] == pytest.approx(0.0, abs=0.05)
    assert row["solidity"] >= MIN_SOLIDITY_CONVEX


def test_disk_perimeter():
    radius = 50
    row = measure_single(disk_labels(radius=radius))
    assert_close(row["perimeter_px"], 2 * math.pi * radius, 0.02, "disk perimeter")


def test_equivalent_diameter_derives_from_area():
    """Equivalent circular diameter must be 2*sqrt(A/pi) of the reported area."""
    row = measure_single(disk_labels(radius=50))
    expected = 2 * math.sqrt(row["area_px2"] / math.pi)
    assert row["equivalent_diameter_px"] == pytest.approx(expected, rel=1e-12)


# --------------------------------------------------------------------------- #
# Ellipses
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("semi_major,semi_minor", [(60, 30), (100, 40), (100, 50)])
def test_ellipse_axes_and_ratios(semi_major, semi_minor):
    row = measure_single(ellipse_labels(semi_major=semi_major, semi_minor=semi_minor))

    assert_close(row["major_axis_px"], 2 * semi_major, REL_TOL_AXIS, "ellipse major")
    assert_close(row["minor_axis_px"], 2 * semi_minor, REL_TOL_AXIS, "ellipse minor")
    assert_close(row["area_px2"], math.pi * semi_major * semi_minor, REL_TOL_AREA, "ellipse area")

    expected_ar = semi_major / semi_minor
    assert_close(row["aspect_ratio"], expected_ar, REL_TOL_RATIO, "ellipse aspect ratio")

    expected_ecc = math.sqrt(1 - (semi_minor / semi_major) ** 2)
    assert_close(row["eccentricity"], expected_ecc, REL_TOL_RATIO, "ellipse eccentricity")


def test_ellipse_is_not_described_by_one_diameter():
    """§9: an elongated cell's major and minor axes must genuinely differ.

    Guards against a regression where both axes collapse toward the equivalent
    circular diameter, which would silently flatten cardiomyocyte morphology.
    """
    row = measure_single(ellipse_labels(semi_major=100, semi_minor=40))
    assert row["major_axis_px"] > row["equivalent_diameter_px"] > row["minor_axis_px"]
    assert row["aspect_ratio"] > 2.0


def test_ellipse_orientation_horizontal():
    """Major axis along columns reads as ~0 degrees from the +x axis."""
    row = measure_single(ellipse_labels(semi_major=100, semi_minor=40))
    assert row["orientation_deg"] == pytest.approx(0.0, abs=1.0)


# --------------------------------------------------------------------------- #
# Rectangles
# --------------------------------------------------------------------------- #


def test_rectangle_area_is_exact():
    """A rasterised rectangle has no discretisation error at all."""
    row = measure_single(rectangle_labels(height=80, width=160))
    assert row["area_px2"] == pytest.approx(80 * 160)


def test_rectangle_extent_and_solidity_are_one():
    """An axis-aligned rectangle exactly fills both its bbox and its convex hull."""
    row = measure_single(rectangle_labels(height=80, width=160))
    assert row["extent"] == pytest.approx(1.0, abs=1e-9)
    assert row["solidity"] == pytest.approx(1.0, abs=1e-9)


def test_rectangle_feret_is_the_diagonal():
    row = measure_single(rectangle_labels(height=80, width=160))
    assert_feret(row["feret_max_px"], math.hypot(80, 160), "rectangle feret")


# --------------------------------------------------------------------------- #
# Physical units - isotropic
# --------------------------------------------------------------------------- #


def test_isotropic_lengths_scale_linearly_and_areas_quadratically():
    scale = 0.25
    labels = ellipse_labels(semi_major=100, semi_minor=50)
    px = measure_single(labels)
    um = measure_single(labels, calibration_from_resolution(scale, scale))

    assert um["area_um2"] == pytest.approx(px["area_px2"] * scale**2)
    assert um["major_axis_um"] == pytest.approx(px["major_axis_px"] * scale)
    assert um["minor_axis_um"] == pytest.approx(px["minor_axis_px"] * scale)
    assert um["perimeter_um"] == pytest.approx(px["perimeter_px"] * scale)
    assert um["feret_max_um"] == pytest.approx(px["feret_max_px"] * scale)


def test_dimensionless_metrics_are_invariant_under_isotropic_scaling():
    """Aspect ratio, circularity, solidity and extent are pure shape."""
    labels = ellipse_labels(semi_major=100, semi_minor=50)
    px = measure_single(labels)
    um = measure_single(labels, calibration_from_resolution(0.25, 0.25))

    for column in ("aspect_ratio", "eccentricity", "circularity", "solidity", "extent"):
        assert um[column] == pytest.approx(px[column], rel=1e-9), column


def test_pixel_columns_survive_calibration():
    """§10: pixel-space measurements are always preserved alongside physical ones."""
    row = measure_single(
        ellipse_labels(semi_major=100, semi_minor=50),
        calibration_from_resolution(0.25, 0.25),
    )
    for column in ("area_px2", "perimeter_px", "major_axis_px", "minor_axis_px", "feret_max_px"):
        assert column in row.index and np.isfinite(row[column]), column


# --------------------------------------------------------------------------- #
# Physical units - anisotropic. The scalar shortcut fails these.
# --------------------------------------------------------------------------- #


def test_anisotropic_axes_are_recomputed_not_rescaled():
    """An ellipse under sx != sy changes shape, not just size.

    Semi-axes 100 (x) and 50 (y) at sx=0.5, sy=0.25 become 50 and 12.5 um,
    so the major axis is 100 um and the minor 25 um. Multiplying the pixel
    major axis by any single scalar cannot produce both.
    """
    calibration = calibration_from_resolution(0.5, 0.25)
    row = measure_single(ellipse_labels(semi_major=100, semi_minor=50), calibration)

    assert_close(row["major_axis_um"], 100.0, 0.01, "anisotropic major")
    assert_close(row["minor_axis_um"], 25.0, 0.01, "anisotropic minor")

    mean_scale = (0.5 + 0.25) / 2
    assert row["major_axis_um"] != pytest.approx(row["major_axis_px"] * mean_scale, rel=0.05)


def test_anisotropic_aspect_ratio_genuinely_changes():
    """Pixel-frame AR is 2.0; the physical shape is 4:1 and must be reported so."""
    calibration = calibration_from_resolution(0.5, 0.25)
    labels = ellipse_labels(semi_major=100, semi_minor=50)

    assert_close(measure_single(labels)["aspect_ratio"], 2.0, REL_TOL_RATIO, "pixel AR")
    assert_close(
        measure_single(labels, calibration)["aspect_ratio"], 4.0, REL_TOL_RATIO, "physical AR"
    )


def test_anisotropic_second_moments_match_hand_computed_transform():
    """Independent check of C' = S C S^T, bypassing skimage entirely.

    Scales the raw pixel coordinates, computes the covariance eigenvalues with
    plain numpy, and compares to the reported axes.
    """
    sx, sy = 0.5, 0.25
    labels = ellipse_labels(semi_major=100, semi_minor=50, rotation=math.radians(30))

    rows, cols = np.nonzero(labels == 1)
    scaled = np.column_stack([rows * sy, cols * sx])
    covariance = np.cov(scaled, rowvar=False, bias=True)
    eigenvalues = np.sort(np.linalg.eigvalsh(covariance))[::-1]
    expected_major = 4.0 * math.sqrt(eigenvalues[0])
    expected_minor = 4.0 * math.sqrt(eigenvalues[1])

    row = measure_single(labels, calibration_from_resolution(sx, sy))
    assert_close(row["major_axis_um"], expected_major, 0.01, "hand-computed major")
    assert_close(row["minor_axis_um"], expected_minor, 0.01, "hand-computed minor")


def test_anisotropic_perimeter_beats_the_scalar_shortcut():
    """The scaled-contour estimator lands near the analytic ellipse perimeter;
    multiplying the pixel perimeter by a mean scale does not."""
    sx, sy = 0.5, 0.25
    semi_major, semi_minor = 100, 50
    labels = ellipse_labels(semi_major=semi_major, semi_minor=semi_minor)

    a, b = semi_major * sx, semi_minor * sy
    analytic = math.pi * (3 * (a + b) - math.sqrt((3 * a + b) * (a + 3 * b)))

    row = measure_single(labels, calibration_from_resolution(sx, sy))
    error = abs(row["perimeter_um"] - analytic) / analytic
    assert error < 0.05, "scaled-contour perimeter off by {:.1%}".format(error)

    shortcut = row["perimeter_px"] * (sx + sy) / 2
    shortcut_error = abs(shortcut - analytic) / analytic
    assert shortcut_error > 0.10, (
        "the scalar shortcut is supposed to be badly wrong here; if it is not, "
        "this test no longer proves anything"
    )


def test_solidity_and_extent_are_invariant_under_anisotropy():
    """Numerator and denominator both scale by det(S), so the ratios do not move."""
    labels = ellipse_labels(semi_major=100, semi_minor=50)
    px = measure_single(labels)
    um = measure_single(labels, calibration_from_resolution(0.5, 0.25))
    assert um["solidity"] == pytest.approx(px["solidity"], rel=1e-9)
    assert um["extent"] == pytest.approx(px["extent"], rel=1e-9)


def test_perimeter_estimator_selection():
    assert perimeter_estimator_for(uncalibrated()) == "crofton_scaled"
    assert perimeter_estimator_for(calibration_from_resolution(0.25, 0.25)) == "crofton_scaled"
    assert perimeter_estimator_for(calibration_from_resolution(0.5, 0.25)) == "contour_scaled"


def test_contour_perimeter_reduces_to_pixel_length_at_unit_scale():
    mask = disk_labels(radius=50) == 1
    assert contour_perimeter(mask, 1.0, 1.0) == pytest.approx(contour_perimeter(mask))


# --------------------------------------------------------------------------- #
# Border status and unresolved objects
# --------------------------------------------------------------------------- #


def test_touches_border_detection():
    assert touches_border((0, 5, 10, 20), (100, 100)) is True
    assert touches_border((5, 0, 10, 20), (100, 100)) is True
    assert touches_border((5, 5, 100, 20), (100, 100)) is True
    assert touches_border((5, 5, 10, 100), (100, 100)) is True
    assert touches_border((5, 5, 10, 20), (100, 100)) is False


def test_border_flag_appears_in_output():
    labels = np.zeros((100, 100), dtype=np.int32)
    labels[0:30, 0:30] = 1
    row = measure_single(labels)
    assert bool(row["touches_border"]) is True


def test_unresolved_objects_are_absent_from_the_cell_table():
    """§12: no per-cell row may exist for a group we could not separate."""
    labels = disk_labels(radius=50)
    frame = measure_cells(
        labels,
        [ObjectRecord(1, 1, "unresolved_cluster", unresolved_reason="saddle too shallow")],
        [ClusterRecord(1, (1,), (), (1,))],
        uncalibrated(),
    )
    assert len(frame) == 0
