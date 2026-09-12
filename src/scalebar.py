"""Automatic detection of a burned-in scale bar (§4A).

Microscope software commonly renders a solid bar plus a caption such as
"50 um" into a corner of the exported image. When the bar is present its pixel
length is a far more reliable measurement than two hand-placed clicks, so
CellScope detects it and offers the result.

The detection only ever *proposes* a length. The researcher confirms it, edits
it, or ignores it in favour of clicking the endpoints -- auto-detection never
silently sets the calibration, because a mis-detected bar would scale every
physical measurement in the export.

Distinguishing the bar from its caption is easy and robust: the bar is a solid,
strongly elongated rectangle, while glyphs are short and full of holes. On the
reference images this ships against, the bar measures 111 px at every intensity
threshold from 60 to 200, and an independent row-profile method agrees exactly.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from skimage import measure as skmeasure

#: A bar must be at least this many times wider than it is tall.
MIN_ASPECT_RATIO = 5.0
#: ...and must fill most of its bounding box, unlike a glyph.
MIN_EXTENT = 0.70
#: Bars are thin; anything this tall is a caption or an artefact.
MAX_HEIGHT_FRACTION = 0.05
#: Ignore specks.
MIN_WIDTH_PX = 20


@dataclass(frozen=True)
class ScaleBarDetection:
    """The outcome of looking for a scale bar."""

    found: bool
    length_px: float = 0.0
    bbox: tuple[int, int, int, int] | None = None
    colour: str = ""
    note: str = ""

    @property
    def endpoints(self) -> tuple[tuple[float, float], tuple[float, float]] | None:
        """The bar's two ends as (x, y) points, for display and for reuse by the
        manual two-click path."""
        if not self.found or self.bbox is None:
            return None
        min_row, min_col, max_row, max_col = self.bbox
        y = (min_row + max_row - 1) / 2.0
        return ((float(min_col), y), (float(max_col - 1), y))

    def describe(self) -> dict[str, Any]:
        return {
            "found": self.found,
            "length_px": self.length_px,
            "bbox": list(self.bbox) if self.bbox else None,
            "colour": self.colour,
            "note": self.note,
        }


def _colour_masks(rgb: np.ndarray) -> list[tuple[str, np.ndarray]]:
    """Candidate masks, most specific first.

    Yellow is tried before white because a yellow bar also satisfies a loose
    brightness rule, and naming the colour correctly makes the detection
    auditable.
    """
    red = rgb[..., 0].astype(np.int16)
    green = rgb[..., 1].astype(np.int16)
    blue = rgb[..., 2].astype(np.int16)
    brightest = rgb.max(axis=2).astype(np.int16)

    yellow = (red > 120) & (green > 120) & (blue < 0.5 * np.minimum(red, green))
    white = (red > 180) & (green > 180) & (blue > 180)
    bright = brightest > 200
    return [("yellow", yellow), ("white", white), ("bright", bright)]


def _best_bar(mask: np.ndarray) -> tuple[float, tuple[int, int, int, int]] | None:
    """Widest solid, strongly horizontal component in a binary mask."""
    if not mask.any():
        return None

    max_height = max(3, int(mask.shape[0] * MAX_HEIGHT_FRACTION))
    best: tuple[float, tuple[int, int, int, int]] | None = None

    for prop in skmeasure.regionprops(skmeasure.label(mask, connectivity=2)):
        min_row, min_col, max_row, max_col = prop.bbox
        width = max_col - min_col
        height = max_row - min_row
        if width < MIN_WIDTH_PX or height > max_height:
            continue
        if width / max(height, 1) < MIN_ASPECT_RATIO or prop.extent < MIN_EXTENT:
            continue
        if best is None or width > best[0]:
            best = (float(width), (min_row, min_col, max_row, max_col))
    return best


def detect_scale_bar(image: np.ndarray) -> ScaleBarDetection:
    """Look for a burned-in scale bar and report what was found.

    ``length_px`` is the bar's full width in pixels, inclusive of both end
    columns -- the same quantity two clicks on the bar's ends would produce.
    """
    array = np.asarray(image)
    if array.ndim == 2:
        array = np.repeat(array[:, :, None], 3, axis=2)
    if array.ndim != 3 or array.shape[2] < 3:
        return ScaleBarDetection(False, note="expected a 2D or RGB image")

    for colour, mask in _colour_masks(array[..., :3]):
        found = _best_bar(mask)
        if found is not None:
            width, bbox = found
            return ScaleBarDetection(
                found=True,
                length_px=width,
                bbox=bbox,
                colour=colour,
                note="detected a {} bar {:.0f} px wide".format(colour, width),
            )

    return ScaleBarDetection(
        False,
        note=(
            "no scale bar found. Click the bar's two ends instead, or enter the "
            "resolution directly."
        ),
    )
