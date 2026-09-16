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


def _colour_masks(rgb: np.ndarray, include_bright: bool = True) -> list[tuple[str, np.ndarray]]:
    """Candidate masks, most specific first.

    Yellow is tried before white because a yellow bar also satisfies a loose
    brightness rule, and naming the colour correctly makes the detection
    auditable. ``include_bright`` adds the loosest rule, which is only safe on
    a reference frame the researcher chose for its bar.
    """
    red = rgb[..., 0].astype(np.int16)
    green = rgb[..., 1].astype(np.int16)
    blue = rgb[..., 2].astype(np.int16)

    yellow = (red > 120) & (green > 120) & (blue < 0.5 * np.minimum(red, green))
    white = (red > 180) & (green > 180) & (blue > 180)
    masks = [("yellow", yellow), ("white", white)]
    if include_bright:
        masks.append(("bright", rgb.max(axis=2).astype(np.int16) > 200))
    return masks


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


# --------------------------------------------------------------------------- #
# Burned-in annotations on data images
# --------------------------------------------------------------------------- #
#
# Some exports carry the scale bar and its caption on the data image itself,
# usually in the bottom-right corner. The ruler is drawn over the cells, so an
# object touching it has hidden, truncated morphology -- the same reason
# border-touching objects are excluded -- and in the red channel the bright
# yellow bar and glyphs are segmented as objects themselves.
#
# The rules here are deliberately stricter than for a reference frame. A data
# image is full of bright structure, so only saturated yellow or (in a true
# colour image) white counts, the bar must sit in an outer band of the frame,
# and it must pass the same solidity and aspect tests. On the reference data
# set this finds every burned-in ruler and nothing else in 101 images.

#: The bar must lie within this fraction of the frame from its top or bottom.
ANNOTATION_EDGE_BAND = 0.25
#: Pixels around the annotation that also count as touching it. JPEG ringing
#: spreads glyph colour by a pixel or two, and a cell this close is occluded.
ANNOTATION_MARGIN_PX = 2
#: How far soft, JPEG-blurred annotation colour may extend past the glyphs.
ANNOTATION_GROWTH_PX = 8
#: Caption glyphs are searched for this many bar heights above and below it.
CAPTION_SEARCH_BAR_HEIGHTS = 8
#: ...and this far beyond the bar's ends, as a fraction of its length.
CAPTION_SEARCH_OVERHANG = 0.25


def _strict_colour_masks(rgb: np.ndarray) -> list[tuple[str, np.ndarray]]:
    red = rgb[..., 0].astype(np.int16)
    green = rgb[..., 1].astype(np.int16)
    blue = rgb[..., 2].astype(np.int16)
    masks = [("yellow", (red > 200) & (green > 200) & (blue < 80))]
    # A greyscale image stored as RGB has no white that a cell could not also be.
    if np.any(red != green) or np.any(green != blue):
        masks.append(("white", (red > 200) & (green > 200) & (blue > 200)))
    return masks


def _loose_mask(rgb: np.ndarray, colour: str) -> np.ndarray:
    """The same colour with JPEG-softened edges included."""
    red = rgb[..., 0].astype(np.int16)
    green = rgb[..., 1].astype(np.int16)
    blue = rgb[..., 2].astype(np.int16)
    if colour == "yellow":
        return (red > 110) & (green > 110) & (blue < 0.6 * np.minimum(red, green))
    return (red > 150) & (green > 150) & (blue > 150)


def _as_rgb8(image: np.ndarray) -> np.ndarray | None:
    array = np.asarray(image)
    if array.ndim != 3 or array.shape[2] < 3:
        return None
    rgb = array[..., :3]
    if rgb.dtype == np.uint8:
        return rgb
    finite = rgb[np.isfinite(rgb)] if rgb.dtype.kind == "f" else rgb
    top = float(finite.max()) if finite.size else 0.0
    if top <= 0:
        return None
    return np.clip(rgb.astype(np.float64) / top * 255.0, 0, 255).astype(np.uint8)


def _union(a, b):
    if a is None:
        return b
    return (min(a[0], b[0]), min(a[1], b[1]), max(a[2], b[2]), max(a[3], b[3]))


def detect_annotation(image: np.ndarray, margin_px: int = ANNOTATION_MARGIN_PX):
    """Find a burned-in scale bar and its caption in a data image.

    Returns an :class:`~src.types.AnnotationDetection`. Detection never changes
    the pixels that are segmented or measured; it only marks which objects are
    occluded by the annotation.
    """
    from scipy import ndimage as ndi

    from .types import AnnotationDetection

    rgb = _as_rgb8(image)
    if rgb is None:
        return AnnotationDetection(note="single-channel image: no coloured annotation to find")
    height, width = rgb.shape[:2]

    for colour, mask in _strict_colour_masks(rgb):
        found = _best_bar(mask)
        if found is None:
            continue
        length, bar = found
        r0, c0, r1, c1 = bar
        if not (r0 >= (1 - ANNOTATION_EDGE_BAND) * height or r1 <= ANNOTATION_EDGE_BAND * height):
            continue

        # Search a window around the bar for glyph-sized pieces of the same colour.
        bar_height = max(3, r1 - r0)
        reach = CAPTION_SEARCH_BAR_HEIGHTS * bar_height
        overhang = int(round(CAPTION_SEARCH_OVERHANG * (c1 - c0)))
        w0, w1 = max(0, r0 - reach), min(height, r1 + reach)
        v0, v1 = max(0, c0 - overhang), min(width, c1 + overhang)
        components, _ = ndi.label(mask[w0:w1, v0:v1], structure=np.ones((3, 3), bool))
        boxes = ndi.find_objects(components)
        keep = np.zeros(len(boxes) + 1, bool)
        caption = None
        for index, box in enumerate(boxes, start=1):
            if box is None:
                continue
            piece = (box[0].start + w0, box[1].start + v0, box[0].stop + w0, box[1].stop + v0)
            if piece == bar:
                keep[index] = True
                continue
            tall = piece[2] - piece[0]
            wide = piece[3] - piece[1]
            if 3 <= tall <= max(6 * bar_height, 0.1 * height) and wide <= 1.2 * (c1 - c0):
                keep[index] = True
                caption = _union(caption, piece)

        region = _union(caption, bar)
        pad = margin_px + ANNOTATION_GROWTH_PX
        b0, b1 = max(0, region[0] - pad), min(height, region[2] + pad)
        a0, a1 = max(0, region[1] - pad), min(width, region[3] + pad)

        # Place the kept components in the exclusion window.
        core = np.zeros((b1 - b0, a1 - a0), bool)
        kept = keep[components]
        top, left = max(b0, w0), max(a0, v0)
        bottom, right = min(b1, w1), min(a1, v1)
        core[top - b0 : bottom - b0, left - a0 : right - a0] = kept[
            top - w0 : bottom - w0, left - v0 : right - v0
        ]

        # Grow into JPEG-softened edges of the same colour (to convergence, but
        # only within this small window). The margin is applied when contact is
        # tested, so ``mask`` is the annotation itself.
        loose = _loose_mask(rgb[b0:b1, a0:a1], colour) | core
        region_mask = ndi.binary_propagation(core, mask=loose, structure=np.ones((3, 3), bool))

        note = "{} scale bar, {:.0f} px, at rows {}-{} and columns {}-{}{}".format(
            colour, length, r0, r1 - 1, c0, c1 - 1,
            "; caption found" if caption else "; no caption found",
        )
        return AnnotationDetection(
            found=True, colour=colour, bar_bbox=bar, caption_bbox=caption,
            mask_bbox=(b0, a0, b1, a1), mask=region_mask, margin_px=int(margin_px),
            bar_length_px=float(length), note=note,
        )

    return AnnotationDetection(note="no burned-in scale bar found")


def _disk(radius: int) -> np.ndarray:
    yy, xx = np.mgrid[-radius : radius + 1, -radius : radius + 1]
    return (yy**2 + xx**2) <= radius**2


#: An object at least this much annotation is the annotation, segmented.
ANNOTATION_OBJECT_FRACTION = 0.5


def annotation_touching_ids(labels: np.ndarray, annotation) -> set[int]:
    """IDs of objects the annotation occludes.

    An object is occluded when any of its pixels lies on the annotation or
    within ``margin_px`` of it. Segmentation on a channel that sees the ruler
    also produces objects that are mostly bar or glyph, and these can swallow
    a strip of a neighbouring cell; so objects within the margin of such an
    *annotation object* count as occluded too.
    """
    from scipy import ndimage as ndi

    if annotation is None or not annotation.found or annotation.mask is None:
        return set()
    labels = np.asarray(labels)
    height, width = labels.shape[:2]
    margin = int(annotation.margin_px)
    r0, c0, r1, c1 = annotation.mask_bbox
    core = np.zeros((r1 - r0, c1 - c0), bool)
    painted = annotation.mask[: r1 - r0, : c1 - c0]
    core[: painted.shape[0], : painted.shape[1]] = painted
    window = labels[r0:r1, c0:c1]
    near = ndi.binary_dilation(core, structure=_disk(margin)) if margin else core
    touching = {int(v) for v in np.unique(window[near]) if v}

    # Objects that are mostly annotation pass contact on to their neighbours.
    on_core = window[core]
    if on_core.size:
        overlap = np.bincount(on_core.ravel())
        candidates = [int(v) for v in np.flatnonzero(overlap) if v]
        if candidates:
            areas = np.bincount(labels.ravel(), minlength=max(candidates) + 1)
            glyphs = [v for v in candidates if overlap[v] >= ANNOTATION_OBJECT_FRACTION * areas[v]]
            slices = ndi.find_objects(labels, max_label=max(glyphs)) if glyphs else []
            for value in glyphs:
                box = slices[value - 1]
                g0, g1 = max(0, box[0].start - margin - 1), min(height, box[0].stop + margin + 1)
                h0, h1 = max(0, box[1].start - margin - 1), min(width, box[1].stop + margin + 1)
                local = labels[g0:g1, h0:h1]
                reach = ndi.binary_dilation(local == value, structure=_disk(max(margin, 1)))
                touching |= {int(v) for v in np.unique(local[reach]) if v}
    return touching
