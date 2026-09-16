"""Frozen copies of implementations that were later optimised.

These are the oracles for ``test_golden_equivalence.py``: each is the code as it
stood before the performance pass, kept verbatim apart from imports, so the
optimised versions can be shown to produce identical output rather than merely
plausible output. Do not "fix" or speed these up -- their value is that they
do not change.
"""

from __future__ import annotations

import math
from typing import Any, Iterable, Sequence

import numpy as np
import pandas as pd
from PIL import Image
from scipy import ndimage as ndi
from skimage import measure as skmeasure
from skimage.morphology import binary_closing, binary_dilation, disk
from skimage.segmentation import find_boundaries

from src.morphometry import (
    _isotropic_physical,
    _member_summary,
    _safe_div,
    contour_perimeter,
    measure_region,
)
from src.types import Calibration, ClusterRecord, ObjectRecord
from src.visualize import (
    CATEGORY_COLOURS,
    UNRESOLVED_BOUNDARY_WIDTH,
    _draw_ids,
    _resize_labels,
    categorise,
)

# --------------------------------------------------------------------------- #
# morphometry.measure_clusters
# --------------------------------------------------------------------------- #


def _cluster_mask(labels: np.ndarray, member_ids: Sequence[int]) -> np.ndarray:
    return np.isin(labels, np.asarray(member_ids, dtype=labels.dtype))


def _cluster_geometry(mask, calibration, contact_distance_px) -> dict[str, Any]:
    n_components = int(skmeasure.label(mask, connectivity=2).max())
    closed = False
    boundary_mask = mask
    if n_components > 1 and contact_distance_px > 0:
        radius = max(1, int(math.ceil(contact_distance_px / 2)))
        boundary_mask = binary_closing(mask, disk(radius))
        closed = True

    union_label = mask.astype(np.int32)
    prop_px = skmeasure.regionprops(union_label)[0]
    prop_scaled = None
    if calibration.is_calibrated and not calibration.is_isotropic:
        sx, sy = calibration.scales
        prop_scaled = skmeasure.regionprops(union_label, spacing=(sy, sx))[0]

    if calibration.is_calibrated and calibration.is_isotropic:
        measured = measure_region(prop_px, Calibration(), None)
        measured["perimeter_px"] = float(skmeasure.perimeter_crofton(boundary_mask))
        measured["_recompute_circularity"] = True
        _isotropic_physical(measured, calibration)
    else:
        measured = measure_region(prop_px, calibration, prop_scaled)
        if calibration.is_calibrated:
            sx, sy = calibration.scales
            measured["perimeter_px"] = float(skmeasure.perimeter_crofton(boundary_mask))
            measured["perimeter_um"] = contour_perimeter(boundary_mask, sy=sy, sx=sx)
        else:
            measured["perimeter_px"] = float(skmeasure.perimeter_crofton(boundary_mask))

    measured.pop("_recompute_circularity", None)
    if calibration.is_calibrated:
        measured["circularity"] = _safe_div(
            4.0 * math.pi * measured["area_um2"], measured["perimeter_um"] ** 2
        )
    else:
        measured["circularity"] = _safe_div(
            4.0 * math.pi * measured["area_px2"], measured["perimeter_px"] ** 2
        )

    min_row, min_col, max_row, max_col = prop_px.bbox
    bbox_px = float((max_row - min_row) * (max_col - min_col))
    measured["bbox_area_px2"] = bbox_px
    if calibration.is_calibrated:
        sx, sy = calibration.scales
        measured["bbox_area_um2"] = bbox_px * sx * sy

    measured["n_mask_components"] = n_components
    measured["geometry_closed"] = closed
    return measured


def measure_clusters(labels, objects, clusters, calibration, cell_df, contact_distance_px=2):
    clusters = list(clusters)
    rows: list[dict[str, Any]] = []
    for cluster in clusters:
        mask = _cluster_mask(labels, cluster.member_ids)
        if not mask.any():
            continue
        geometry = _cluster_geometry(mask, calibration, contact_distance_px)
        members = (
            cell_df[cell_df["cell_id"].isin(list(cluster.resolved_ids))]
            if len(cell_df)
            else cell_df
        )
        row: dict[str, Any] = {
            "cluster_id": cluster.cluster_id,
            "cluster_status": cluster.status,
            "n_resolved_cells": cluster.n_resolved_cells,
            "n_unresolved_objects": cluster.n_unresolved_objects,
            "cell_count": cluster.cell_count if cluster.cell_count is not None else pd.NA,
            "cluster_size": cluster.cluster_size,
        }
        for key, value in geometry.items():
            if key in ("n_mask_components", "geometry_closed"):
                row[key] = value
            else:
                row[f"cluster_{key}"] = value
        row.update(_member_summary(members, calibration))
        row["constituent_object_ids"] = ";".join(str(i) for i in cluster.member_ids)
        rows.append(row)
    frame = pd.DataFrame(rows)
    return frame.sort_values("cluster_id").reset_index(drop=True) if len(frame) else frame


# --------------------------------------------------------------------------- #
# clustering.build_cell_contact_graph
# --------------------------------------------------------------------------- #


def build_cell_contact_graph(labels, contact_distance_px=2):
    labels = np.asarray(labels)
    height, width = labels.shape[:2]
    radius = int(contact_distance_px)
    footprint = disk(radius) if radius > 0 else None
    edges: set[tuple[int, int]] = set()
    for prop in skmeasure.regionprops(labels):
        label = int(prop.label)
        min_row, min_col, max_row, max_col = prop.bbox
        pad = radius + 1
        r0, c0 = max(0, min_row - pad), max(0, min_col - pad)
        r1, c1 = min(height, max_row + pad), min(width, max_col + pad)
        crop = labels[r0:r1, c0:c1]
        own = crop == label
        reach = binary_dilation(own, footprint) if footprint is not None else own
        for neighbour in np.unique(crop[reach]):
            neighbour = int(neighbour)
            if neighbour == 0 or neighbour == label:
                continue
            edges.add((min(label, neighbour), max(label, neighbour)))
    return sorted(edges)


# --------------------------------------------------------------------------- #
# segmentation._drop_small_labels
# --------------------------------------------------------------------------- #


def drop_small_labels(labels, min_area):
    labels = np.asarray(labels).copy()
    counts = np.bincount(labels.ravel())
    for value, count in enumerate(counts):
        if value != 0 and count < min_area:
            labels[labels == value] = 0
    return labels


# --------------------------------------------------------------------------- #
# nuclei
# --------------------------------------------------------------------------- #


def nuclear_edges(labels, gap=0):
    if gap:
        return build_cell_contact_graph(labels, int(gap) + 1)
    pairs = set()
    for dy, dx in [(0, 1), (1, 0), (1, 1), (1, -1)]:
        a = labels[:labels.shape[0]-dy, max(0, -dx):labels.shape[1]-max(0, dx)]
        b = labels[dy:, max(0, dx):labels.shape[1]-max(0, -dx)]
        good = (a > 0) & (b > 0) & (a != b)
        for x, y in zip(a[good], b[good]):
            pairs.add(tuple(sorted((int(x), int(y)))))
    return sorted(pairs)


def nuclear_seed_and_filter(mask, distance, coordinates, min_area):
    """The seeding and small-object loop from ``segment_nuclei``."""
    from skimage.segmentation import watershed

    markers = np.zeros(mask.shape, np.int32)
    for i, (y, x) in enumerate(coordinates, 1):
        markers[y, x] = i
    components, n = ndi.label(mask)
    next_id = len(coordinates) + 1
    for component in range(1, n + 1):
        region = components == component
        if not markers[region].any():
            y, x = np.unravel_index(np.argmax(np.where(region, distance, -1)), mask.shape)
            markers[y, x] = next_id
            next_id += 1
    labels = watershed(-distance, markers, mask=mask).astype(np.int32)
    for value, area in zip(*np.unique(labels, return_counts=True)):
        if value and area < min_area:
            labels[labels == value] = 0
    return labels


# --------------------------------------------------------------------------- #
# visualize.make_overlay
# --------------------------------------------------------------------------- #


def make_overlay(display_rgb, labels, objects: Iterable, clusters: Iterable, alpha=0.28,
                 show_cell_ids=True, show_cluster_ids=False, excluded_ids: Iterable[int] = ()):
    objects = list(objects)
    clusters = list(clusters)
    base = np.asarray(display_rgb).astype(np.float32).copy()
    if base.ndim == 2:
        base = np.repeat(base[:, :, None], 3, axis=2)
    small = _resize_labels(np.asarray(labels), base.shape[:2])
    categories = categorise(objects, clusters)
    for category, colour in CATEGORY_COLOURS.items():
        ids = [oid for oid, cat in categories.items() if cat == category]
        if not ids:
            continue
        member = np.isin(small, np.asarray(ids, dtype=small.dtype))
        if not member.any():
            continue
        tint = np.asarray(colour, dtype=np.float32)
        base[member] = (1.0 - alpha) * base[member] + alpha * tint
        edge = find_boundaries(np.where(member, small, 0), mode="inner")
        if category == "unresolved" and UNRESOLVED_BOUNDARY_WIDTH > 1:
            edge = binary_dilation(
                edge, np.ones((UNRESOLVED_BOUNDARY_WIDTH, UNRESOLVED_BOUNDARY_WIDTH), bool)
            ) & member
        base[edge] = tint
    excluded = np.isin(small, list(excluded_ids))
    if excluded.any():
        base[excluded] *= 0.35
        edge = find_boundaries(np.where(excluded, small, 0), mode="inner")
        base[edge] = (160, 160, 160)
    image = Image.fromarray(np.clip(base, 0, 255).astype(np.uint8))
    if show_cell_ids or show_cluster_ids:
        _draw_ids(image, small, objects, clusters, categories, show_cell_ids, show_cluster_ids)
    return np.asarray(image)


__all__ = [
    "ClusterRecord", "ObjectRecord", "build_cell_contact_graph", "drop_small_labels",
    "make_overlay", "measure_clusters", "nuclear_edges", "nuclear_seed_and_filter",
]
