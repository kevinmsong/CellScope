"""Morphology measurement for cells (§9) and clusters (§11), with correct
handling of physical units and anisotropic pixels (§10).

Coordinate frames
-----------------
Every *dimensional* quantity is reported twice: once in the pixel frame
(``*_px`` / ``*_px2``) and, when calibrated, once in the physical frame
(``*_um`` / ``*_um2``). Both are kept because under anisotropic pixels they are
**not** related by a scalar — recomputation in scaled coordinates is required,
and a reader must be able to see both.

*Dimensionless* shape descriptors (aspect ratio, eccentricity, circularity,
orientation) get a single column, computed in the physical frame when
calibrated and the pixel frame otherwise. Solidity and extent are genuinely
invariant under any axis scaling — the numerator and denominator both scale by
``det(S)`` — so their single column is exact in either frame.

Perimeter
---------
Perimeter is the one metric that cannot be rescaled by a scalar under
anisotropy, and ``skimage`` explicitly refuses to try
(``NotImplementedError: perimeter supports isotropic spacings only``). Two
estimators are therefore used, and which one ran is recorded in metadata:

``crofton_scaled``
    Isotropic pixels. The Crofton estimator in pixel space multiplied by the
    scale. Measured against an analytic ellipse this is accurate to ~0.5%.

``contour_scaled``
    Anisotropic pixels. Marching-squares contour with physically scaled
    coordinates. This carries the usual staircase overestimate of a few percent
    (~2.5% on a smooth ellipse), but the scalar shortcut it replaces was wrong
    by ~16% on the same shape.
"""

from __future__ import annotations

import math
from typing import Any, Iterable, Sequence

import numpy as np
import pandas as pd
from skimage import measure as skmeasure
from skimage.morphology import binary_closing, disk

from .types import Calibration, ClusterRecord, ObjectRecord, PerimeterEstimator

# --------------------------------------------------------------------------- #
# Small geometric helpers
# --------------------------------------------------------------------------- #


def touches_border(bbox: Sequence[int], shape: Sequence[int]) -> bool:
    """True when a region's bounding box reaches the image edge.

    Shared by segmentation (to tag :class:`ObjectRecord`) and by measurement, so
    both use one definition. ``bbox`` is skimage's ``(min_row, min_col,
    max_row, max_col)`` with exclusive maxima.
    """
    min_row, min_col, max_row, max_col = bbox[:4]
    return bool(
        min_row <= 0 or min_col <= 0 or max_row >= shape[0] or max_col >= shape[1]
    )


def contour_perimeter(mask: np.ndarray, sy: float = 1.0, sx: float = 1.0) -> float:
    """Total boundary length from marching-squares contours in scaled coordinates.

    Holes and disconnected components each contribute their own closed contour,
    which is the correct total boundary length for a compound region.
    """
    padded = np.pad(np.asarray(mask, dtype=float), 1)
    total = 0.0
    for contour in skmeasure.find_contours(padded, 0.5):
        pts = contour * np.array([sy, sx], dtype=float)
        if not np.allclose(pts[0], pts[-1]):
            pts = np.vstack([pts, pts[0]])
        deltas = np.diff(pts, axis=0)
        total += float(np.hypot(deltas[:, 0], deltas[:, 1]).sum())
    return total


def _safe_div(numerator: float, denominator: float) -> float:
    """Division that yields NaN rather than an exception or an infinity.

    Degenerate objects (a one-pixel-wide sliver has zero minor axis) must not
    produce ``inf``, which would silently poison downstream means.
    """
    if denominator is None or not np.isfinite(denominator) or abs(denominator) < 1e-12:
        return float("nan")
    return float(numerator) / float(denominator)


def _orientation_deg(prop_orientation: float) -> float:
    """Degrees counter-clockwise from the +x (column) axis, wrapped to [-90, 90).

    ``skimage`` measures from the 0th (row) axis; this converts to the more
    familiar image-space convention.
    """
    angle = 90.0 - math.degrees(float(prop_orientation))
    return ((angle + 90.0) % 180.0) - 90.0


def perimeter_estimator_for(calibration: Calibration) -> PerimeterEstimator:
    return "crofton_scaled" if calibration.is_isotropic else "contour_scaled"


# --------------------------------------------------------------------------- #
# Single-region measurement
# --------------------------------------------------------------------------- #


def _pixel_frame(prop) -> dict[str, float]:
    """Dimensional and shape measurements in the pixel frame."""
    area = float(prop.area)
    perimeter = float(skmeasure.perimeter_crofton(prop.image))
    major = float(prop.axis_major_length)
    minor = float(prop.axis_minor_length)
    centroid_row, centroid_col = prop.centroid
    return {
        "area_px2": area,
        "perimeter_px": perimeter,
        "major_axis_px": major,
        "minor_axis_px": minor,
        "feret_max_px": float(prop.feret_diameter_max),
        "equivalent_diameter_px": 2.0 * math.sqrt(area / math.pi) if area > 0 else 0.0,
        "centroid_x_px": float(centroid_col),
        "centroid_y_px": float(centroid_row),
        "_aspect_ratio": _safe_div(major, minor),
        "_eccentricity": float(prop.eccentricity),
        "_circularity": _safe_div(4.0 * math.pi * area, perimeter**2),
        "_orientation_deg": _orientation_deg(prop.orientation),
    }


def _physical_frame(
    prop_px, prop_scaled, mask: np.ndarray, calibration: Calibration
) -> dict[str, float]:
    """Dimensional and shape measurements in the physical (µm) frame.

    ``prop_scaled`` is a ``regionprops`` computed with ``spacing=(sy, sx)``,
    which correctly recomputes the second-moment tensor, eccentricity, Feret
    diameter and centroid in scaled coordinates. Only perimeter has to be done
    by hand, because skimage refuses anisotropic spacing there.
    """
    sx, sy = calibration.scales
    area = float(prop_scaled.area)
    major = float(prop_scaled.axis_major_length)
    minor = float(prop_scaled.axis_minor_length)
    centroid_row, centroid_col = prop_scaled.centroid

    if calibration.is_isotropic:
        # Exactly equivalent to the spacing computation (verified), and lets us
        # use the more accurate Crofton estimator for perimeter.
        perimeter = float(skmeasure.perimeter_crofton(mask)) * sx
    else:
        perimeter = contour_perimeter(mask, sy=sy, sx=sx)

    return {
        "area_um2": area,
        "perimeter_um": perimeter,
        "major_axis_um": major,
        "minor_axis_um": minor,
        "feret_max_um": float(prop_scaled.feret_diameter_max),
        "equivalent_diameter_um": 2.0 * math.sqrt(area / math.pi) if area > 0 else 0.0,
        "centroid_x_um": float(centroid_col),
        "centroid_y_um": float(centroid_row),
        "_aspect_ratio": _safe_div(major, minor),
        "_eccentricity": float(prop_scaled.eccentricity),
        "_circularity": _safe_div(4.0 * math.pi * area, perimeter**2),
        "_orientation_deg": _orientation_deg(prop_scaled.orientation),
    }


def measure_region(
    prop_px, calibration: Calibration, prop_scaled=None
) -> dict[str, Any]:
    """Full measurement of one region in both frames.

    ``prop_px`` is a ``regionprops`` entry with no spacing; ``prop_scaled`` is
    the same region measured with ``spacing=(sy, sx)`` and is required whenever
    the calibration is anisotropic.
    """
    row: dict[str, Any] = {}
    pixel = _pixel_frame(prop_px)
    shape_source = pixel

    for key, value in pixel.items():
        if not key.startswith("_"):
            row[key] = value

    # Invariant under any axis scaling — identical in both frames.
    row["solidity"] = float(prop_px.solidity)
    row["extent"] = float(prop_px.extent)

    if calibration.is_calibrated:
        scaled = prop_scaled if prop_scaled is not None else prop_px
        physical = _physical_frame(prop_px, scaled, prop_px.image, calibration)
        for key, value in physical.items():
            if not key.startswith("_"):
                row[key] = value
        shape_source = physical

    # Dimensionless descriptors: reported in the measurement frame.
    row["aspect_ratio"] = shape_source["_aspect_ratio"]
    row["eccentricity"] = shape_source["_eccentricity"]
    row["circularity"] = shape_source["_circularity"]
    row["orientation_deg"] = shape_source["_orientation_deg"]
    return row


# --------------------------------------------------------------------------- #
# regionprops plumbing
# --------------------------------------------------------------------------- #


def _props_by_label(labels: np.ndarray, calibration: Calibration):
    """Return ``(px_props, scaled_props)`` keyed by label value.

    ``scaled_props`` is ``None`` for isotropic calibrations, where the physical
    frame is obtained by exact scalar scaling instead.
    """
    px = {int(p.label): p for p in skmeasure.regionprops(labels)}
    scaled = None
    if calibration.is_calibrated and not calibration.is_isotropic:
        sx, sy = calibration.scales
        scaled = {
            int(p.label): p
            for p in skmeasure.regionprops(labels, spacing=(sy, sx))
        }
    return px, scaled


def _isotropic_physical(row: dict[str, Any], calibration: Calibration) -> None:
    """Fill physical columns by exact scalar scaling (isotropic pixels only)."""
    sx, sy = calibration.scales
    row["area_um2"] = row["area_px2"] * sx * sy
    row["perimeter_um"] = row["perimeter_px"] * sx
    row["major_axis_um"] = row["major_axis_px"] * sx
    row["minor_axis_um"] = row["minor_axis_px"] * sx
    row["feret_max_um"] = row["feret_max_px"] * sx
    row["equivalent_diameter_um"] = row["equivalent_diameter_px"] * sx
    row["centroid_x_um"] = row["centroid_x_px"] * sx
    row["centroid_y_um"] = row["centroid_y_px"] * sy


# --------------------------------------------------------------------------- #
# Cell-level measurement (§9)
# --------------------------------------------------------------------------- #

#: Order of the cell table. Physical columns are appended only when calibrated.
CELL_PIXEL_COLUMNS = [
    "area_px2",
    "perimeter_px",
    "major_axis_px",
    "minor_axis_px",
    "feret_max_px",
    "equivalent_diameter_px",
    "centroid_x_px",
    "centroid_y_px",
]
CELL_PHYSICAL_COLUMNS = [
    "area_um2",
    "perimeter_um",
    "major_axis_um",
    "minor_axis_um",
    "feret_max_um",
    "equivalent_diameter_um",
    "centroid_x_um",
    "centroid_y_um",
]
CELL_SHAPE_COLUMNS = [
    "aspect_ratio",
    "eccentricity",
    "solidity",
    "extent",
    "circularity",
    "orientation_deg",
]


def present_labels(labels: np.ndarray) -> set[int]:
    """Every non-zero label value in an image, in one counting pass."""
    labels = np.asarray(labels)
    if not labels.size:
        return set()
    counts = np.bincount(labels.ravel())
    return {int(v) for v in np.flatnonzero(counts) if v}


def _cell_geometry(prop, labels_shape, calibration, scaled) -> dict[str, Any]:
    """Every per-cell measurement that depends on the cell's own mask alone."""
    if calibration.is_calibrated and calibration.is_isotropic:
        measured = measure_region(prop, Calibration(), None)  # pixel frame
        _isotropic_physical(measured, calibration)
    else:
        measured = measure_region(prop, calibration, scaled)
    measured["touches_border"] = touches_border(prop.bbox, labels_shape)
    return measured


def measure_cells(
    labels: np.ndarray,
    objects: Iterable[ObjectRecord],
    clusters: Iterable[ClusterRecord],
    calibration: Calibration,
    cache: dict[int, dict[str, Any]] | None = None,
) -> pd.DataFrame:
    """One row per **resolved** object (§9).

    Objects with ``status == 'unresolved_cluster'`` are structurally absent:
    §12 forbids per-cell values for a group we could not separate, so there is
    no row for them to occupy.

    ``cache`` maps object ID to its measured geometry. A cell's geometry comes
    from its own mask alone (``regionprops`` crops each label to its bounding
    box), so it is unchanged when *other* cells are excluded, and a caller can
    reuse it across QC edits. The caller owns the cache and must discard it
    whenever the mask or the calibration changes -- see
    :func:`src.pipeline.compute_results`.
    """
    objects = list(objects)
    cluster_of: dict[int, ClusterRecord] = {}
    for cluster in clusters:
        for member in cluster.member_ids:
            cluster_of[member] = cluster

    present = present_labels(labels)
    wanted = [o for o in objects if o.is_resolved and o.object_id in present]
    missing = [o.object_id for o in wanted if cache is None or o.object_id not in cache]

    geometry: dict[int, dict[str, Any]] = {} if cache is None else cache
    if missing:
        px_props, scaled_props = _props_by_label(labels, calibration)
        for object_id in missing:
            geometry[object_id] = _cell_geometry(
                px_props[object_id], labels.shape, calibration,
                scaled_props.get(object_id) if scaled_props else None,
            )

    rows: list[dict[str, Any]] = []
    for obj in wanted:
        cluster = cluster_of.get(obj.object_id)
        row: dict[str, Any] = {
            "cell_id": obj.object_id,
            "cluster_id": cluster.cluster_id if cluster else pd.NA,
            "cluster_size": cluster.cluster_size if cluster else pd.NA,
            "cluster_status": cluster.status if cluster else pd.NA,
            "cell_status": "isolated"
            if cluster and cluster.status == "isolated_cell"
            else "clustered",
            "segmentation_status": obj.status,
        }
        row.update(geometry[obj.object_id])
        row["touches_annotation"] = obj.touches_annotation
        row["split_from"] = obj.split_from if obj.split_from is not None else pd.NA
        row["raw_label"] = obj.raw_label if obj.raw_label is not None else pd.NA
        rows.append(row)

    columns = (
        ["cell_id", "cluster_id", "cluster_size", "cluster_status", "cell_status",
         "segmentation_status"]
        + CELL_PIXEL_COLUMNS
        + (CELL_PHYSICAL_COLUMNS if calibration.is_calibrated else [])
        + CELL_SHAPE_COLUMNS
        + ["touches_border", "touches_annotation", "split_from", "raw_label"]
    )
    frame = pd.DataFrame(rows, columns=columns)
    return frame.sort_values("cell_id").reset_index(drop=True) if len(frame) else frame


# --------------------------------------------------------------------------- #
# Cluster-level measurement (§11)
# --------------------------------------------------------------------------- #

#: Per-cell metrics summarised across the resolved members of a cluster.
#: The ``member_`` prefix keeps §11's two levels distinct: cluster morphology is
#: not the mean morphology of the cells inside it.
_MEMBER_METRICS_PIXEL = {
    "area": "area_px2",
    "major": "major_axis_px",
    "minor": "minor_axis_px",
    "feret": "feret_max_px",
}
_MEMBER_METRICS_PHYSICAL = {
    "area": "area_um2",
    "major": "major_axis_um",
    "minor": "minor_axis_um",
    "feret": "feret_max_um",
}
_MEMBER_METRICS_SHAPE = {
    "aspect_ratio": "aspect_ratio",
    "circularity": "circularity",
    "solidity": "solidity",
}


def _cluster_mask(labels: np.ndarray, member_ids: Sequence[int]) -> np.ndarray:
    return np.isin(labels, np.asarray(member_ids, dtype=labels.dtype))


def _closing_radius(contact_distance_px: int) -> int:
    return max(1, int(math.ceil(contact_distance_px / 2))) if contact_distance_px > 0 else 0


def _cluster_window(slices, member_ids, shape, pad: int):
    """Bounding window of a cluster's members, padded and clipped to the image.

    ``slices`` is ``ndimage.find_objects`` output. Returns ``None`` when no
    member is present in the label image.
    """
    boxes = [
        slices[i - 1] for i in member_ids
        if 0 < i <= len(slices) and slices[i - 1] is not None
    ]
    if not boxes:
        return None
    r0 = max(0, min(b[0].start for b in boxes) - pad)
    c0 = max(0, min(b[1].start for b in boxes) - pad)
    r1 = min(shape[0], max(b[0].stop for b in boxes) + pad)
    c1 = min(shape[1], max(b[1].stop for b in boxes) + pad)
    return r0, c0, r1, c1


def _offset_geometry(measured: dict[str, Any], r0: int, c0: int, calibration) -> None:
    """Move crop-relative centroids back into whole-image coordinates."""
    measured["centroid_x_px"] += c0
    measured["centroid_y_px"] += r0
    if calibration.is_calibrated:
        sx, sy = calibration.scales
        measured["centroid_x_um"] += c0 * sx
        measured["centroid_y_um"] += r0 * sy


def _cluster_geometry_cropped(
    labels: np.ndarray, member_ids, window, calibration: Calibration,
    contact_distance_px: int,
) -> dict[str, Any]:
    """:func:`_cluster_geometry` on a window of the image instead of all of it.

    Every quantity is translation-invariant except the centroid, which is moved
    back afterwards. The window is padded by twice the closing radius plus
    one, so the morphological closing sees the same neighbourhood it would on
    the full image; where the window is clipped, it is clipped by the image
    edge, exactly as the full-image computation is. ``test_golden_equivalence``
    holds this to the full-image result.
    """
    r0, c0, r1, c1 = window
    crop = labels[r0:r1, c0:c1]
    mask = _cluster_mask(crop, member_ids)
    geometry = _cluster_geometry(mask, calibration, contact_distance_px)
    if calibration.is_calibrated and calibration.is_isotropic:
        # Physical centroids were derived from the crop-relative pixel ones.
        sx, sy = calibration.scales
        geometry["centroid_x_px"] += c0
        geometry["centroid_y_px"] += r0
        geometry["centroid_x_um"] = geometry["centroid_x_px"] * sx
        geometry["centroid_y_um"] = geometry["centroid_y_px"] * sy
    else:
        _offset_geometry(geometry, r0, c0, calibration)
    return geometry


def _cluster_geometry(
    mask: np.ndarray, calibration: Calibration, contact_distance_px: int
) -> dict[str, Any]:
    """Geometry of a cluster's union mask.

    ``area`` is always the exact union pixel count — never inflated by the
    morphological closing. Closing is applied only to make boundary-derived
    metrics (perimeter, circularity) well defined when a non-zero contact
    distance grouped members that do not literally touch, and the fact that it
    happened is reported alongside.
    """
    n_components = int(skmeasure.label(mask, connectivity=2).max())
    closed = False
    boundary_mask = mask
    if n_components > 1 and contact_distance_px > 0:
        radius = max(1, int(math.ceil(contact_distance_px / 2)))
        boundary_mask = binary_closing(mask, disk(radius))
        closed = True

    # regionprops over the whole union: a single label spanning every member,
    # so moments, hull and bbox describe the cluster as one object even when
    # the union is not connected.
    union_label = mask.astype(np.int32)
    prop_px = skmeasure.regionprops(union_label)[0]
    prop_scaled = None
    if calibration.is_calibrated and not calibration.is_isotropic:
        sx, sy = calibration.scales
        prop_scaled = skmeasure.regionprops(union_label, spacing=(sy, sx))[0]

    if calibration.is_calibrated and calibration.is_isotropic:
        measured = measure_region(prop_px, Calibration(), None)
        # Recompute perimeter on the closed mask, then scale.
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

    # Circularity must follow the perimeter actually used.
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


def _member_summary(
    member_rows: pd.DataFrame, calibration: Calibration
) -> dict[str, Any]:
    """Mean / median / SD across the resolved members of one cluster (§11).

    Returns NaN throughout when there are no resolved members — §12's rule that
    an unresolved group gets no fabricated per-cell statistics. SD is NaN for a
    single member (ddof=1), which is the honest answer rather than 0.
    """
    metrics = dict(_MEMBER_METRICS_PHYSICAL if calibration.is_calibrated else _MEMBER_METRICS_PIXEL)
    unit = "um" if calibration.is_calibrated else "px"
    area_unit = "um2" if calibration.is_calibrated else "px2"

    summary: dict[str, Any] = {}
    for name, column in metrics.items():
        suffix = area_unit if name == "area" else unit
        values = (
            member_rows[column].astype(float)
            if len(member_rows) and column in member_rows
            else pd.Series(dtype=float)
        )
        summary[f"member_{name}_mean_{suffix}"] = float(values.mean()) if len(values) else float("nan")
        if name == "area":
            summary[f"member_{name}_median_{suffix}"] = (
                float(values.median()) if len(values) else float("nan")
            )
            summary[f"member_{name}_sd_{suffix}"] = (
                float(values.std(ddof=1)) if len(values) > 1 else float("nan")
            )

    for name, column in _MEMBER_METRICS_SHAPE.items():
        values = (
            member_rows[column].astype(float)
            if len(member_rows) and column in member_rows
            else pd.Series(dtype=float)
        )
        summary[f"member_{name}_mean"] = float(values.mean()) if len(values) else float("nan")
    return summary


class _MemberSummaries:
    """:func:`_member_summary` for many clusters from one extraction of the table.

    Selecting rows out of a DataFrame once per cluster dominated the cost of
    crowded images. The statistics are identical: NaN-skipping mean and median,
    and a sample SD that is NaN below two members, exactly as pandas computes
    them for :func:`_member_summary`.
    """

    def __init__(self, cell_df: pd.DataFrame, calibration: Calibration):
        self.calibrated = calibration.is_calibrated
        metrics = _MEMBER_METRICS_PHYSICAL if self.calibrated else _MEMBER_METRICS_PIXEL
        self.metrics = dict(metrics)
        self.unit = "um" if self.calibrated else "px"
        self.area_unit = "um2" if self.calibrated else "px2"
        self.position: dict[int, int] = {}
        self.columns: dict[str, np.ndarray] = {}
        if cell_df is not None and len(cell_df):
            self.position = {int(v): i for i, v in enumerate(cell_df["cell_id"].to_numpy())}
            for column in list(self.metrics.values()) + list(_MEMBER_METRICS_SHAPE.values()):
                if column in cell_df:
                    self.columns[column] = cell_df[column].to_numpy(dtype=float)

    def _values(self, column: str, index: np.ndarray) -> np.ndarray | None:
        if not len(index) or column not in self.columns:
            return None
        return self.columns[column][index]

    def for_members(self, resolved_ids) -> dict[str, Any]:
        index = np.asarray(
            sorted(self.position[i] for i in resolved_ids if i in self.position), dtype=np.intp
        )
        nan = float("nan")
        summary: dict[str, Any] = {}
        for name, column in self.metrics.items():
            suffix = self.area_unit if name == "area" else self.unit
            values = self._values(column, index)
            summary[f"member_{name}_mean_{suffix}"] = _nan_stat(values, "mean")
            if name == "area":
                summary[f"member_{name}_median_{suffix}"] = _nan_stat(values, "median")
                summary[f"member_{name}_sd_{suffix}"] = (
                    _nan_stat(values, "sd") if values is not None and len(values) > 1 else nan
                )
        for name, column in _MEMBER_METRICS_SHAPE.items():
            summary[f"member_{name}_mean"] = _nan_stat(self._values(column, index), "mean")
        return summary


def _nan_stat(values: np.ndarray | None, function: str) -> float:
    """pandas' skipna statistics on a float array; NaN when there is no data.

    Written out the way ``pandas.core.nanops`` computes them (sum over count;
    sample variance from squared deviations), without building a Series for
    every cluster.
    """
    if values is None or not len(values):
        return float("nan")
    finite = values[~np.isnan(values)]
    count = len(finite)
    if not count:
        return float("nan")
    if function == "median":
        return float(np.median(finite))
    mean = finite.sum(dtype=np.float64) / count
    if function == "mean":
        return float(mean)
    if count < 2:
        return float("nan")
    return float(np.sqrt(((mean - finite) ** 2).sum(dtype=np.float64) / (count - 1)))


def measure_clusters(
    labels: np.ndarray,
    objects: Iterable[ObjectRecord],
    clusters: Iterable[ClusterRecord],
    calibration: Calibration,
    cell_df: pd.DataFrame,
    contact_distance_px: int = 2,
    cache: dict[tuple[int, ...], dict[str, Any]] | None = None,
) -> pd.DataFrame:
    """One row per cluster (§11), including clusters that are wholly unresolved.

    Two groups of columns are kept deliberately distinct:

    * ``cluster_*`` — geometry of the cluster treated as one object.
    * ``member_*`` — summary statistics over its resolved constituent cells.

    They answer different questions and are never interchangeable.

    Geometry is computed on a window around each cluster rather than the whole
    image, which turns a cost of clusters x image area into one of total
    cluster area. ``cache`` maps a cluster's member IDs to its geometry; a
    cluster whose membership is unchanged by a QC edit keeps its entry. As with
    :func:`measure_cells`, the caller discards the cache when the mask, the
    calibration or the contact distance changes.
    """
    from scipy import ndimage as ndi

    clusters = list(clusters)
    rows: list[dict[str, Any]] = []
    labels = np.asarray(labels)
    slices = ndi.find_objects(labels) if labels.size and labels.max() > 0 else []
    pad = 2 * _closing_radius(contact_distance_px) + 1
    summaries = _MemberSummaries(cell_df, calibration)

    for cluster in clusters:
        window = _cluster_window(slices, cluster.member_ids, labels.shape, pad)
        if window is None:
            continue

        key = tuple(cluster.member_ids)
        geometry = cache.get(key) if cache is not None else None
        if geometry is None:
            geometry = _cluster_geometry_cropped(
                labels, cluster.member_ids, window, calibration, contact_distance_px
            )
            if cache is not None:
                cache[key] = geometry

        row: dict[str, Any] = {
            "cluster_id": cluster.cluster_id,
            "cluster_status": cluster.status,
            "n_resolved_cells": cluster.n_resolved_cells,
            "n_unresolved_objects": cluster.n_unresolved_objects,
            # NA whenever any member could not be resolved — §12.
            "cell_count": cluster.cell_count if cluster.cell_count is not None else pd.NA,
            "cluster_size": cluster.cluster_size,
        }

        for key, value in geometry.items():
            if key in ("n_mask_components", "geometry_closed"):
                row[key] = value
            else:
                row[f"cluster_{key}"] = value

        row.update(summaries.for_members(cluster.resolved_ids))
        # Provenance, not a summary statistic -- deliberately NOT prefixed
        # ``member_``, which is reserved for per-cell summaries.
        row["constituent_object_ids"] = ";".join(str(i) for i in cluster.member_ids)
        rows.append(row)

    frame = pd.DataFrame(rows)
    return frame.sort_values("cluster_id").reset_index(drop=True) if len(frame) else frame
