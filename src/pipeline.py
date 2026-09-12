"""Wiring the scientific steps into one analysis (§23).

This module holds the order of operations so that ``app.py`` stays a thin layer
of Gradio callbacks and the whole pipeline can be driven from a test without a
browser.

One ordering decision matters scientifically: clusters and every measurement are
computed on the **QC-approved** label image, not the raw one. Excluding a cell
can genuinely break a cluster apart, and reporting the pre-QC membership
alongside post-QC measurements would be wrong. The raw labels are kept intact
for provenance.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from .clustering import build_clusters
from .morphometry import measure_cells, measure_clusters
from .preprocessing import preprocess_image
from .qc import apply_qc, qc_counts, exclude_border_touching
from .segmentation import segment_cells, split_touching_cells
from .types import AnalysisSession, ClusterRecord, ObjectRecord


@dataclass
class AnalysisResults:
    """Everything derived from a QC-approved segmentation."""

    qc_labels: np.ndarray
    objects: list[ObjectRecord]
    clusters: list[ClusterRecord]
    contact_edges: list[tuple[int, int]]
    cells: pd.DataFrame
    cluster_summary: pd.DataFrame
    image_summary: pd.DataFrame
    counts: dict[str, int] = field(default_factory=dict)


def run_segmentation(
    session: AnalysisSession, image_2d, engine: str = "auto", nuclear_2d=None
) -> AnalysisSession:
    """Preprocess, segment, split, and store the result on the session.

    Runs only when explicitly called -- §6 requires that changing a parameter
    never silently re-segments.

    ``nuclear_2d`` is an optional second channel carrying nuclei. It guides the
    segmenter without ever being measured: morphology still comes from the
    chosen segmentation channel alone.
    """
    prepared = preprocess_image(image_2d, session.preprocess_params)
    prepared_nuclear = (
        preprocess_image(nuclear_2d, session.preprocess_params)
        if nuclear_2d is not None
        else None
    )
    raw_labels, engine_info = segment_cells(
        prepared, session.segmentation_params, engine=engine, nuclear=prepared_nuclear
    )

    object_labels, objects = split_touching_cells(
        raw_labels, session.split_params, session.segmentation_params.min_object_area
    )

    raw_labels = np.asarray(raw_labels)
    raw_labels.flags.writeable = False
    session.raw_labels = raw_labels
    session.object_labels = object_labels
    session.objects = objects
    session.engine_info = engine_info
    session.qc.reset()
    exclude_border_touching(session.qc, objects, reason="automatically excluded: touches image border")
    session.reviewed = False
    session.undo_stack.clear()
    session.redo_stack.clear()
    session.correction_points.clear()
    session.selected_object = 0
    session.segmented_channel = session.channel
    session.segmentation_version += 1
    session.results_cache = None
    session.error = ""
    return session


# --------------------------------------------------------------------------- #
# Resolving an image's inputs
# --------------------------------------------------------------------------- #


def nuclear_array_for(session: AnalysisSession):
    """The nuclear guide image for this session, or ``None``.

    Two sources are supported, chosen per image: a channel of the image itself,
    or a separate file attached to it. A separate file must match the image's
    dimensions -- otherwise the guidance would be spatially meaningless, and
    silently so.
    """
    from .image_io import extract_channel

    choice = session.segmentation_params.nuclear_channel
    if session.nuclear_image is not None:
        if (session.nuclear_image.height, session.nuclear_image.width) != (
            session.image.height,
            session.image.width,
        ):
            raise ValueError(
                "Nuclear file {} is {}x{} but {} is {}x{}. They must be the same "
                "size for the nuclei to line up with the cells.".format(
                    session.nuclear_image.filename,
                    session.nuclear_image.width,
                    session.nuclear_image.height,
                    session.image.filename,
                    session.image.width,
                    session.image.height,
                )
            )
        return extract_channel(session.nuclear_image, session.nuclear_image_channel)

    if choice in (None, "", "none"):
        return None
    return extract_channel(session.image, choice)


def segment_session(session: AnalysisSession, engine: str = "auto") -> AnalysisSession:
    """Segment one session from its own image and channel choices."""
    from .image_io import extract_channel

    if session.image is None:
        raise ValueError("No image loaded.")
    channel_image = extract_channel(session.image, session.channel)
    return run_segmentation(
        session, channel_image, engine=engine, nuclear_2d=nuclear_array_for(session)
    )


def _batch_targets(batch, only_missing: bool):
    """Which images a run will touch, and how many it skips.

    Shared by the generator and its wrapper so the two can never disagree about
    what was run.
    """
    targets = [
        session
        for session in batch.images
        if not (only_missing and session.has_segmentation and not session.error)
    ]
    return targets, len(batch.images) - len(targets)


def iter_batch_segmentation(
    batch, engine: str = "auto", only_missing: bool = False
):
    """Segment a batch, yielding after each image.

    Yields ``(position, total, session, report)`` once per image actually run,
    where ``report`` is the running tally. Yielding lets the interface show each
    result as it lands, so the researcher can begin reviewing image 1 while the
    rest are still on the GPU -- the only parallelism available here, since
    inference is 95% of the cost and does not batch.

    One image failing must not cost the other eleven: the error is recorded on
    that image and reported, rather than raised. Cellpose weights are cached
    module-level in ``src.segmentation``, so the model loads once for the whole
    batch however many images it holds.

    ``only_missing`` skips images that already carry a segmentation, so adding
    one field to a batch of twelve does not re-run the eleven already done.
    """
    targets, skipped = _batch_targets(batch, only_missing)
    total = len(targets)
    report: dict[str, Any] = {
        "total": total, "succeeded": 0, "failed": [], "skipped": skipped,
    }

    for position, session in enumerate(targets):
        try:
            segment_session(session, engine=engine)
            report["succeeded"] += 1
        except Exception as error:
            session.error = "{}: {}".format(type(error).__name__, error)
            session.raw_labels = None
            session.object_labels = None
            session.objects = []
            session.results_cache = None
            report["failed"].append((session.display_name, session.error))
        yield position, total, session, report


def run_batch_segmentation(
    batch, engine: str = "auto", progress=None, only_missing: bool = False
) -> dict[str, Any]:
    """Segment every image in a batch and return the report.

    A thin wrapper that drains :func:`iter_batch_segmentation`, for callers that
    want the whole run in one call rather than streamed.
    """
    # Seeded from the same target selection the generator uses, so a run where
    # every image is skipped still reports the skip count -- the generator never
    # yields in that case and cannot report it itself.
    targets, skipped = _batch_targets(batch, only_missing)
    report = {
        "total": len(targets), "succeeded": 0, "failed": [], "skipped": skipped,
    }
    for position, total, session, report in iter_batch_segmentation(
        batch, engine=engine, only_missing=only_missing
    ):
        if progress is not None:
            progress(
                position / max(total, 1),
                # The yield arrives once the image is finished, so this
                # reports what just completed rather than what is starting.
                desc="Segmented {} ({}/{})".format(
                    session.display_name, position + 1, total
                ),
            )
    if progress is not None:
        progress(1.0, desc="Done")
    return report


def _cache_key(session: AnalysisSession):
    """Everything ``compute_results`` reads. If any of it changes, so does the key.

    Both versions are counters that only ever increase, and the two parameter
    objects are frozen dataclasses compared by value, so a stale entry cannot
    match. Correctness here matters more than the recomputation it saves.
    """
    return (
        session.segmentation_version,
        session.qc.version,
        session.calibration,
        session.cluster_params,
    )


def compute_results(session: AnalysisSession, use_cache: bool = True) -> AnalysisResults:
    """Apply QC, rebuild clusters, and measure everything.

    Repeated calls with nothing changed reuse the previous result -- the UI asks
    for it several times per interaction, and redrawing labels should not
    re-measure every cell.
    """
    if session.object_labels is None:
        raise ValueError("Run segmentation before computing results.")

    key = _cache_key(session)
    if use_cache and session.results_cache is not None:
        cached_key, cached_value = session.results_cache
        if cached_key == key:
            return cached_value

    qc_labels = apply_qc(session.object_labels, session.qc)
    included = session.included_objects()

    clusters, edges = build_clusters(
        qc_labels, included, session.cluster_params.contact_distance_px
    )
    cells = measure_cells(qc_labels, included, clusters, session.calibration)
    cluster_summary = measure_clusters(
        qc_labels,
        included,
        clusters,
        session.calibration,
        cells,
        session.cluster_params.contact_distance_px,
    )
    counts = qc_counts(session.objects, session.qc)
    summary = build_image_summary(session, cells, cluster_summary, clusters, counts)

    results = AnalysisResults(
        qc_labels=qc_labels,
        objects=included,
        clusters=clusters,
        contact_edges=edges,
        cells=cells,
        cluster_summary=cluster_summary,
        image_summary=summary,
        counts=counts,
    )
    session.results_cache = (key, results)
    return results


def _stat(series, function: str) -> float:
    """A statistic over cells, or NaN when there is nothing to average.

    Never substitutes 0 for "no data" -- an empty mean is unknown, not zero.
    """
    values = pd.Series(series, dtype=float).dropna()
    if not len(values):
        return float("nan")
    if function == "sd":
        return float(values.std(ddof=1)) if len(values) > 1 else float("nan")
    return float(getattr(values, function)())


def build_image_summary(session, cells, cluster_summary, clusters, counts) -> pd.DataFrame:
    """One row describing the whole image (§14).

    Cluster-size statistics are computed over clusters with a **known** cell
    count. Unresolved clusters are reported as their own count rather than
    being folded in with an assumed size.
    """
    calibrated = session.calibration.is_calibrated
    area_column = "area_um2" if calibrated else "area_px2"
    major_column = "major_axis_um" if calibrated else "major_axis_px"
    area_unit = "um2" if calibrated else "px2"
    length_unit = "um" if calibrated else "px"

    multicell = [c for c in clusters if c.cell_count is not None and c.cell_count > 1]
    unresolved = [c for c in clusters if c.cell_count is None]
    known_sizes = pd.Series(
        [c.cell_count for c in clusters if c.cell_count is not None], dtype=float
    )
    clustered_cells = sum(
        c.n_resolved_cells for c in clusters if c.cluster_size > 1
    )

    row: dict[str, Any] = {
        "analysis_id": session.analysis_id,
        "filename": session.image.filename if session.image else "",
        "segmentation_channel": session.segmented_channel,
        "nuclear_channel": session.segmentation_params.nuclear_channel,
        "segmentation_engine": session.engine_info.get("engine", ""),
        "objects_detected": counts.get("objects_detected", 0),
        "objects_accepted": counts.get("objects_accepted", 0),
        "objects_excluded": counts.get("objects_excluded", 0),
        "cells_resolved": counts.get("cells_accepted", 0),
        "isolated_cells": sum(1 for c in clusters if c.status == "isolated_cell"),
        "cells_in_clusters": clustered_cells,
        "multicell_clusters": len(multicell),
        "unresolved_clusters": len(unresolved),
        "clusters_total": len(clusters),
    }

    row["mean_cell_area_" + area_unit] = _stat(cells.get(area_column, []), "mean")
    row["median_cell_area_" + area_unit] = _stat(cells.get(area_column, []), "median")
    row["sd_cell_area_" + area_unit] = _stat(cells.get(area_column, []), "sd")
    row["mean_major_axis_" + length_unit] = _stat(cells.get(major_column, []), "mean")
    row["median_major_axis_" + length_unit] = _stat(cells.get(major_column, []), "median")
    row["mean_cluster_size_cells"] = _stat(known_sizes, "mean")
    row["median_cluster_size_cells"] = _stat(known_sizes, "median")

    row["calibration_method"] = session.calibration.method
    row["um_per_px_x"] = session.calibration.um_per_px_x
    row["um_per_px_y"] = session.calibration.um_per_px_y
    row["calibration_summary"] = session.calibration.summary()
    return pd.DataFrame([row])
