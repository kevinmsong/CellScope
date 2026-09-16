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

import threading
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from contextlib import nullcontext
from dataclasses import dataclass, field, replace
from time import perf_counter
from typing import Any

import numpy as np
import pandas as pd

from .clustering import build_clusters
from .morphometry import measure_cells, measure_clusters
from .preprocessing import preprocess_image
from .qc import apply_automatic_qc, apply_qc, exclusion_counts, qc_counts
from .scalebar import annotation_touching_ids, detect_annotation
from .segmentation import segment_cells, split_touching_cells
from .types import AnalysisSession, ClusterRecord, ObjectRecord
from .workflow import segmentation_fingerprint


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


@dataclass
class PreparedImage:
    """Stage 1 output: everything inference needs, captured from the session.

    Settings are snapshotted here, so an edit made while the image waits for
    the GPU cannot leak into the run, and the recorded provenance is exactly
    what was used.
    """

    session: AnalysisSession
    image: np.ndarray
    nuclear: np.ndarray | None
    preprocess_params: Any
    segmentation_params: Any
    split_params: Any
    channel: str
    fingerprint: str
    pixels: np.ndarray | None
    timings: dict[str, float] = field(default_factory=dict)


@dataclass
class SegmentationOutcome:
    """Stage 3 output: the finished masks, not yet attached to the session."""

    raw_labels: np.ndarray
    object_labels: np.ndarray
    objects: list[ObjectRecord]
    annotation: Any
    engine_info: dict[str, Any]
    channel: str
    fingerprint: str
    params: tuple


def prepare_segmentation(session: AnalysisSession, image_2d=None, nuclear_2d=None) -> PreparedImage:
    """Read the channels and preprocess them. CPU only; touches no session state."""
    from .image_io import extract_channel

    started = perf_counter()
    if image_2d is None:
        if session.image is None:
            raise ValueError("No image loaded.")
        image_2d = extract_channel(session.image, session.channel)
        nuclear_2d = nuclear_array_for(session)
    params = session.preprocess_params
    prepared = preprocess_image(image_2d, params)
    prepared_nuclear = preprocess_image(nuclear_2d, params) if nuclear_2d is not None else None
    return PreparedImage(
        session=session,
        image=prepared,
        nuclear=prepared_nuclear,
        preprocess_params=params,
        segmentation_params=session.segmentation_params,
        split_params=session.split_params,
        channel=session.channel,
        fingerprint=segmentation_fingerprint(session),
        pixels=session.image.pixels if session.image is not None else None,
        timings={"preprocessing": perf_counter() - started},
    )


def infer_segmentation(prepared: PreparedImage, engine: str = "auto"):
    """Run the segmenter. The only stage that uses the GPU."""
    started = perf_counter()
    raw_labels, engine_info = segment_cells(
        prepared.image, prepared.segmentation_params, engine=engine, nuclear=prepared.nuclear
    )
    prepared.timings["segmentation"] = perf_counter() - started
    return raw_labels, engine_info


def finalize_segmentation(prepared: PreparedImage, raw_labels, engine_info) -> SegmentationOutcome:
    """Split touching cells and find burned-in annotations. CPU only."""
    started = perf_counter()
    object_labels, objects = split_touching_cells(
        raw_labels, prepared.split_params, prepared.segmentation_params.min_object_area
    )
    # The annotation is found on the original image, never on the preprocessed
    # channel, and its detection changes no pixel that is segmented or measured.
    annotation = detect_annotation(prepared.pixels) if prepared.pixels is not None else None
    objects = flag_annotation_contact(object_labels, objects, annotation)

    raw_labels = np.asarray(raw_labels)
    raw_labels.flags.writeable = False
    object_labels.flags.writeable = False     # copy-on-write; see review.freeze_labels
    prepared.timings["splitting"] = perf_counter() - started
    timings = {key: round(value, 3) for key, value in prepared.timings.items()}
    timings["total"] = round(sum(prepared.timings.values()), 3)
    return SegmentationOutcome(
        raw_labels=raw_labels,
        object_labels=object_labels,
        objects=objects,
        annotation=annotation,
        engine_info={**engine_info, "timings_seconds": timings},
        channel=prepared.channel,
        fingerprint=prepared.fingerprint,
        params=(prepared.preprocess_params, prepared.segmentation_params, prepared.split_params),
    )


def commit_segmentation(session: AnalysisSession, outcome: SegmentationOutcome) -> AnalysisSession:
    """Attach finished masks to the session and reset everything derived from them."""
    session.raw_labels = outcome.raw_labels
    session.object_labels = outcome.object_labels
    session.objects = outcome.objects
    session.annotation = outcome.annotation
    session.engine_info = outcome.engine_info
    session.qc.reset()
    apply_automatic_qc(session.qc, outcome.objects, session.qc_params)
    session.reviewed = False
    session.undo_stack.clear()
    session.redo_stack.clear()
    session.correction_points.clear()
    session.selected_object = 0
    session.segmented_channel = outcome.channel
    session.segmented_with = outcome.fingerprint
    session.segmentation_version += 1
    session.results_cache = None
    session.error = ""
    return session


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
    prepared = prepare_segmentation(session, image_2d, nuclear_2d)
    raw_labels, engine_info = infer_segmentation(prepared, engine)
    return commit_segmentation(session, finalize_segmentation(prepared, raw_labels, engine_info))


def flag_annotation_contact(labels, objects, annotation) -> list[ObjectRecord]:
    """Set ``touches_annotation`` on every object from the current mask."""
    touching = annotation_touching_ids(labels, annotation)
    return [
        o if o.touches_annotation == (o.object_id in touching)
        else replace(o, touches_annotation=o.object_id in touching)
        for o in objects
    ]


def reapply_automatic_qc(session: AnalysisSession) -> dict[str, int]:
    """Re-run the automatic exclusion rules after the QC settings changed.

    Masks are untouched, and manual decisions -- exclusions and restorations --
    are kept.
    """
    if not session.has_segmentation:
        return {"excluded": 0, "restored": 0}
    return apply_automatic_qc(session.qc, session.objects, session.qc_params)


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
    prepared = prepare_segmentation(session)
    raw_labels, engine_info = infer_segmentation(prepared, engine)
    return commit_segmentation(session, finalize_segmentation(prepared, raw_labels, engine_info))


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


#: CPU threads that prepare and finalize images around GPU inference. With
#: Cellpose on a 4 GB card inference is >95% of the time, and the measured gain
#: is small but real on large images; see docs/PERFORMANCE.md.
DEFAULT_CPU_WORKERS = 2

#: Weight of the latest image in the running per-image time used for the ETA.
ETA_SMOOTHING = 0.3


class BatchCancelled(Exception):
    """Raised inside a stage when the run was cancelled."""


class BatchReport(dict):
    """The running tally of a batch run.

    ``status`` lists ``[filename, state]`` per target image, in order. The
    timing keys are measurements of this run, so two reports describing the
    same outcome compare equal whatever their timings.
    """

    TIMING_KEYS = frozenset({"seconds_per_image", "eta_seconds", "elapsed_seconds"})

    def _outcome(self):
        return {k: v for k, v in self.items() if k not in self.TIMING_KEYS}

    def __eq__(self, other):
        if isinstance(other, BatchReport):
            return self._outcome() == other._outcome()
        return dict.__eq__(self, other)

    __hash__ = None

    def set_status(self, index: int, state: str) -> None:
        self["status"][index][1] = state


def _new_report(targets, skipped: int) -> BatchReport:
    return BatchReport(
        total=len(targets),
        succeeded=0,
        failed=[],
        skipped=skipped,
        cancelled=False,
        status=[[s.display_name, "queued"] for s in targets],
        seconds_per_image=None,
        eta_seconds=None,
        elapsed_seconds=0.0,
    )


def _record_failure(session: AnalysisSession, report, index: int, error: BaseException) -> None:
    session.error = "{}: {}".format(type(error).__name__, error)
    session.raw_labels = None
    session.object_labels = None
    session.objects = []
    session.annotation = None
    session.results_cache = None
    report["failed"].append((session.display_name, session.error))
    report.set_status(index, "failed")


def iter_batch_segmentation(
    batch, engine: str = "auto", only_missing: bool = False,
    cancel: threading.Event | None = None, workers: int | None = None,
):
    """Segment a batch, yielding after each image.

    Yields ``(position, total, session, report)`` once per image actually run,
    in input order, where ``report`` is the running tally with a per-image
    ``status``, the smoothed seconds per image, and an ETA.

    Stages are pipelined when ``workers`` > 0: while the GPU runs image *n*, a
    CPU thread prepares image *n+1* and another finalizes image *n-1*. GPU
    inference stays strictly serial, and results are committed on this thread
    in input order under the batch lock, so what a run produces does not
    depend on thread timing. At most three images are held in memory.

    One image failing must not cost the rest: its error is recorded on that
    image and reported. ``cancel`` stops the run between stages; images that
    were already committed keep their results, and nothing is half-written.

    ``only_missing`` skips images that already carry a segmentation, so adding
    one field to a batch of twelve does not re-run the eleven already done.
    """
    targets, skipped = _batch_targets(batch, only_missing)
    report = _new_report(targets, skipped)
    cancel = cancel or threading.Event()
    workers = DEFAULT_CPU_WORKERS if workers is None else int(workers)
    started = perf_counter()
    last_commit = started

    def finished(position, session):
        nonlocal last_commit
        now = perf_counter()
        spent, last_commit = now - last_commit, now
        previous = report["seconds_per_image"]
        report["seconds_per_image"] = (
            spent if previous is None else ETA_SMOOTHING * spent + (1 - ETA_SMOOTHING) * previous
        )
        report["elapsed_seconds"] = now - started
        report["eta_seconds"] = report["seconds_per_image"] * (len(targets) - position - 1)
        return position, len(targets), session, report

    if workers <= 0:
        for position, session in enumerate(targets):
            if cancel.is_set():
                report["cancelled"] = True
                break
            report.set_status(position, "segmenting")
            try:
                with batch_lock(batch):
                    prepared = prepare_segmentation(session)
                raw, info = infer_segmentation(prepared, engine)
                outcome = finalize_segmentation(prepared, raw, info)
                with batch_lock(batch):
                    commit_segmentation(session, outcome)
                report["succeeded"] += 1
                report.set_status(position, "done")
            except Exception as error:
                with batch_lock(batch):
                    _record_failure(session, report, position, error)
            yield finished(position, session)
        return

    cpu = ThreadPoolExecutor(max_workers=workers, thread_name_prefix="cellscope-cpu")
    gpu = ThreadPoolExecutor(max_workers=1, thread_name_prefix="cellscope-gpu")

    def guarded(stage, *args):
        if cancel.is_set():
            raise BatchCancelled()
        return stage(*args)

    def prepare(session):
        # Reading settings must not race an edit on the interface thread.
        with batch_lock(batch):
            return prepare_segmentation(session)

    prepared_futures: dict[int, Any] = {}
    inference: tuple[int, Any] | None = None
    finalizing: dict[int, Any] = {}
    next_to_start = 0
    next_to_commit = 0
    try:
        while next_to_commit < len(targets):
            if cancel.is_set():
                report["cancelled"] = True
                break
            # Keep one image prepared ahead of the GPU.
            for index in (next_to_start, next_to_start + 1):
                if index < len(targets) and index not in prepared_futures:
                    prepared_futures[index] = cpu.submit(guarded, prepare, targets[index])
            # Start inference on the next image once the GPU is free.
            if inference is None and next_to_start < len(targets):
                future = prepared_futures[next_to_start]
                if future.done():
                    report.set_status(next_to_start, "segmenting")
                    if future.exception() is None:
                        inference = (next_to_start, gpu.submit(
                            guarded, infer_segmentation, future.result(), engine))
                    else:
                        finalizing[next_to_start] = future
                    next_to_start += 1
            # Hand a finished inference to a CPU thread for splitting.
            if inference is not None and inference[1].done():
                index, future = inference
                inference = None
                prepared = prepared_futures[index].result()
                if future.exception() is None:
                    report.set_status(index, "postprocessing")
                    raw, info = future.result()
                    finalizing[index] = cpu.submit(
                        guarded, finalize_segmentation, prepared, raw, info)
                else:
                    finalizing[index] = future
                continue
            # Commit, in input order, whatever is ready.
            ready = finalizing.get(next_to_commit)
            if ready is not None and ready.done():
                session = targets[next_to_commit]
                del finalizing[next_to_commit]
                prepared_futures.pop(next_to_commit, None)
                error = ready.exception()
                with batch_lock(batch):
                    if error is None:
                        commit_segmentation(session, ready.result())
                        report["succeeded"] += 1
                        report.set_status(next_to_commit, "done")
                    elif isinstance(error, BatchCancelled):
                        report["cancelled"] = True
                        break
                    else:
                        _record_failure(session, report, next_to_commit, error)
                yield finished(next_to_commit, session)
                next_to_commit += 1
                continue
            waiting = [f for f in (
                *prepared_futures.values(), *finalizing.values(),
                *([inference[1]] if inference else []),
            ) if not f.done()]
            if waiting:
                wait(waiting, timeout=0.25, return_when=FIRST_COMPLETED)
    finally:
        # Cancelled, finished, or abandoned by the caller: stop starting work.
        # A running inference cannot be interrupted, but its result is never
        # committed.
        if next_to_commit < len(targets):
            cancel.set()
        cpu.shutdown(wait=False, cancel_futures=True)
        gpu.shutdown(wait=False, cancel_futures=True)
        for index in range(next_to_commit, len(targets)):
            if report["status"][index][1] not in ("done", "failed"):
                report.set_status(index, "cancelled")


def batch_lock(batch):
    """The batch's re-entrant lock, or a no-op for objects without one."""
    getter = getattr(batch, "lock", None)
    return getter() if callable(getter) else nullcontext()


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
    report = _new_report(targets, skipped)
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


def results_key(session: AnalysisSession):
    """A key that changes whenever :func:`compute_results` would change.

    Other caches (overlays, tables, batch pools) build on this so that they can
    never outlive the measurements they display.
    """
    return (*_cache_key(session), id(session.object_labels))


def _geometry_caches(session: AnalysisSession, use_cache: bool):
    """Per-cell and per-cluster geometry caches for the current mask.

    Geometry depends on the mask, the calibration and (for clusters) the
    contact distance -- not on which *other* objects are excluded. The cache is
    keyed on the label array itself (held by reference, so its identity cannot
    be recycled) as well as the segmentation version.
    """
    if not use_cache:
        return None, None
    key = (
        session.segmentation_version, session.calibration,
        session.cluster_params.contact_distance_px,
    )
    cached = session.geometry_cache
    if cached is not None and cached[0] == key and cached[1] is session.object_labels:
        return cached[2], cached[3]
    cells: dict = {}
    clusters: dict = {}
    session.geometry_cache = (key, session.object_labels, cells, clusters)
    return cells, clusters


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
    qc_labels.flags.writeable = False
    included = session.included_objects()
    cell_cache, cluster_cache = _geometry_caches(session, use_cache)

    clusters, edges = build_clusters(
        qc_labels, included, session.cluster_params.contact_distance_px
    )
    cells = measure_cells(qc_labels, included, clusters, session.calibration, cache=cell_cache)
    cluster_summary = measure_clusters(
        qc_labels,
        included,
        clusters,
        session.calibration,
        cells,
        session.cluster_params.contact_distance_px,
        cache=cluster_cache,
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


def excluded_cell_measurements(session: AnalysisSession) -> pd.DataFrame:
    """Diagnostic geometry for excluded, resolved objects.

    Shown during review so a researcher can see *why* a cell looked wrong. These
    rows never enter an accepted-cell summary. They reuse the geometry cache,
    which is valid because a cell's geometry depends only on its own mask.
    """
    if session.object_labels is None:
        return pd.DataFrame()
    excluded = [o for o in session.objects if not session.qc.is_included(o.object_id)]
    if not excluded:
        return pd.DataFrame()
    cell_cache, _ = _geometry_caches(session, True)
    return measure_cells(session.object_labels, excluded, [], session.calibration,
                         cache=cell_cache)


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
        **exclusion_counts(session.objects, session.qc),
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
