"""Batch-level aggregation.

A batch is the images of one experiment, spanning its wells, and its result is a
single **pooled** population: every accepted cell from every image in one set.

Pooling is weighted by cell count, not by image. An image contributing 40 cells
counts for more than one contributing 6, because they are fields of the same
experiment rather than independent experiments.

What that figure is *not*
-------------------------
It describes the batch. It is not a set of independent biological replicates.
Comparing conditions requires the well or the sample as the replicate unit --
CellScope deliberately does not compute that, and produces no p-values (§16).

The per-image summary is kept alongside as a **QC view**: one row per image is
what lets a reviewer spot the out-of-focus or over-segmented field before
trusting the pooled number.
"""

from __future__ import annotations

from typing import Any

import pandas as pd

from .pipeline import AnalysisResults, compute_results, results_key
from .qc import qc_log_dataframe
from .types import AnalysisSession, BatchSession

#: Metrics summarised over the pooled cells, as ``(name, kind)``. ``kind``
#: selects the source column and the unit suffix once the batch's calibration
#: is known: "area" -> _um2/_px2, "length" -> _um/_px, "ratio" -> no unit.
POOLED_METRICS = (
    ("area", "area"),
    ("major_axis", "length"),
    ("minor_axis", "length"),
    ("feret_max", "length"),
    ("aspect_ratio", "ratio"),
    ("circularity", "ratio"),
    ("solidity", "ratio"),
)


def _column_and_suffix(name: str, kind: str, calibrated: bool) -> tuple[str, str]:
    """Source column in the cell table, and the suffix its summary carries."""
    if kind == "ratio":
        return name, ""
    suffix = ("_um2" if calibrated else "_px2") if kind == "area" else (
        "_um" if calibrated else "_px"
    )
    return f"{name}{suffix}", suffix


# --------------------------------------------------------------------------- #
# Calibration consistency
# --------------------------------------------------------------------------- #


class CalibrationMismatch(ValueError):
    """Raised when a batch mixes calibrated and uncalibrated images.

    Pooling those together would put px and µm in one column. There is no
    sensible unit for the result, so no result is produced.
    """


def calibration_consistency(batch: BatchSession) -> dict[str, Any]:
    """Report whether the batch can be pooled, and in which units.

    Differing µm/px across images is fine -- each image's µm values are already
    correct in µm. Mixing calibrated with uncalibrated images is not.
    """
    segmented = [s for s in batch.images if s.has_segmentation]
    if not segmented:
        return {"consistent": True, "calibrated": False, "reason": "no segmented images"}

    calibrated = [s.calibration.is_calibrated for s in segmented]
    if all(calibrated):
        scales = {s.calibration.scales for s in segmented}
        return {
            "consistent": True,
            "calibrated": True,
            "distinct_scales": len(scales),
            "reason": "",
        }
    if not any(calibrated):
        return {"consistent": True, "calibrated": False, "distinct_scales": 0, "reason": ""}

    with_cal = [s.display_name for s in segmented if s.calibration.is_calibrated]
    without = [s.display_name for s in segmented if not s.calibration.is_calibrated]
    return {
        "consistent": False,
        "calibrated": False,
        "reason": (
            "The batch mixes calibrated and uncalibrated images, so its cells "
            "cannot be pooled into one column: {} calibrated ({}), {} "
            "uncalibrated ({}). Calibrate the remaining images, or clear the "
            "calibration, so the whole batch shares one unit.".format(
                len(with_cal), ", ".join(with_cal[:3]) + ("..." if len(with_cal) > 3 else ""),
                len(without), ", ".join(without[:3]) + ("..." if len(without) > 3 else ""),
            )
        ),
    }


def require_consistent(batch: BatchSession) -> dict[str, Any]:
    verdict = calibration_consistency(batch)
    if not verdict["consistent"]:
        raise CalibrationMismatch(verdict["reason"])
    return verdict


# --------------------------------------------------------------------------- #
# Per-image results
# --------------------------------------------------------------------------- #


def image_id_for(session: AnalysisSession) -> str:
    """Stable identifier for an image within a batch."""
    return session.display_name


def batch_key(batch: BatchSession) -> tuple:
    """Changes whenever any pooled table could change.

    Built from each image's results key plus the labels the tables carry, so a
    memoised table can never be served after an edit.
    """
    return tuple(
        (s.analysis_id, s.has_segmentation, results_key(s), s.well, s.reviewed, s.error,
         s.display_name)
        for s in batch.images
    )


def memoised(batch: BatchSession, name: str, compute):
    """Return ``compute()``, reusing the previous value while the batch is unchanged.

    Callers get the cached object itself, so they must treat it as read-only.
    """
    key = batch_key(batch)
    cache = batch.derived_cache
    entry = cache.get(name)
    if entry is not None and entry[0] == key:
        return entry[1]
    value = compute()
    cache[name] = (key, value)
    return value


def results_for(batch: BatchSession) -> list[tuple[AnalysisSession, AnalysisResults]]:
    """Compute (or reuse cached) results for every segmented image."""
    out = []
    for session in batch.images:
        if session.has_segmentation:
            out.append((session, compute_results(session)))
    return out


def _tag(frame: pd.DataFrame, session: AnalysisSession) -> pd.DataFrame:
    """Prefix a per-image frame with the columns that identify its origin."""
    if frame is None or not len(frame):
        return pd.DataFrame()
    tagged = frame.copy()
    tagged.insert(0, "image_id", image_id_for(session))
    tagged.insert(1, "well", session.well)
    return tagged


def pooled_cells(batch: BatchSession) -> pd.DataFrame:
    """Every accepted cell from every image, tagged with its origin.

    This is the batch population: the input to the headline statistics and to
    every distribution plot.
    """
    def compute():
        frames = [_tag(results.cells, session) for session, results in results_for(batch)]
        frames = [f for f in frames if len(f)]
        return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()

    return memoised(batch, "pooled_cells", compute)


def pooled_clusters(batch: BatchSession) -> pd.DataFrame:
    """Every cluster from every image, tagged with its origin."""
    def compute():
        frames = [
            _tag(results.cluster_summary, session) for session, results in results_for(batch)
        ]
        frames = [f for f in frames if len(f)]
        return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()

    return memoised(batch, "pooled_clusters", compute)


def per_image_summary(batch: BatchSession) -> pd.DataFrame:
    """One row per image -- the QC view, not a second result."""
    results = results_for(batch)
    rows = []
    for session, result in results:
        row = result.image_summary.copy()
        row.insert(0, "image_id", image_id_for(session))
        row.insert(1, "well", session.well)
        row.insert(2, "reviewed", session.reviewed)
        rows.append(row)
    if not rows:
        return pd.DataFrame()
    frame = pd.concat(rows, ignore_index=True)
    if "error" not in frame.columns:
        frame["error"] = [s.error for s, _ in results]
    return frame


def batch_qc_log(batch: BatchSession) -> pd.DataFrame:
    """Every image's QC actions, tagged with the image they belong to."""
    frames = []
    for session in batch.images:
        log = qc_log_dataframe(session.qc)
        if len(log):
            log = log.copy()
            log.insert(0, "image_id", image_id_for(session))
            frames.append(log)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


# --------------------------------------------------------------------------- #
# The pooled summary
# --------------------------------------------------------------------------- #


def _stat(series: pd.Series, function: str) -> float:
    values = pd.to_numeric(series, errors="coerce").dropna()
    if not len(values):
        return float("nan")
    if function == "sd":
        return float(values.std(ddof=1)) if len(values) > 1 else float("nan")
    return float(getattr(values, function)())


def batch_summary(batch: BatchSession) -> pd.DataFrame:
    """One row describing the whole batch, pooled across every image.

    Raises :class:`CalibrationMismatch` when the batch mixes calibrated and
    uncalibrated images -- there is no unit the pooled column could carry.
    """
    verdict = require_consistent(batch)
    calibrated = verdict["calibrated"]
    cells = pooled_cells(batch)
    clusters = pooled_clusters(batch)
    results = results_for(batch)

    row: dict[str, Any] = {
        "batch_id": batch.batch_id,
        "batch_label": batch.label,
        "n_images": len(batch.images),
        "n_images_segmented": len(results),
        "n_images_reviewed": batch.n_reviewed,
        "n_images_unreviewed": batch.n_unreviewed,
        "n_images_failed": len(batch.failed),
        # The pooled counts. n_cells is the number of cells; it is not a count
        # of independent replicates (see the module docstring).
        "n_cells": int(len(cells)),
        "n_clusters": int(len(clusters)),
        "wells": ";".join(sorted({s.well for s in batch.images if s.well})),
    }

    if len(clusters):
        status = clusters.get("cluster_status", pd.Series(dtype=str))
        row["isolated_cells"] = int((status == "isolated_cell").sum())
        row["multicell_clusters"] = int((status == "resolved_cluster").sum())
        row["unresolved_clusters"] = int(
            status.isin(["unresolved_cluster", "partially_resolved_cluster"]).sum()
        )
    else:
        row["isolated_cells"] = 0
        row["multicell_clusters"] = 0
        row["unresolved_clusters"] = 0

    for name, kind in POOLED_METRICS:
        column, suffix = _column_and_suffix(name, kind, calibrated)
        values = cells.get(column, pd.Series(dtype=float))
        row[f"mean_{name}{suffix}"] = _stat(values, "mean")
        row[f"median_{name}{suffix}"] = _stat(values, "median")
        row[f"sd_{name}{suffix}"] = _stat(values, "sd")

    row["calibration_consistent"] = verdict["consistent"]
    row["calibrated"] = calibrated
    row["distinct_calibrations"] = verdict.get("distinct_scales", 0)
    row["units"] = "um" if calibrated else "px"
    return pd.DataFrame([row])


def headline_counts(batch: BatchSession) -> dict[str, Any]:
    """Small dict for the status strip and the Results banner."""
    verdict = calibration_consistency(batch)
    cells = pooled_cells(batch) if verdict["consistent"] else pd.DataFrame()
    return {
        "n_images": len(batch.images),
        "n_segmented": len(batch.segmented),
        "n_reviewed": batch.n_reviewed,
        "n_unreviewed": batch.n_unreviewed,
        "n_cells": int(len(cells)),
        "consistent": verdict["consistent"],
        "reason": verdict["reason"],
    }
