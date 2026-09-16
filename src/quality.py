"""Readiness checks and summaries with explicit experimental units."""
import numpy as np
import pandas as pd

from .batch import calibration_consistency
from .pipeline import compute_results
from .review import validate_review
from .workflow import is_stale


def readiness(batch):
    rows = []
    if not batch.images:
        rows.append(["Batch", "Action", "No images loaded"])
    for s in batch.images:
        validate_review(s)
        issues = []
        if s.error:
            issues.append("Segmentation failed: " + s.error)
        if not s.has_segmentation:
            issues.append("Not segmented")
        elif not s.objects:
            issues.append("No objects detected")
        if not s.calibration.is_calibrated:
            issues.append("Uncalibrated: measurements use pixels")
        if not s.reviewed:
            issues.append("Needs review")
        if s.has_segmentation and s.segmented_channel != s.channel:
            issues.append("Channel changed since segmentation")
        elif is_stale(s):
            issues.append("Settings changed since segmentation; re-run to apply them")
        if s.objects and len(s.qc.excluded_ids & {o.object_id for o in s.objects}) / len(s.objects) > .5:
            issues.append("More than half of objects excluded")
        if any(not o.is_resolved for o in s.included_objects()):
            issues.append("Unresolved clusters remain")
        if batch.overrides_for(s):
            issues.append("Settings differ from batch defaults")
        rows.extend([[s.display_name, "Check", issue] for issue in issues] or [[s.display_name, "Ready", "Checks passed"]])
    consistency = calibration_consistency(batch)
    if not consistency["consistent"]:
        rows.append(["Batch", "Action", consistency["reason"]])
    return pd.DataFrame(rows, columns=["Image", "Status", "Detail"])


#: Metrics offered for replicate-level comparison: (label, column stem, kind).
DESIGN_METRICS = {
    "area": ("Cell area", "area", "area"),
    "major_axis": ("Major axis", "major_axis", "length"),
    "minor_axis": ("Minor axis", "minor_axis", "length"),
    "feret_max": ("Max Feret diameter", "feret_max", "length"),
    "perimeter": ("Perimeter", "perimeter", "length"),
    "aspect_ratio": ("Aspect ratio", "aspect_ratio", "ratio"),
    "circularity": ("Circularity", "circularity", "ratio"),
    "solidity": ("Solidity", "solidity", "ratio"),
}

AGGREGATION_RULE = (
    "Cells from all fields of a well are pooled into a well mean. Well means are "
    "averaged with equal weight within each biological replicate, and replicate "
    "means with equal weight within each condition. Condition n is the number of "
    "biological replicates, never the number of cells. No significance tests are run."
)


def metric_column(metric: str, calibrated: bool) -> tuple[str, str]:
    """The cell-table column for a metric and the unit it is reported in."""
    _, stem, kind = DESIGN_METRICS[metric]
    if kind == "ratio":
        return stem, ""
    if kind == "area":
        return (stem + "_um2", "µm²") if calibrated else (stem + "_px2", "px²")
    return (stem + "_um", "µm") if calibrated else (stem + "_px", "px")


def _sd(values) -> float:
    values = pd.Series(values, dtype=float).dropna()
    return float(values.std(ddof=1)) if len(values) > 1 else float("nan")


def design_summary(batch, metric: str = "area") -> dict[str, pd.DataFrame]:
    """Image, well, biological-replicate and condition summaries of one metric.

    Returns frames keyed ``images``, ``wells``, ``replicates``, ``conditions``
    and ``unassigned``. Each level summarises the *means* of the level below,
    with its own n, so the replicate structure is explicit at every step. Images
    missing a well, condition or replicate label are listed in ``unassigned``
    and left out of the comparison rather than guessed at. Calibrated and
    uncalibrated images are never combined: ``unit`` is part of every key.
    """
    if metric not in DESIGN_METRICS:
        raise ValueError("Unknown metric {!r}.".format(metric))
    from .qc import exclusion_counts

    image_rows, unassigned = [], []
    for s in batch.images:
        validate_review(s)
        labelled = all(str(v).strip() for v in (s.well, s.condition, s.replicate))
        if not labelled or not s.has_segmentation:
            unassigned.append({
                "image": s.display_name, "well": s.well, "condition": s.condition,
                "biological_replicate": s.replicate,
                "reason": "not segmented" if not s.has_segmentation else "design labels missing",
            })
            continue
        results = compute_results(s)
        calibrated = s.calibration.is_calibrated
        column, unit = metric_column(metric, calibrated)
        values = pd.to_numeric(results.cells.get(column, pd.Series(dtype=float)), errors="coerce")
        image_rows.append({
            "condition": s.condition.strip(), "biological_replicate": s.replicate.strip(),
            "well": s.well.strip(), "image": s.display_name, "unit": unit or "ratio",
            "calibrated": calibrated, "reviewed": s.reviewed,
            "cells": int(values.notna().sum()),
            "unresolved_objects": int(results.counts.get("unresolved_accepted", 0)),
            "excluded_objects": int(results.counts.get("objects_excluded", 0)),
            **{k: v for k, v in exclusion_counts(s.objects, s.qc).items()},
            "cell_mean": float(values.mean()) if values.notna().any() else float("nan"),
            "cell_median": float(values.median()) if values.notna().any() else float("nan"),
            "_values": values.dropna().to_numpy(),
        })

    images = pd.DataFrame(image_rows)
    empty = {
        "images": pd.DataFrame(columns=["condition", "biological_replicate", "well", "image",
                                        "unit", "cells", "cell_mean", "cell_median"]),
        "wells": pd.DataFrame(columns=["condition", "biological_replicate", "well", "unit",
                                       "fields", "cells", "mean", "median", "sd_cells"]),
        "replicates": pd.DataFrame(columns=["condition", "biological_replicate", "unit", "wells",
                                            "cells", "mean", "median_of_wells", "sd_between_wells"]),
        "conditions": pd.DataFrame(columns=["condition", "unit", "biological_replicates", "wells",
                                            "cells", "mean", "median_of_replicates",
                                            "sd_between_replicates"]),
        "unassigned": pd.DataFrame(unassigned, columns=["image", "well", "condition",
                                                         "biological_replicate", "reason"]),
    }
    if images.empty:
        return empty

    well_rows = []
    for key, group in images.groupby(["condition", "biological_replicate", "well", "unit"], sort=True):
        pooled = np.concatenate(list(group["_values"])) if len(group) else np.array([])
        well_rows.append({
            "condition": key[0], "biological_replicate": key[1], "well": key[2], "unit": key[3],
            "fields": len(group), "cells": int(len(pooled)),
            "mean": float(pooled.mean()) if len(pooled) else float("nan"),
            "median": float(np.median(pooled)) if len(pooled) else float("nan"),
            "sd_cells": _sd(pooled),
            "all_reviewed": bool(group["reviewed"].all()),
            "unresolved_objects": int(group["unresolved_objects"].sum()),
        })
    wells = pd.DataFrame(well_rows)

    rep_rows = []
    for key, group in wells.groupby(["condition", "biological_replicate", "unit"], sort=True):
        means = group["mean"].dropna()
        rep_rows.append({
            "condition": key[0], "biological_replicate": key[1], "unit": key[2],
            "wells": len(group), "cells": int(group["cells"].sum()),
            "mean": float(means.mean()) if len(means) else float("nan"),
            "median_of_wells": float(means.median()) if len(means) else float("nan"),
            "sd_between_wells": _sd(means),
            "all_reviewed": bool(group["all_reviewed"].all()),
        })
    replicates = pd.DataFrame(rep_rows)

    cond_rows = []
    for key, group in replicates.groupby(["condition", "unit"], sort=True):
        means = group["mean"].dropna()
        cond_rows.append({
            "condition": key[0], "unit": key[1],
            "biological_replicates": int(len(means)),
            "wells": int(group["wells"].sum()), "cells": int(group["cells"].sum()),
            "mean": float(means.mean()) if len(means) else float("nan"),
            "median_of_replicates": float(means.median()) if len(means) else float("nan"),
            "sd_between_replicates": _sd(means),
            "all_reviewed": bool(group["all_reviewed"].all()),
        })
    conditions = pd.DataFrame(cond_rows)

    return {
        "images": images.drop(columns=["_values"]),
        "wells": wells,
        "replicates": replicates,
        "conditions": conditions,
        "unassigned": empty["unassigned"],
    }


def experimental_summary(batch):
    """Pool fields within wells, average wells within replicates, then compare replicates.

    Units stay separate; missing design labels do not become invented replicates.
    """
    rows = []
    for s in batch.images:
        if not s.has_segmentation or not all([s.well.strip(), s.condition.strip(), s.replicate.strip()]):
            continue
        cells = compute_results(s).cells
        if cells.empty:
            continue
        unit = "um2" if s.calibration.is_calibrated else "px2"
        for value in cells["area_" + unit]:
            rows.append([s.condition, s.replicate, s.well, unit, value])
    data = pd.DataFrame(rows, columns=["condition", "replicate", "well", "area_unit", "area"])
    keys = ["condition", "replicate", "well", "area_unit"]
    wells = data.groupby(keys, as_index=False).agg(cell_count=("area", "size"), mean_area=("area", "mean"))
    reps = wells.groupby(["condition", "replicate", "area_unit"], as_index=False).agg(
        well_count=("well", "size"), mean_area=("mean_area", "mean"))
    conditions = reps.groupby(["condition", "area_unit"], as_index=False).agg(
        biological_replicates=("replicate", "size"), mean_area=("mean_area", "mean"),
        sd_between_replicates=("mean_area", "std"))
    return wells, reps, conditions
