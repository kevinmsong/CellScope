"""Readiness checks and summaries with explicit experimental units."""
import pandas as pd

from .batch import calibration_consistency
from .pipeline import compute_results
from .review import validate_review


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
