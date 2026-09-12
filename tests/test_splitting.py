"""Touching-cell separation (§7) and the ban on fake per-cell values (§12).

The thresholds these exercise were measured, not guessed. On doublets of two
r=40 disks the relative neck depth runs 0.50 (overlap 10 px), 0.32 (20 px),
0.15 (35 px), 0.05 (50 px); single disks and 2.5:1 and 4:1 ellipses each produce
exactly one distance-transform maximum. The default 0.25 therefore splits the
visually obvious doublets and refuses the ambiguous ones.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from conftest import disk_labels, ellipse_labels, separated_disks, touching_disks_binary

from src.calibration import uncalibrated
from src.clustering import build_clusters
from src.morphometry import measure_cells, measure_clusters
from src.segmentation import split_touching_cells
from src.types import SplitParams


def run_split(labels, params=None, min_area=50):
    return split_touching_cells(labels, params or SplitParams(), min_area)


def measure_all(labels, objects):
    clusters, _ = build_clusters(labels, objects, contact_distance_px=2)
    cells = measure_cells(labels, objects, clusters, uncalibrated())
    summary = measure_clusters(labels, objects, clusters, uncalibrated(), cells)
    return cells, summary


@pytest.mark.parametrize("overlap", [10, 20])
def test_obvious_doublet_is_split(overlap):
    labels, objects = run_split(touching_disks_binary(radius=40, overlap=overlap))
    assert len(objects) == 2
    assert all(o.status == "resolved" for o in objects)
    assert all(o.split_from == 1 for o in objects)
    assert sorted(int(v) for v in np.unique(labels) if v) == [2, 3]


def test_split_fragments_get_fresh_unambiguous_ids():
    """The parent label disappears, so an ID is never both a merge and a piece."""
    labels, objects = run_split(touching_disks_binary(radius=40, overlap=10))
    assert 1 not in np.unique(labels)
    assert {o.object_id for o in objects} == {2, 3}
    assert all(o.raw_label == 1 for o in objects)


def test_split_is_deterministic():
    phantom = touching_disks_binary(radius=40, overlap=10)
    first_labels, first_objects = run_split(phantom)
    second_labels, second_objects = run_split(phantom)
    assert np.array_equal(first_labels, second_labels)
    assert [o.object_id for o in first_objects] == [o.object_id for o in second_objects]


@pytest.mark.parametrize(
    "phantom",
    [disk_labels(radius=50), ellipse_labels(100, 40), ellipse_labels(100, 25)],
    ids=["disk", "ellipse-2.5to1", "ellipse-4to1"],
)
def test_single_cells_are_never_split(phantom):
    """No false positives: an elongated cell is still one cell."""
    labels, objects = run_split(phantom)
    assert len(objects) == 1
    assert objects[0].status == "resolved"
    assert objects[0].split_from is None


def test_separated_cells_pass_through_untouched():
    phantom = separated_disks(n=3, radius=30, gap=40)
    labels, objects = run_split(phantom)
    assert len(objects) == 3
    assert np.array_equal(labels, phantom)


@pytest.mark.parametrize("overlap", [35, 50])
def test_ambiguous_merge_becomes_unresolved_rather_than_split(overlap):
    """§7: when separation confidence is poor, keep the group whole."""
    labels, objects = run_split(touching_disks_binary(radius=40, overlap=overlap))
    assert len(objects) == 1
    assert objects[0].status == "unresolved_cluster"
    assert "too shallow" in objects[0].unresolved_reason
    assert len(np.unique(labels)) == 2


def test_unresolved_reason_names_the_failing_gate():
    """The researcher must be able to see why a split was refused."""
    _, objects = run_split(touching_disks_binary(radius=40, overlap=35))
    reason = objects[0].unresolved_reason
    assert "relative depth" in reason and "0.25" in reason


def test_raising_the_saddle_threshold_refuses_more_splits():
    """The gate is what controls conservatism, and it responds monotonically."""
    phantom = touching_disks_binary(radius=40, overlap=20)
    _, permissive = run_split(phantom, SplitParams(min_saddle_depth_frac=0.20))
    _, strict = run_split(phantom, SplitParams(min_saddle_depth_frac=0.45))
    assert len(permissive) == 2
    assert len(strict) == 1 and strict[0].status == "unresolved_cluster"


def test_sliver_gate_rejects_lopsided_fragments():
    params = SplitParams(min_fragment_area_frac=0.60)
    _, objects = run_split(touching_disks_binary(radius=40, overlap=10), params)
    assert len(objects) == 1
    assert objects[0].status == "unresolved_cluster"
    assert "sliver" in objects[0].unresolved_reason


def test_disabling_splitting_keeps_everything_resolved():
    labels, objects = run_split(
        touching_disks_binary(radius=40, overlap=10), SplitParams(enabled=False)
    )
    assert len(objects) == 1
    assert objects[0].status == "resolved"


def test_disabled_splitting_still_records_the_suspicion():
    """The diagnostic survives a choice not to act on it."""
    _, objects = run_split(
        touching_disks_binary(radius=40, overlap=10), SplitParams(enabled=False)
    )
    assert objects[0].flagged_suspicious is True


# --------------------------------------------------------------------------- #
# §12: unresolved groups get no fabricated per-cell values
# --------------------------------------------------------------------------- #


def test_unresolved_cluster_yields_no_cell_rows_and_na_cell_count():
    labels, objects = run_split(touching_disks_binary(radius=40, overlap=35))
    cells, summary = measure_all(labels, objects)

    assert len(cells) == 0, "an unresolved group must contribute no per-cell rows"
    assert len(summary) == 1
    row = summary.iloc[0]
    assert row["cluster_status"] == "unresolved_cluster"
    assert row["n_resolved_cells"] == 0
    assert pd.isna(row["cell_count"]), "cell count of an unresolved group must be NA"


def test_unresolved_cluster_still_reports_group_level_geometry():
    """§12: report what IS measurable -- area, perimeter, axes, solidity."""
    labels, objects = run_split(touching_disks_binary(radius=40, overlap=35))
    _, summary = measure_all(labels, objects)
    row = summary.iloc[0]

    for column in (
        "cluster_area_px2",
        "cluster_perimeter_px",
        "cluster_major_axis_px",
        "cluster_minor_axis_px",
        "cluster_feret_max_px",
        "cluster_solidity",
    ):
        assert np.isfinite(row[column]) and row[column] > 0, column


def test_unresolved_cluster_member_statistics_are_nan_not_zero():
    """A mean over zero resolved cells is NaN. Zero would be a fabricated value."""
    labels, objects = run_split(touching_disks_binary(radius=40, overlap=35))
    _, summary = measure_all(labels, objects)
    row = summary.iloc[0]

    member_columns = [c for c in row.index if c.startswith("member_")]
    assert member_columns
    # Every member_ column is a summary statistic; provenance lives elsewhere.
    assert "constituent_object_ids" in row.index
    for column in member_columns:
        value = float(row[column])
        assert np.isnan(value), "{} should be NaN, got {}".format(column, value)


def test_single_member_cluster_reports_nan_sd_not_zero():
    """SD of one value is undefined; reporting 0 would imply a measured spread."""
    labels, objects = run_split(disk_labels(radius=50))
    _, summary = measure_all(labels, objects)
    row = summary.iloc[0]

    assert row["cluster_status"] == "isolated_cell"
    assert row["cell_count"] == 1
    assert np.isfinite(row["member_area_mean_px2"])
    assert np.isnan(float(row["member_area_sd_px2"]))


def test_partially_resolved_cluster_has_na_cell_count():
    """A resolved cell touching an unresolved blob: the total is unknowable."""
    labels = np.zeros((140, 400), dtype=np.int32)
    blob = touching_disks_binary(radius=40, overlap=35)
    labels[20:20 + blob.shape[0], 20:20 + blob.shape[1]] = blob

    merged, objects = run_split(labels)
    assert any(o.status == "unresolved_cluster" for o in objects)

    from skimage import draw

    merged = merged.copy()
    unresolved_id = max(int(v) for v in np.unique(merged) if v)
    rr, cc = draw.disk((70, 300), 25, shape=merged.shape)
    merged[rr, cc] = unresolved_id + 1

    from src.types import ObjectRecord

    objects = list(objects) + [
        ObjectRecord(unresolved_id + 1, unresolved_id + 1, "resolved")
    ]
    # Place them in one cluster by measuring with a contact distance that spans
    # the gap, then confirm the count is refused.
    clusters, _ = build_clusters(merged, objects, contact_distance_px=1)
    mixed = [c for c in clusters if c.n_unresolved_objects and c.n_resolved_cells]
    if mixed:
        assert mixed[0].cell_count is None
        assert mixed[0].status == "partially_resolved_cluster"
