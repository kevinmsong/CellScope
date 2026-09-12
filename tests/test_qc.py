"""QC tests (§13). The raw segmentation must survive every action untouched."""

from __future__ import annotations

import numpy as np
from conftest import separated_disks

from src.clustering import build_clusters
from src.qc import (
    apply_qc,
    exclude_border_touching,
    exclude_by_area,
    exclude_cluster,
    exclude_object,
    qc_counts,
    qc_log_dataframe,
    restore_object,
)
from src.types import ObjectRecord, QCState


def objects_for(labels):
    return [ObjectRecord(int(v), int(v), "resolved") for v in np.unique(labels) if v]


def test_apply_qc_never_modifies_the_raw_mask():
    """§13: the raw segmentation is never permanently modified."""
    raw = separated_disks(n=3, radius=30, gap=40)
    original = raw.copy()

    state = QCState()
    exclude_object(state, 2, "test")
    approved = apply_qc(raw, state)

    assert np.array_equal(raw, original), "apply_qc mutated its input"
    assert 2 in np.unique(raw)
    assert 2 not in np.unique(approved)


def test_surviving_objects_keep_their_ids():
    raw = separated_disks(n=3, radius=30, gap=40)
    state = QCState()
    exclude_object(state, 2, "test")
    approved = apply_qc(raw, state)
    assert sorted(int(v) for v in np.unique(approved) if v) == [1, 3]


def test_exclusion_is_reversible():
    raw = separated_disks(n=3, radius=30, gap=40)
    state = QCState()
    exclude_object(state, 2, "mistake")
    restore_object(state, 2, "on reflection, fine")
    assert np.array_equal(apply_qc(raw, state), raw)


def test_log_records_id_action_and_reason():
    state = QCState()
    exclude_object(state, 7, "debris")
    restore_object(state, 7, "not debris")

    log = qc_log_dataframe(state)
    assert list(log["object_id"]) == [7, 7]
    assert list(log["action"]) == ["exclude", "restore"]
    assert list(log["reason"]) == ["debris", "not debris"]
    assert all(log["timestamp"].str.len() > 0)


def test_repeated_exclusion_logs_once():
    state = QCState()
    assert exclude_object(state, 1, "first") is True
    assert exclude_object(state, 1, "again") is False
    assert len(qc_log_dataframe(state)) == 1


def test_empty_log_has_the_right_columns():
    log = qc_log_dataframe(QCState())
    assert len(log) == 0
    assert list(log.columns) == ["timestamp", "object_id", "action", "scope", "reason"]


def test_exclude_border_touching():
    objects = [
        ObjectRecord(1, 1, "resolved", touches_border=True),
        ObjectRecord(2, 2, "resolved", touches_border=False),
    ]
    state = QCState()
    assert exclude_border_touching(state, objects) == 1
    assert state.is_included(1) is False
    assert state.is_included(2) is True
    assert "border" in qc_log_dataframe(state).loc[0, "reason"]


def test_exclude_by_area_bounds():
    labels = np.zeros((50, 90), dtype=np.int32)
    labels[10:20, 10:20] = 1   # 100 px
    labels[10:15, 30:35] = 2   # 25 px
    labels[10:40, 50:80] = 3   # 900 px

    state = QCState()
    assert exclude_by_area(state, labels, min_area=50, max_area=500) == 2
    assert state.is_included(1) is True
    assert state.is_included(2) is False
    assert state.is_included(3) is False


def test_exclude_by_area_with_no_bounds_does_nothing():
    labels = separated_disks(n=2, radius=30, gap=40)
    state = QCState()
    assert exclude_by_area(state, labels) == 0
    assert state.excluded_ids == set()


def test_excluding_a_cell_can_split_a_cluster():
    """Clusters are recomputed on the approved mask, so removing a bridging cell
    genuinely separates what it was holding together."""
    labels = np.zeros((60, 300), dtype=np.int32)
    labels[20:40, 10:100] = 1
    labels[20:40, 101:190] = 2   # the bridge
    labels[20:40, 191:280] = 3
    objects = objects_for(labels)

    before, _ = build_clusters(labels, objects, contact_distance_px=2)
    assert [c.cluster_size for c in before] == [3]

    state = QCState()
    exclude_object(state, 2, "segmentation error")
    approved = apply_qc(labels, state)
    survivors = [o for o in objects if state.is_included(o.object_id)]

    after, _ = build_clusters(approved, survivors, contact_distance_px=2)
    assert sorted(c.cluster_size for c in after) == [1, 1]


def test_exclude_cluster_removes_every_member():
    labels = np.zeros((60, 300), dtype=np.int32)
    labels[20:40, 10:100] = 1
    labels[20:40, 101:190] = 2
    objects = objects_for(labels)
    clusters, _ = build_clusters(labels, objects, contact_distance_px=2)

    state = QCState()
    assert exclude_cluster(state, clusters[0]) == 2
    assert apply_qc(labels, state).max() == 0


def test_qc_counts_separate_cells_from_unresolved():
    objects = [
        ObjectRecord(1, 1, "resolved"),
        ObjectRecord(2, 2, "resolved"),
        ObjectRecord(3, 3, "unresolved_cluster"),
    ]
    state = QCState()
    exclude_object(state, 2, "debris")

    assert qc_counts(objects, state) == {
        "objects_detected": 3,
        "objects_excluded": 1,
        "objects_accepted": 2,
        "cells_accepted": 1,
        "unresolved_accepted": 1,
    }
