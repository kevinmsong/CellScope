"""Undo/redo: copy-on-write masks, and QC-only undo that keeps the geometry cache."""

from __future__ import annotations

import numpy as np
import pytest
from conftest import fluorescence_phantom

from src.pipeline import compute_results, run_segmentation
from src.review import checkpoint, correct_mask, history
from src.types import AnalysisSession, ImageRecord


@pytest.fixture
def session():
    array = fluorescence_phantom(grid=3, size=320)
    s = AnalysisSession()
    s.image = ImageRecord("f.png", "0" * 64, 320, 320, "float32", 1, array)
    run_segmentation(s, array, engine="threshold_watershed")
    return s


def test_segmented_labels_are_read_only(session):
    assert not session.object_labels.flags.writeable
    with pytest.raises(ValueError):
        session.object_labels[0, 0] = 5


def test_corrected_labels_are_read_only(session):
    ids = sorted(o.object_id for o in session.objects)[:1]
    rows, cols = np.nonzero(session.object_labels == ids[0])
    points = [(int(cols.min()), int(rows.min())), (int(cols.max()), int(rows.min())),
              (int(cols.max()), int(rows.max()))]
    correct_mask(session, "Replace boundary", ids, points)
    assert not session.object_labels.flags.writeable


def test_snapshots_share_the_mask_instead_of_copying(session):
    checkpoint(session)
    assert session.undo_stack[-1][0] is session.object_labels


def test_undoing_an_exclusion_keeps_the_segmentation_version(session):
    version = session.segmentation_version
    target = session.objects[0].object_id
    before = compute_results(session).cells.copy()

    checkpoint(session)
    session.qc.exclude(target, "artifact")
    assert history(session)

    assert session.segmentation_version == version
    assert target not in session.qc.excluded_ids
    after = compute_results(session).cells
    pd_equal(before, after)


def test_undoing_a_correction_restores_the_mask_and_bumps_the_version(session):
    original = session.object_labels
    ids = [o.object_id for o in session.objects][:1]
    rows, cols = np.nonzero(original == ids[0])
    points = [(int(cols.min()), int(rows.min())), (int(cols.max()), int(rows.min())),
              (int(cols.max()), int(rows.max()))]
    correct_mask(session, "Replace boundary", ids, points)
    edited_version = session.segmentation_version
    assert session.object_labels is not original

    assert history(session)
    assert session.object_labels is original
    assert session.segmentation_version == edited_version + 1

    assert history(session, redo=True)
    assert session.object_labels is not original
    assert session.segmentation_version == edited_version + 2


def test_results_after_undo_match_a_fresh_computation(session):
    target = session.objects[1].object_id
    checkpoint(session)
    session.qc.exclude(target, "artifact")
    compute_results(session)
    history(session)
    cached = compute_results(session)
    fresh = compute_results(session, use_cache=False)
    pd_equal(cached.cells, fresh.cells)
    pd_equal(cached.cluster_summary, fresh.cluster_summary)


def pd_equal(a, b):
    import pandas as pd

    pd.testing.assert_frame_equal(a.reset_index(drop=True), b.reset_index(drop=True))
