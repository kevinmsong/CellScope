"""The stepper and next-action guidance, and stale-segmentation detection."""

from __future__ import annotations

from dataclasses import replace

import pytest
from conftest import fluorescence_phantom

from src.calibration import calibration_from_resolution
from src.pipeline import run_segmentation
from src.types import AnalysisSession, BatchSession, ImageRecord
from src.workflow import change_token, image_state, is_stale, workflow_state


def _batch(n=2, segment=False):
    batch = BatchSession(label="exp")
    array = fluorescence_phantom(grid=2, size=200, spacing=80)
    for i in range(n):
        session = AnalysisSession()
        session.image = ImageRecord("f{}.png".format(i), str(i) * 64, 200, 200, "float32", 1, array)
        batch.add(session)
        if segment:
            run_segmentation(session, array, engine="threshold_watershed")
    return batch


def _states(batch):
    steps, action = workflow_state(batch)
    return {s.key: s.state for s in steps}, action


def test_empty_batch_points_at_import():
    states, action = _states(BatchSession())
    assert states["import"] == "current"
    assert "Add images" in action


def test_calibration_comes_before_segmentation():
    states, action = _states(_batch())
    assert states["import"] == "done"
    assert states["calibrate"] == "current"
    assert "Calibrate" in action


def test_confirmed_pixel_units_are_a_warning_not_a_blocker():
    batch = _batch()
    batch.calibration_confirmed = True
    states, action = _states(batch)
    assert states["calibrate"] == "warn"
    assert states["segment"] == "current"
    assert "Segment" in action


def test_mixed_calibration_is_an_error():
    batch = _batch()
    batch.images[0].calibration = calibration_from_resolution(0.5)
    states, action = _states(batch)
    assert states["calibrate"] == "error"
    assert "remaining 1 image" in action


def test_review_is_next_after_segmentation():
    batch = _batch(segment=True)
    batch.calibration = calibration_from_resolution(0.5)
    batch.apply_shared("calibration")
    states, action = _states(batch)
    assert states["segment"] == "done"
    assert states["review"] == "current"
    assert action == "Review f0.png (2 left)."


def test_failed_images_are_reported():
    batch = _batch(segment=True)
    batch.calibration_confirmed = True
    batch.images[1].error = "RuntimeError: boom"
    states, action = _states(batch)
    assert states["segment"] == "error"
    assert "failed" in action


def test_changing_a_segmentation_setting_marks_the_image_stale():
    batch = _batch(segment=True)
    session = batch.images[0]
    assert not is_stale(session)
    session.segmentation_params = replace(session.segmentation_params, flow_threshold=0.9)
    assert is_stale(session)
    assert image_state(session) == "stale"
    batch.calibration_confirmed = True
    states, action = _states(batch)
    assert states["segment"] == "warn"
    assert "settings that have since changed" in action


def test_calibration_and_clustering_do_not_make_masks_stale():
    batch = _batch(segment=True)
    session = batch.images[0]
    session.calibration = calibration_from_resolution(0.3)
    session.cluster_params = replace(session.cluster_params, contact_distance_px=5)
    assert not is_stale(session)


def test_restoring_the_setting_clears_staleness():
    session = _batch(segment=True).images[0]
    original = session.segmentation_params
    session.segmentation_params = replace(original, cellprob_threshold=2.0)
    assert is_stale(session)
    session.segmentation_params = original
    assert not is_stale(session)


def test_export_step_tracks_changes_after_export():
    batch = _batch(segment=True)
    batch.calibration_confirmed = True
    for s in batch.images:
        s.reviewed = True
        s.well, s.condition, s.replicate = "A1", "ctrl", "R1"
    batch.exported_token = change_token(batch)
    states, action = _states(batch)
    assert states["export"] == "done"
    assert action == "Analysis complete and exported."

    batch.images[0].qc.exclude(1, "artifact")
    states, _ = _states(batch)
    assert states["export"] != "done"


@pytest.mark.parametrize("field,value", [
    ("well", "B2"), ("review_note", "blurry"), ("label", "renamed"),
])
def test_change_token_moves_on_edits(field, value):
    batch = _batch()
    before = change_token(batch)
    target = batch if field == "label" else batch.images[0]
    setattr(target, field, value)
    assert change_token(batch) != before
