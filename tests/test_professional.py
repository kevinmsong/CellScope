"""Regression coverage for persistence, corrections, design summaries and DAPI contact."""
from dataclasses import replace
import json
import zipfile

import numpy as np
import pytest
from src.types import AnalysisSession, BatchSession, ImageRecord, ObjectRecord
from src.project import save_project, load_project, preset_dict, apply_preset
from src.review import checkpoint, history, correct_mask, signature, validate_review, click_position
from src.nuclei import nuclear_counts, nuclear_edges, segment_nuclei
from src.quality import readiness, experimental_summary


@pytest.fixture
def project():
    labels = np.zeros((32, 48), np.int32)
    labels[5:15, 5:15] = 1
    labels[5:15, 15:25] = 2
    s = AnalysisSession()
    pixels = np.zeros((32, 48, 3), np.uint8)
    pixels[labels > 0, 2] = 255
    s.image = ImageRecord("synthetic.png", "0"*64, 32, 48, "uint8", 3, pixels)
    s.raw_labels = labels.copy()
    s.object_labels = labels.copy()
    s.objects = [ObjectRecord(1, 1, "resolved"), ObjectRecord(2, 2, "resolved")]
    s.segmented_channel = s.channel
    b = BatchSession(label="Experiment")
    b.add(s)
    return b


def test_portable_project_roundtrip_and_readonly_sources(project, tmp_path):
    s = project.active
    s.qc.exclude(2, "artifact")
    s.review_note = "Reviewed carefully"
    s.condition, s.replicate, s.well = "Control", "Mouse1", "A1"
    s.nuclei_labels = s.object_labels.copy()
    s.nuclei_excluded = {2}
    path = save_project(project, tmp_path / "test.cellscope")
    loaded = load_project(path)
    assert loaded.label == "Experiment"
    assert loaded.active.qc.excluded_ids == {2}
    assert loaded.active.nuclei_excluded == {2}
    assert loaded.active.review_note == s.review_note
    assert loaded.active.replicate == "Mouse1"
    np.testing.assert_array_equal(loaded.active.raw_labels, s.raw_labels)
    assert not loaded.active.raw_labels.flags.writeable
    assert not loaded.active.image.pixels.flags.writeable
    assert loaded.active.results_cache is None


def test_project_rejects_unknown_schema(tmp_path):
    path = tmp_path / "bad.cellscope"
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("project.json", json.dumps({"schema": 999}))
    with pytest.raises(ValueError, match="version"):
        load_project(path)


def test_bulk_undo_redo_retains_audit_log(project):
    s = project.active
    checkpoint(s)
    s.qc.exclude(1, "artifact")
    s.qc.exclude(2, "artifact")
    assert history(s)
    assert not s.qc.excluded_ids
    assert history(s, redo=True)
    assert s.qc.excluded_ids == {1, 2}
    assert len(s.qc.log) == 6


def test_merge_split_boundary_and_undo_preserve_raw(project):
    s = project.active
    raw = s.raw_labels.copy()
    correct_mask(s, "Merge", [1, 2], [])
    assert len(s.objects) == 1
    correct_mask(s, "Split", [1], [(7, 8), (22, 8)])
    assert len(s.objects) == 2
    assert history(s)
    assert len(s.objects) == 1
    correct_mask(s, "Replace boundary", [1], [(4, 4), (24, 4), (24, 14), (4, 14)])
    assert s.object_labels[4, 4] == 1
    np.testing.assert_array_equal(raw, s.raw_labels)
    assert s.edit_log


def test_bad_correction_does_not_mutate(project):
    s = project.active
    raw = s.object_labels.copy()
    with pytest.raises(ValueError):
        correct_mask(s, "Split", [1], [(0, 0), (8, 8)])
    np.testing.assert_array_equal(s.object_labels, raw)
    assert not s.undo_stack


def test_stale_review_and_zoom_hit_testing(project):
    s = project.active
    s.reviewed = True
    s.reviewed_signature = signature(s)
    s.calibration = replace(s.calibration, um_per_px_x=.5, um_per_px_y=.5)
    validate_review(s)
    assert not s.reviewed
    s.review_zoom = 2
    s.review_pan_x = s.review_pan_y = 100
    assert click_position(s, (0, 0), (32, 48)) == (24, 16)
    assert click_position(s, (-1, 0), (32, 48)) is None


def test_presets_roundtrip_dont_copy_calibration(project):
    project.segmentation_params = replace(project.segmentation_params, diameter=42)
    data = preset_dict(project)
    other = BatchSession()
    apply_preset(other, data)
    assert other.segmentation_params.diameter == 42
    assert not other.calibration.is_calibrated


def test_experimental_summary_uses_replicates(project):
    import copy
    s = project.active
    s.well, s.condition, s.replicate = "A1", "Control", "R1"
    other = copy.deepcopy(s)
    other.analysis_id = "other"
    other.well, other.replicate = "A2", "R2"
    other.object_labels[5:15, 15:25] = 0
    other.objects = other.objects[:1]
    project.add(other)
    wells, reps, conditions = experimental_summary(project)
    assert len(wells) == 2
    assert len(reps) == 2
    assert conditions.iloc[0].biological_replicates == 2
    assert conditions.iloc[0].mean_area == 100


def test_readiness_identifies_unreviewed_uncalibrated(project):
    details = " ".join(readiness(project).Detail)
    assert "Needs review" in details
    assert "Uncalibrated" in details


def test_direct_nuclear_contact_counts_and_exclusion(project):
    s = project.active
    labels = np.zeros((12, 12), np.int32)
    labels[2:5, 2:5] = 1
    labels[2:5, 5:8] = 2
    labels[8:10, 8:10] = 3
    s.nuclei_labels = labels
    counts = nuclear_counts(s)
    assert counts['nuclei_included'] == 3
    assert counts['touching_nuclei'] == 2
    assert counts['touching_groups'] == 1
    assert counts['isolated_nuclei'] == 1
    s.nuclei_excluded.add(2)
    assert nuclear_counts(s)['touching_nuclei'] == 0


def test_diagonal_contact_counts_but_one_pixel_gap_does_not():
    labels = np.zeros((6, 6), np.int32)
    labels[1, 1], labels[2, 2], labels[2, 4] = 1, 2, 3
    assert nuclear_edges(labels) == [(1, 2)]


def test_dapi_run_does_not_change_cell_masks(project):
    s = project.active
    raw = s.object_labels.copy()
    counts = segment_nuclei(s, min_area=10, seed_distance=4)
    assert counts['nuclei_detected'] > 0
    assert s.nuclei_settings['channel'] == 'blue'
    np.testing.assert_array_equal(raw, s.object_labels)


def test_report_and_export_contain_nuclear_and_edit_provenance(project, tmp_path):
    from src.export import export_batch
    s = project.active
    segment_nuclei(s, min_area=10)
    path = export_batch(project, str(tmp_path))
    with zipfile.ZipFile(path) as archive:
        names = archive.namelist()
        assert 'report.pdf' in names
        assert 'nuclei_counts.csv' in names
        assert 'replicate_summary.csv' in names
        assert any(name.endswith('nuclei_labels.tif') for name in names)
        assert any(name.endswith('object_labels.tif') for name in names)
        assert archive.read('report.pdf').startswith(b'%PDF')


def test_nuclei_only_export_does_not_require_cell_segmentation(project, tmp_path):
    from src.nuclei import export_nuclei
    s = project.active
    segment_nuclei(s, min_area=10)
    s.object_labels = None
    s.objects = []
    with zipfile.ZipFile(export_nuclei(project, tmp_path)) as archive:
        assert "nuclei_counts.csv" in archive.namelist()
        assert "image_1/nuclei_labels.tif" in archive.namelist()


def test_review_inspection_includes_excluded_measurements(project):
    from professional_ui import review_table
    s = project.active
    s.qc.exclude(1, "artifact")
    s.selected_object = 1
    frame, detail, _, _ = review_table(project)
    row = frame[frame.cell_id == 1].iloc[0]
    assert not row.included
    assert row.area_px2 == 100
    assert "diagnostic only" in detail
    assert "NaN" not in detail


def test_atomic_save_failure_preserves_previous_file(project, tmp_path, monkeypatch):
    import src.project as module
    path = tmp_path / "project.cellscope"
    save_project(project, path)
    original = path.read_bytes()
    monkeypatch.setattr(module.os, "replace", lambda *args: (_ for _ in ()).throw(OSError("disk error")))
    with pytest.raises(OSError):
        save_project(project, path)
    assert path.read_bytes() == original
    assert not list(tmp_path.glob("*.tmp"))
