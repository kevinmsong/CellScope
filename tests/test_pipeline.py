"""End-to-end pipeline and export tests (§17, §18, §22)."""

from __future__ import annotations

import json
import os
import zipfile

import numpy as np
import pandas as pd
import pytest
from conftest import fluorescence_phantom

from src.calibration import calibration_from_resolution, uncalibrated
from src.export import ARCHIVE_MEMBERS, export_analysis
from src.pipeline import compute_results, run_segmentation
from src.qc import exclude_object, restore_object
from src.types import AnalysisSession, ClusterParams, ImageRecord
from src.visualize import make_overlay


def build_session(calibration=None, image=None):
    image = fluorescence_phantom() if image is None else image
    session = AnalysisSession()
    session.image = ImageRecord(
        "phantom.png", "0" * 64, image.shape[0], image.shape[1], "float32", 1, image
    )
    session.calibration = calibration or calibration_from_resolution(0.4505)
    return session, image


def run_all(calibration=None):
    session, image = build_session(calibration)
    run_segmentation(session, image, engine="threshold_watershed")
    return session, compute_results(session)


@pytest.fixture
def analysis():
    return run_all()


def test_pipeline_finds_the_expected_objects(analysis):
    session, results = analysis
    assert len(session.objects) == 9
    assert len(results.cells) == 9
    assert set(results.cells["segmentation_status"]) == {"resolved"}


def test_one_row_per_resolved_cell(analysis):
    """§17: cell_measurements.csv has exactly one row per resolved cell."""
    session, results = analysis
    resolved = [o for o in results.objects if o.is_resolved]
    assert len(results.cells) == len(resolved)
    assert sorted(results.cells["cell_id"]) == sorted(o.object_id for o in resolved)
    assert results.cells["cell_id"].is_unique


def test_one_row_per_cluster(analysis):
    session, results = analysis
    assert len(results.cluster_summary) == len(results.clusters)
    assert results.cluster_summary["cluster_id"].is_unique


def test_engine_is_reported_honestly(analysis):
    """The app must never imply Cellpose ran when it did not."""
    session, _ = analysis
    assert session.engine_info["engine"] == "threshold_watershed"


def test_segmentation_is_deterministic():
    first_session, first = run_all()
    second_session, second = run_all()
    assert np.array_equal(first.qc_labels, second.qc_labels)
    pd.testing.assert_frame_equal(first.cells, second.cells)


def test_uncalibrated_pipeline_emits_no_physical_columns():
    _, results = run_all(uncalibrated())
    leaked = [c for c in results.cells.columns if c.endswith(("_um", "_um2"))]
    assert leaked == []
    assert "area_px2" in results.cells.columns


# --------------------------------------------------------------------------- #
# QC through the pipeline
# --------------------------------------------------------------------------- #


def test_qc_exclusion_removes_the_cell_but_not_the_raw_mask(analysis):
    session, _ = analysis
    raw_before = np.array(session.raw_labels, copy=True)

    exclude_object(session.qc, 3, "test exclusion")
    after = compute_results(session)

    assert 3 not in set(after.cells["cell_id"])
    assert len(after.cells) == 8
    assert np.array_equal(session.raw_labels, raw_before), "raw labels were mutated"
    assert 3 in np.unique(session.object_labels)


def test_counts_reflect_exclusions(analysis):
    session, _ = analysis
    exclude_object(session.qc, 1, "debris")
    results = compute_results(session)
    assert results.counts["objects_detected"] == 9
    assert results.counts["objects_accepted"] == 8
    assert results.counts["objects_excluded"] == 1


# --------------------------------------------------------------------------- #
# Export
# --------------------------------------------------------------------------- #


def test_archive_contains_every_required_member(analysis, tmp_path):
    session, results = analysis
    overlay = make_overlay(
        np.zeros(results.qc_labels.shape + (3,), np.uint8),
        results.qc_labels,
        results.objects,
        results.clusters,
    )
    archive_path = export_analysis(session, results, str(tmp_path), overlay_rgb=overlay)

    with zipfile.ZipFile(archive_path) as archive:
        names = set(archive.namelist())
    assert names == set(ARCHIVE_MEMBERS)


def test_metadata_records_every_required_field(analysis, tmp_path):
    """§18: the provenance record must be complete."""
    session, results = analysis
    archive_path = export_analysis(session, results, str(tmp_path))

    with zipfile.ZipFile(archive_path) as archive:
        metadata = json.loads(archive.read("metadata.json"))

    for key in (
        "analysis_id", "exported_at", "software", "source_image",
        "segmentation_channel", "segmentation_engine", "preprocessing",
        "segmentation_parameters", "splitting_parameters", "clustering_parameters",
        "calibration", "measurement", "analysis_roi", "qc", "object_provenance",
    ):
        assert key in metadata, "metadata.json is missing {}".format(key)

    assert metadata["source_image"]["sha256"] == "0" * 64
    assert metadata["software"]["python_version"]
    assert metadata["calibration"]["um_per_px_x"] == pytest.approx(0.4505)
    assert metadata["measurement"]["perimeter_estimator"] == "crofton_scaled"
    assert metadata["clustering_parameters"]["contact_distance_px"] == 2


def test_metadata_records_qc_actions(analysis, tmp_path):
    session, _ = analysis
    exclude_object(session.qc, 2, "obvious debris")
    results = compute_results(session)
    archive_path = export_analysis(session, results, str(tmp_path))

    with zipfile.ZipFile(archive_path) as archive:
        metadata = json.loads(archive.read("metadata.json"))

    assert metadata["qc"]["excluded_object_ids"] == [2]
    assert metadata["qc"]["actions"][0]["reason"] == "obvious debris"
    assert any(
        entry["object_id"] == 2 and entry["included"] is False
        for entry in metadata["object_provenance"]
    )


# --------------------------------------------------------------------------- #
# §22.18 -- measurements must be reproducible from the exported artefacts alone
# --------------------------------------------------------------------------- #


def test_areas_are_reproducible_from_the_exported_mask_and_metadata(analysis, tmp_path):
    """The traceability chain, executed.

    Nothing from the live session is used: the mask, the calibration and the
    measurements all come back out of the archive, and the areas are recomputed
    from first principles.
    """
    import tifffile

    session, results = analysis
    archive_path = export_analysis(session, results, str(tmp_path))
    extracted = tmp_path / "unpacked"
    with zipfile.ZipFile(archive_path) as archive:
        archive.extractall(extracted)

    labels = tifffile.imread(extracted / "masks" / "qc_labels.tif")
    metadata = json.loads((extracted / "metadata.json").read_text(encoding="utf-8"))
    cells = pd.read_csv(extracted / "cell_measurements.csv")

    sx = metadata["calibration"]["um_per_px_x"]
    sy = metadata["calibration"]["um_per_px_y"]

    counts = np.bincount(labels.ravel())
    for _, row in cells.iterrows():
        cell_id = int(row["cell_id"])
        pixels = int(counts[cell_id])
        assert pixels == pytest.approx(row["area_px2"]), (
            "pixel count for cell {} does not match the exported area".format(cell_id)
        )
        assert pixels * sx * sy == pytest.approx(row["area_um2"], rel=1e-9), (
            "area_um2 for cell {} is not recoverable from mask x calibration".format(cell_id)
        )


def test_exported_mask_preserves_every_cell_id(analysis, tmp_path):
    """§17: integer TIFF masks must preserve Cell IDs exactly."""
    import tifffile

    session, results = analysis
    archive_path = export_analysis(session, results, str(tmp_path))
    extracted = tmp_path / "unpacked"
    with zipfile.ZipFile(archive_path) as archive:
        archive.extractall(extracted)

    labels = tifffile.imread(extracted / "masks" / "qc_labels.tif")
    assert np.array_equal(
        np.unique(labels).astype(np.int64), np.unique(results.qc_labels).astype(np.int64)
    )
    assert np.array_equal(labels.astype(np.int64), results.qc_labels.astype(np.int64))


def test_raw_mask_in_the_archive_still_holds_the_excluded_cell(analysis, tmp_path):
    """A QC decision must be auditable: the raw mask shows what was removed."""
    import tifffile

    session, _ = analysis
    exclude_object(session.qc, 4, "test")
    results = compute_results(session)
    archive_path = export_analysis(session, results, str(tmp_path))

    extracted = tmp_path / "unpacked"
    with zipfile.ZipFile(archive_path) as archive:
        archive.extractall(extracted)

    raw = tifffile.imread(extracted / "masks" / "raw_labels.tif")
    approved = tifffile.imread(extracted / "masks" / "qc_labels.tif")
    assert 4 in np.unique(raw), "raw mask lost the excluded object"
    assert 4 not in np.unique(approved)


def test_image_summary_counts_are_internally_consistent(analysis, tmp_path):
    session, results = analysis
    summary = results.image_summary.iloc[0]
    assert summary["objects_detected"] == summary["objects_accepted"] + summary["objects_excluded"]
    assert summary["clusters_total"] == len(results.clusters)
    assert summary["isolated_cells"] + summary["cells_in_clusters"] == summary["cells_resolved"]


# --------------------------------------------------------------------------- #
# Cellpose. Skipped when it is not installed, so `pytest` passes on a clean
# checkout without triggering a model download.
# --------------------------------------------------------------------------- #

from src.segmentation import cellpose_available

requires_cellpose = pytest.mark.skipif(
    not cellpose_available(), reason="Cellpose is not installed in this environment"
)


def test_engine_falls_back_and_says_so_when_cellpose_is_unavailable(monkeypatch):
    """A degraded run must be labelled degraded, never passed off as Cellpose."""
    import src.segmentation as segmentation

    monkeypatch.setattr(segmentation, "cellpose_available", lambda: False)
    labels, info = segmentation.segment_cells(fluorescence_phantom(), engine="auto")
    assert info["engine"] == "threshold_watershed"
    assert labels.max() == 9


def test_requesting_cellpose_explicitly_raises_when_absent(monkeypatch):
    import src.segmentation as segmentation

    monkeypatch.setattr(segmentation, "cellpose_available", lambda: False)
    with pytest.raises(segmentation.CellposeUnavailable, match="not installed"):
        segmentation.segment_cells(fluorescence_phantom(), engine="cellpose")


@requires_cellpose
def test_cellpose_adapter_builds_three_channel_input():
    """Cellpose 4 requires 3 channels; we stack the chosen one explicitly so the
    channel recorded in metadata is provably the channel segmented."""
    from src.segmentation import _as_three_channel

    stacked = _as_three_channel(np.zeros((32, 48), dtype=np.float32))
    assert stacked.shape == (32, 48, 3)
    assert np.array_equal(stacked[..., 0], stacked[..., 2])


@requires_cellpose
def test_cellpose_version_is_reported():
    from src.segmentation import cellpose_version

    version = cellpose_version()
    assert version and version[0].isdigit()


def test_export_does_not_leak_files_from_a_previous_run(analysis, tmp_path):
    """Staging must be fresh each time.

    A reused staging directory let an export without an overlay ship the
    previous run's overlay -- a stale artefact presented as this analysis's
    output, in the module whose whole job is provenance.
    """
    session, results = analysis
    overlay = make_overlay(
        np.zeros(results.qc_labels.shape + (3,), np.uint8),
        results.qc_labels,
        results.objects,
        results.clusters,
    )
    first = export_analysis(session, results, str(tmp_path), overlay_rgb=overlay)
    with zipfile.ZipFile(first) as archive:
        assert "overlays/segmentation_overlay.png" in archive.namelist()

    second = export_analysis(session, results, str(tmp_path), overlay_rgb=None)
    with zipfile.ZipFile(second) as archive:
        assert "overlays/segmentation_overlay.png" not in archive.namelist()


def test_export_leaves_no_staging_directory(analysis, tmp_path):
    session, results = analysis
    export_analysis(session, results, str(tmp_path))
    leftovers = [p.name for p in tmp_path.iterdir() if p.is_dir() and p.name.startswith("_")]
    assert leftovers == []


def test_concurrent_sessions_do_not_overwrite_each_others_archive(analysis, tmp_path):
    """Two sessions exporting at once must get separate files."""
    session, results = analysis
    first = export_analysis(session, results, str(tmp_path))

    other = AnalysisSession()
    other.image = session.image
    other.calibration = session.calibration
    other.raw_labels = session.raw_labels
    other.object_labels = session.object_labels
    other.objects = session.objects
    other.engine_info = session.engine_info
    second = export_analysis(other, compute_results(other), str(tmp_path))

    assert first != second
    assert os.path.exists(first) and os.path.exists(second)


# --------------------------------------------------------------------------- #
# The derived-results cache must never serve a stale answer
# --------------------------------------------------------------------------- #


def test_cache_returns_the_same_object_when_nothing_changed(analysis):
    session, first = analysis
    assert compute_results(session) is first


def test_cache_is_invalidated_by_a_qc_exclusion(analysis):
    session, first = analysis
    exclude_object(session.qc, 3, "test")
    second = compute_results(session)
    assert second is not first
    assert 3 not in set(second.cells["cell_id"])


def test_cache_is_invalidated_by_a_qc_restore(analysis):
    session, _ = analysis
    exclude_object(session.qc, 3, "test")
    excluded = compute_results(session)
    restore_object(session.qc, 3)
    restored = compute_results(session)
    assert restored is not excluded
    assert 3 in set(restored.cells["cell_id"])


def test_cache_is_invalidated_by_a_calibration_change(analysis):
    """A recalibration must not leave the old micron values on screen."""
    session, first = analysis
    session.calibration = calibration_from_resolution(1.0, 1.0)
    second = compute_results(session)
    assert second is not first
    assert second.cells["area_um2"].iloc[0] != pytest.approx(first.cells["area_um2"].iloc[0])


def test_cache_is_invalidated_by_a_contact_distance_change(analysis):
    session, first = analysis
    session.cluster_params = ClusterParams(contact_distance_px=9)
    assert compute_results(session) is not first


def test_cache_is_invalidated_by_re_segmentation(analysis):
    session, first = analysis
    run_segmentation(session, fluorescence_phantom(), engine="threshold_watershed")
    assert compute_results(session) is not first


def test_cache_can_be_bypassed(analysis):
    session, first = analysis
    assert compute_results(session, use_cache=False) is not first


# --------------------------------------------------------------------------- #
# Channel provenance: metadata must describe the masks it ships with
# --------------------------------------------------------------------------- #


def test_metadata_reports_the_channel_that_was_segmented(analysis, tmp_path):
    """Changing the channel selection must not rewrite history.

    Selecting a different channel without re-running leaves the old masks in
    place. Reporting the new selection as `segmentation_channel` would claim a
    provenance that is false of the exported masks.
    """
    session, _ = analysis
    session.channel = "grayscale"
    assert session.segmented_channel == "grayscale"

    session.channel = "blue"  # selection moves on; masks do not
    results = compute_results(session)
    archive_path = export_analysis(session, results, str(tmp_path))
    with zipfile.ZipFile(archive_path) as archive:
        metadata = json.loads(archive.read("metadata.json"))

    assert metadata["segmentation_channel"] == "grayscale"
    assert metadata["selected_channel_at_export"] == "blue"
    assert results.image_summary.iloc[0]["segmentation_channel"] == "grayscale"


def test_re_running_updates_the_recorded_channel(analysis):
    session, _ = analysis
    session.channel = "blue"
    run_segmentation(session, fluorescence_phantom(), engine="threshold_watershed")
    assert session.segmented_channel == "blue"


# --------------------------------------------------------------------------- #
# Optional nuclear channel
# --------------------------------------------------------------------------- #


def test_three_channel_stack_replicates_without_a_nuclear_image():
    from src.segmentation import _as_three_channel

    stack = _as_three_channel(np.arange(12, dtype=np.float32).reshape(3, 4))
    assert stack.shape == (3, 4, 3)
    assert np.array_equal(stack[..., 0], stack[..., 1])
    assert np.array_equal(stack[..., 1], stack[..., 2])


def test_three_channel_stack_places_nuclei_in_planes_two_and_three():
    """The arrangement Cellpose reads as cytoplasm plus nucleus."""
    from src.segmentation import _as_three_channel

    cyto = np.ones((3, 4), dtype=np.float32)
    nuclei = np.full((3, 4), 7.0, dtype=np.float32)
    stack = _as_three_channel(cyto, nuclei)

    assert np.array_equal(stack[..., 0], cyto)
    assert np.array_equal(stack[..., 1], nuclei)
    assert np.array_equal(stack[..., 2], nuclei)


def test_mismatched_nuclear_channel_is_rejected():
    """A nuclear image of the wrong size would silently misalign the guidance."""
    from src.segmentation import _as_three_channel

    with pytest.raises(ValueError, match="does not match"):
        _as_three_channel(np.zeros((10, 10), np.float32), np.zeros((5, 5), np.float32))


def test_nuclear_channel_is_recorded_in_metadata(tmp_path):
    """§18: guidance that changed the masks must appear in the provenance."""
    from src.types import SegmentationParams

    session, image = build_session()
    session.segmentation_params = SegmentationParams(nuclear_channel="blue")
    run_segmentation(session, image, engine="threshold_watershed")
    results = compute_results(session)

    assert results.image_summary.iloc[0]["nuclear_channel"] == "blue"
    archive = export_analysis(session, results, str(tmp_path))
    with zipfile.ZipFile(archive) as z:
        metadata = json.loads(z.read("metadata.json"))
    assert metadata["segmentation_parameters"]["nuclear_channel"] == "blue"


def test_nuclear_channel_defaults_to_off():
    from src.types import SegmentationParams

    assert SegmentationParams().nuclear_channel is None


def test_segmentation_automatically_excludes_border_objects(monkeypatch):
    import src.pipeline as pipeline
    from src.types import AnalysisSession, ObjectRecord
    labels = np.zeros((10, 10), dtype=np.int32)
    labels[:3, :3] = 1
    labels[5:8, 5:8] = 2
    objects = [ObjectRecord(1, 1, "resolved", touches_border=True),
               ObjectRecord(2, 2, "resolved")]
    monkeypatch.setattr(pipeline, "segment_cells", lambda *a, **k: (labels.copy(), {}))
    monkeypatch.setattr(pipeline, "split_touching_cells", lambda *a, **k: (labels.copy(), objects))
    session = AnalysisSession()
    session.reviewed = True
    pipeline.run_segmentation(session, np.zeros((10, 10)))
    assert session.qc.excluded_ids == {1}
    assert "automatically" in session.qc.log[0].reason
    assert not session.reviewed
    np.testing.assert_array_equal(session.raw_labels, labels)
    session.qc.restore(1)
    assert session.qc.is_included(1)
    pipeline.run_segmentation(session, np.zeros((10, 10)))
    assert session.qc.excluded_ids == {1}
