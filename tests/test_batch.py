"""Batch pooling and its guards.

The batch result is a single pooled population: every accepted cell from every
image. These tests pin that pooling is weighted by cell count rather than by
image, and that a batch which cannot be pooled says so instead of producing a
number with no unit.
"""

from __future__ import annotations

import sys

import numpy as np
import pandas as pd
import pytest
from conftest import fluorescence_phantom

from src.batch import (
    CalibrationMismatch,
    batch_qc_log,
    batch_summary,
    calibration_consistency,
    headline_counts,
    per_image_summary,
    pooled_cells,
    pooled_clusters,
)
from src.calibration import calibration_from_resolution, uncalibrated
from src.pipeline import compute_results, run_segmentation
from src.qc import exclude_object
from src.types import AnalysisSession, BatchSession, ImageRecord

# --------------------------------------------------------------------------- #
# Builders
# --------------------------------------------------------------------------- #


def make_image(name: str, grid: int = 3, size: int = 320, spacing: int = 90):
    """A segmented image whose cell count is controlled by ``grid``."""
    array = fluorescence_phantom(grid=grid, size=size, spacing=spacing)
    session = AnalysisSession()
    session.image = ImageRecord(
        name, "0" * 64, array.shape[0], array.shape[1], "float32", 1, array
    )
    run_segmentation(session, array, engine="threshold_watershed")
    return session, array


def make_batch(specs, calibration=None):
    """Build a batch from ``[(name, grid), ...]``."""
    batch = BatchSession(label="test batch")
    batch.calibration = calibration or calibration_from_resolution(0.25)
    for name, grid in specs:
        session, _ = make_image(name, grid=grid)
        session.calibration = batch.calibration
        batch.add(session)
        # add() seeds shared params; re-run is unnecessary since calibration
        # only affects measurement, not segmentation.
        session.results_cache = None
    return batch


@pytest.fixture
def uneven_batch():
    """Three images with deliberately unequal cell counts."""
    return make_batch([("a.png", 2), ("b.png", 3), ("c.png", 4)])


# --------------------------------------------------------------------------- #
# The decisive test: pooling is weighted by cell count, not by image
# --------------------------------------------------------------------------- #


def test_pooled_mean_is_over_cells_not_over_images(uneven_batch):
    """A 16-cell image must count for more than a 4-cell one.

    The unweighted mean of per-image means and the pooled mean over all cells
    diverge exactly when images hold different numbers of cells. Asserting both
    directions means a regression to per-image averaging cannot pass.
    """
    cells = pooled_cells(uneven_batch)
    per_image = per_image_summary(uneven_batch)

    counts = cells.groupby("image_id").size()
    assert counts.nunique() > 1, "fixture must have unequal cell counts to be meaningful"

    pooled_mean = cells["area_um2"].mean()
    mean_of_image_means = per_image["mean_cell_area_um2"].mean()

    summary = batch_summary(uneven_batch)
    reported = summary.loc[0, "mean_area_um2"]

    assert reported == pytest.approx(pooled_mean), "summary must pool over cells"
    assert reported != pytest.approx(mean_of_image_means, rel=1e-6), (
        "pooled mean coincides with the per-image mean; the fixture no longer "
        "distinguishes the two and this test proves nothing"
    )


def test_n_cells_is_the_cell_count_and_n_images_the_image_count(uneven_batch):
    summary = batch_summary(uneven_batch).iloc[0]
    assert summary["n_cells"] == len(pooled_cells(uneven_batch))
    assert summary["n_images"] == len(uneven_batch.images) == 3
    assert summary["n_cells"] > summary["n_images"]


def test_every_image_contributes_to_the_pool(uneven_batch):
    cells = pooled_cells(uneven_batch)
    assert set(cells["image_id"]) == {"a.png", "b.png", "c.png"}


# --------------------------------------------------------------------------- #
# Origin tagging
# --------------------------------------------------------------------------- #


def test_pooled_cells_carry_image_and_well(uneven_batch):
    for index, well in enumerate(("B3", "B4", "C1")):
        uneven_batch.images[index].well = well
        uneven_batch.images[index].results_cache = None

    cells = pooled_cells(uneven_batch)
    assert list(cells.columns[:2]) == ["image_id", "well"]
    assert set(cells["well"]) == {"B3", "B4", "C1"}
    assert set(cells.loc[cells["image_id"] == "a.png", "well"]) == {"B3"}


def test_pooled_clusters_carry_image_and_well(uneven_batch):
    clusters = pooled_clusters(uneven_batch)
    assert list(clusters.columns[:2]) == ["image_id", "well"]
    assert len(clusters) > 0


def test_cell_ids_repeat_across_images_so_image_id_is_required(uneven_batch):
    """Cell IDs are per image, so image_id is what makes a pooled row unique."""
    cells = pooled_cells(uneven_batch)
    assert cells["cell_id"].duplicated().any()
    assert not cells.duplicated(subset=["image_id", "cell_id"]).any()


def test_wells_are_listed_in_the_summary(uneven_batch):
    uneven_batch.images[0].well = "B3"
    uneven_batch.images[1].well = "B3"
    uneven_batch.images[2].well = "C1"
    assert batch_summary(uneven_batch).loc[0, "wells"] == "B3;C1"


# --------------------------------------------------------------------------- #
# QC interaction
# --------------------------------------------------------------------------- #


def test_excluded_cells_leave_the_pool(uneven_batch):
    before = len(pooled_cells(uneven_batch))
    target = uneven_batch.images[0]
    victim = compute_results(target).cells["cell_id"].iloc[0]
    exclude_object(target.qc, int(victim), "test exclusion")

    after = pooled_cells(uneven_batch)
    assert len(after) == before - 1
    assert not (
        (after["image_id"] == "a.png") & (after["cell_id"] == victim)
    ).any()


def test_qc_log_is_tagged_by_image(uneven_batch):
    exclude_object(uneven_batch.images[1].qc, 1, "debris")
    log = batch_qc_log(uneven_batch)
    assert list(log.columns[:1]) == ["image_id"]
    assert set(log["image_id"]) == {"b.png"}


def test_unreviewed_images_still_contribute(uneven_batch):
    """The review gate is soft: nothing is withheld, but the count is reported."""
    uneven_batch.images[0].reviewed = True
    summary = batch_summary(uneven_batch).iloc[0]
    assert summary["n_images_reviewed"] == 1
    assert summary["n_images_unreviewed"] == 2
    assert summary["n_cells"] == len(pooled_cells(uneven_batch))


def test_per_image_summary_reports_review_state(uneven_batch):
    uneven_batch.images[2].reviewed = True
    frame = per_image_summary(uneven_batch)
    assert list(frame.columns[:3]) == ["image_id", "well", "reviewed"]
    assert frame.set_index("image_id").loc["c.png", "reviewed"]
    assert not frame.set_index("image_id").loc["a.png", "reviewed"]


# --------------------------------------------------------------------------- #
# Calibration consistency guard
# --------------------------------------------------------------------------- #


def test_uniform_calibration_pools_in_microns(uneven_batch):
    verdict = calibration_consistency(uneven_batch)
    assert verdict["consistent"] and verdict["calibrated"]
    assert batch_summary(uneven_batch).loc[0, "units"] == "um"


def test_uniformly_uncalibrated_batch_pools_in_pixels():
    batch = make_batch([("a.png", 2), ("b.png", 3)], calibration=uncalibrated())
    for session in batch.images:
        session.calibration = uncalibrated()
        session.results_cache = None

    summary = batch_summary(batch).iloc[0]
    assert summary["units"] == "px"
    assert "mean_area_px2" in summary.index
    assert "mean_area_um2" not in summary.index


def test_differing_scales_still_pool_because_microns_are_microns():
    """Two magnifications in one batch is fine: each image's µm are already µm."""
    batch = make_batch([("a.png", 2), ("b.png", 3)])
    batch.images[1].calibration = calibration_from_resolution(0.5)
    batch.images[1].results_cache = None

    verdict = calibration_consistency(batch)
    assert verdict["consistent"]
    assert verdict["distinct_scales"] == 2
    assert batch_summary(batch).loc[0, "distinct_calibrations"] == 2


def test_mixing_calibrated_and_uncalibrated_is_refused():
    """px and µm cannot share a column, so no number is produced at all."""
    batch = make_batch([("a.png", 2), ("b.png", 3)])
    batch.images[1].calibration = uncalibrated()
    batch.images[1].results_cache = None

    verdict = calibration_consistency(batch)
    assert not verdict["consistent"]
    assert "mixes calibrated and uncalibrated" in verdict["reason"]
    assert "b.png" in verdict["reason"], "the reason must name the offending image"

    with pytest.raises(CalibrationMismatch, match="mixes calibrated and uncalibrated"):
        batch_summary(batch)


def test_headline_counts_surface_the_mismatch_without_raising():
    batch = make_batch([("a.png", 2), ("b.png", 3)])
    batch.images[1].calibration = uncalibrated()
    batch.images[1].results_cache = None

    counts = headline_counts(batch)
    assert counts["consistent"] is False
    assert counts["n_cells"] == 0
    assert "mixes calibrated" in counts["reason"]


# --------------------------------------------------------------------------- #
# Empty and partial batches
# --------------------------------------------------------------------------- #


def test_empty_batch_produces_empty_frames_not_errors():
    batch = BatchSession(label="empty")
    assert len(pooled_cells(batch)) == 0
    assert len(per_image_summary(batch)) == 0
    summary = batch_summary(batch).iloc[0]
    assert summary["n_images"] == 0 and summary["n_cells"] == 0


def test_unsegmented_images_are_skipped_but_counted():
    batch = make_batch([("a.png", 2)])
    batch.add(AnalysisSession())  # uploaded, never segmented

    summary = batch_summary(batch).iloc[0]
    assert summary["n_images"] == 2
    assert summary["n_images_segmented"] == 1
    assert set(pooled_cells(batch)["image_id"]) == {"a.png"}


def test_failed_image_is_counted_and_does_not_break_pooling():
    batch = make_batch([("a.png", 2)])
    broken = batch.add(AnalysisSession())
    broken.image = ImageRecord("bad.png", "0" * 64, 8, 8, "uint8", 1, np.zeros((8, 8), np.uint8))
    broken.error = "Cellpose failed: out of memory"

    summary = batch_summary(batch).iloc[0]
    assert summary["n_images_failed"] == 1
    assert summary["n_cells"] == len(pooled_cells(batch)) > 0


# --------------------------------------------------------------------------- #
# Shared defaults and per-image override
# --------------------------------------------------------------------------- #


def test_new_images_inherit_batch_defaults():
    batch = BatchSession()
    batch.calibration = calibration_from_resolution(0.33)
    session = batch.add(AnalysisSession())
    assert session.calibration.um_per_px_x == pytest.approx(0.33)
    assert batch.overrides_for(session) == []


def test_override_is_detected_and_reportable():
    batch = BatchSession()
    batch.calibration = calibration_from_resolution(0.33)
    session = batch.add(AnalysisSession())
    session.calibration = calibration_from_resolution(0.66)
    assert batch.overrides_for(session) == ["calibration"]


def test_apply_shared_pushes_defaults_and_clears_caches():
    batch = BatchSession()
    batch.calibration = calibration_from_resolution(0.33)
    session = batch.add(AnalysisSession())
    session.calibration = calibration_from_resolution(0.66)

    assert batch.apply_shared("calibration") == 1
    assert batch.overrides_for(session) == []
    assert session.results_cache is None


def test_summary_is_a_single_row(uneven_batch):
    assert len(batch_summary(uneven_batch)) == 1


def test_dimensionless_metrics_have_no_unit_suffix(uneven_batch):
    summary = batch_summary(uneven_batch).iloc[0]
    for name in ("mean_aspect_ratio", "mean_circularity", "mean_solidity"):
        assert name in summary.index
    assert "mean_aspect_ratio_um" not in summary.index


def test_sd_is_nan_for_a_single_cell():
    """ddof=1 over one value is undefined and must stay NaN, never 0."""
    batch = make_batch([("solo.png", 1)])
    cells = pooled_cells(batch)
    if len(cells) == 1:
        assert pd.isna(batch_summary(batch).loc[0, "sd_area_um2"])


# --------------------------------------------------------------------------- #
# Batch segmentation
# --------------------------------------------------------------------------- #


def test_batch_segmentation_runs_every_image():
    from src.pipeline import run_batch_segmentation

    batch = BatchSession()
    for name in ("a.png", "b.png", "c.png"):
        array = fluorescence_phantom(grid=2, size=240, spacing=90)
        session = AnalysisSession()
        session.image = ImageRecord(name, "0" * 64, *array.shape[:2], "float32", 1, array)
        batch.add(session)

    report = run_batch_segmentation(batch, engine="threshold_watershed")
    assert report["succeeded"] == 3
    assert report["failed"] == []
    assert all(s.has_segmentation for s in batch.images)


def test_one_failing_image_does_not_abort_the_batch():
    """Eleven good images must not be lost to the twelfth."""
    from src.pipeline import run_batch_segmentation

    batch = BatchSession()
    array = fluorescence_phantom(grid=2, size=240, spacing=90)
    good = AnalysisSession()
    good.image = ImageRecord("good.png", "0" * 64, *array.shape[:2], "float32", 1, array)
    batch.add(good)

    broken = AnalysisSession()   # no image at all
    batch.add(broken)

    after = AnalysisSession()
    after.image = ImageRecord("after.png", "0" * 64, *array.shape[:2], "float32", 1, array)
    batch.add(after)

    report = run_batch_segmentation(batch, engine="threshold_watershed")
    assert report["succeeded"] == 2
    assert len(report["failed"]) == 1
    assert good.has_segmentation and after.has_segmentation
    assert broken.error and not broken.has_segmentation
    assert batch.failed == [broken]


def test_progress_is_reported_per_image():
    from src.pipeline import run_batch_segmentation

    batch = BatchSession()
    array = fluorescence_phantom(grid=2, size=240, spacing=90)
    for name in ("a.png", "b.png"):
        session = AnalysisSession()
        session.image = ImageRecord(name, "0" * 64, *array.shape[:2], "float32", 1, array)
        batch.add(session)

    seen = []
    run_batch_segmentation(
        batch, engine="threshold_watershed",
        progress=lambda fraction, desc="": seen.append((fraction, desc)),
    )
    assert len(seen) == 3            # one per image, plus the final tick
    assert seen[-1][0] == 1.0
    assert "a.png" in seen[0][1]


def test_a_successful_rerun_clears_a_previous_error():
    from src.pipeline import run_batch_segmentation

    batch = BatchSession()
    array = fluorescence_phantom(grid=2, size=240, spacing=90)
    session = AnalysisSession()
    session.error = "stale failure from an earlier run"
    session.image = ImageRecord("a.png", "0" * 64, *array.shape[:2], "float32", 1, array)
    batch.add(session)

    run_batch_segmentation(batch, engine="threshold_watershed")
    assert session.error == ""


# --------------------------------------------------------------------------- #
# Nuclear guidance from a separate file
# --------------------------------------------------------------------------- #


def _rgb_record(name, height=240, width=240):
    """An RGB record of arbitrary shape, with signal in the red plane.

    ``fluorescence_phantom`` is square, so it is generated at the larger
    dimension and cropped -- the tests here care about shape agreement, not
    about the pattern itself.
    """
    array = np.zeros((height, width, 3), dtype=np.uint8)
    square = fluorescence_phantom(grid=2, size=max(height, width), spacing=90)
    array[..., 0] = square[:height, :width].astype(np.uint8)
    return ImageRecord(name, "0" * 64, height, width, "uint8", 3, array)


def test_nuclear_channel_from_the_same_file():
    from src.pipeline import nuclear_array_for
    from src.types import SegmentationParams

    session = AnalysisSession()
    session.image = _rgb_record("merge.jpg")
    session.channel = "red"
    session.segmentation_params = SegmentationParams(nuclear_channel="blue")

    guide = nuclear_array_for(session)
    assert guide is not None and guide.shape == (240, 240)


def test_no_nuclear_source_returns_none():
    from src.pipeline import nuclear_array_for

    session = AnalysisSession()
    session.image = _rgb_record("merge.jpg")
    assert nuclear_array_for(session) is None


def test_nuclear_file_of_matching_size_is_used():
    from src.pipeline import nuclear_array_for

    session = AnalysisSession()
    session.image = _rgb_record("cells.jpg")
    session.nuclear_image = _rgb_record("dapi.jpg")
    session.nuclear_image_channel = "blue"

    guide = nuclear_array_for(session)
    assert guide is not None and guide.shape == (240, 240)


def test_mismatched_nuclear_file_is_rejected_and_names_the_file():
    """Guidance from a differently sized file would be silently misaligned."""
    from src.pipeline import nuclear_array_for

    session = AnalysisSession()
    session.image = _rgb_record("cells.jpg", 240, 240)
    session.nuclear_image = _rgb_record("dapi.jpg", 120, 200)

    with pytest.raises(ValueError, match="dapi.jpg") as excinfo:
        nuclear_array_for(session)
    message = str(excinfo.value)
    assert "200x120" in message and "240x240" in message
    assert "cells.jpg" in message


def test_a_separate_nuclear_file_takes_precedence_over_a_channel():
    from src.pipeline import nuclear_array_for
    from src.types import SegmentationParams

    session = AnalysisSession()
    session.image = _rgb_record("cells.jpg")
    session.segmentation_params = SegmentationParams(nuclear_channel="blue")
    session.nuclear_image = _rgb_record("dapi.jpg")
    session.nuclear_image_channel = "red"

    guide = nuclear_array_for(session)
    # The red plane of the attached file carries the phantom; blue of the image
    # is empty. A non-zero maximum proves the file won.
    assert guide.max() > 0


# --------------------------------------------------------------------------- #
# Batch archive
# --------------------------------------------------------------------------- #


def _open_batch(batch, tmp_path):
    import zipfile

    from src.export import export_batch

    path = export_batch(batch, str(tmp_path))
    return zipfile.ZipFile(path), path


def test_archive_has_batch_level_files_and_per_image_folders(uneven_batch, tmp_path):
    archive, _ = _open_batch(uneven_batch, tmp_path)
    names = set(archive.namelist())

    for member in ("batch_summary.csv", "per_image_summary.csv", "all_cells.csv",
                   "all_clusters.csv", "qc_log.csv", "metadata.json"):
        assert member in names, member

    for stem in ("a", "b", "c"):
        for member in ("cell_measurements.csv", "metadata.json",
                       "masks/raw_labels.tif", "masks/qc_labels.tif"):
            assert "images/{}/{}".format(stem, member) in names


def test_pooled_csvs_carry_image_and_well(uneven_batch, tmp_path):
    import io

    uneven_batch.images[0].well = "B3"
    archive, _ = _open_batch(uneven_batch, tmp_path)
    cells = pd.read_csv(io.BytesIO(archive.read("all_cells.csv")))
    assert {"image_id", "well"} <= set(cells.columns)
    assert set(cells.loc[cells["image_id"] == "a.png", "well"]) == {"B3"}


def test_duplicate_filenames_get_separate_folders(tmp_path):
    """Two wells may hold a file of the same name; neither may overwrite the other."""
    batch = make_batch([("field.png", 2), ("field.png", 3)])
    archive, _ = _open_batch(batch, tmp_path)
    names = set(archive.namelist())
    assert "images/field/cell_measurements.csv" in names
    assert "images/field_2/cell_measurements.csv" in names


def test_batch_metadata_records_provenance_and_the_aggregation_rule(uneven_batch, tmp_path):
    import json

    uneven_batch.label = "shRNA FST experiment"
    uneven_batch.images[0].well = "B3"
    uneven_batch.images[1].reviewed = True

    archive, _ = _open_batch(uneven_batch, tmp_path)
    meta = json.loads(archive.read("metadata.json"))

    assert meta["batch_label"] == "shRNA FST experiment"
    assert meta["n_images"] == 3
    assert meta["n_images_reviewed"] == 1
    assert meta["n_images_unreviewed"] == 2
    assert set(meta["unreviewed_images"]) == {"a.png", "c.png"}

    rule = meta["aggregation"]
    assert rule["weighted_by"] == "cell count"
    assert rule["units"] == "um"
    assert rule["calibration_consistent"] is True
    assert "not independent biological replicates" in rule["caveat"]

    assert meta["images"]["a"]["well"] == "B3"
    assert meta["images"]["a"]["sha256"]


def test_metadata_records_which_images_overrode_shared_settings(uneven_batch, tmp_path):
    import json

    uneven_batch.images[1].calibration = calibration_from_resolution(0.5)
    uneven_batch.images[1].results_cache = None

    archive, _ = _open_batch(uneven_batch, tmp_path)
    meta = json.loads(archive.read("metadata.json"))
    assert meta["images"]["b"]["overrides"] == ["calibration"]
    assert meta["images"]["a"]["overrides"] == []


def test_per_image_metadata_reports_its_own_segmentation_channel(uneven_batch, tmp_path):
    """The single-image provenance guarantee must survive batching."""
    import json

    archive, _ = _open_batch(uneven_batch, tmp_path)
    inner = json.loads(archive.read("images/a/metadata.json"))
    assert inner["segmentation_channel"] == uneven_batch.images[0].segmented_channel


def test_archive_leaves_no_staging_directory(uneven_batch, tmp_path):
    from src.export import export_batch

    export_batch(uneven_batch, str(tmp_path))
    assert [p.name for p in tmp_path.iterdir() if p.name.startswith("cellscope_")] == []


def test_reproducible_from_the_exported_mask(uneven_batch, tmp_path):
    """§22.18 through a batch: re-derive areas from a shipped mask."""
    import io
    import json

    import tifffile

    archive, _ = _open_batch(uneven_batch, tmp_path)
    labels = tifffile.imread(io.BytesIO(archive.read("images/a/masks/qc_labels.tif")))
    meta = json.loads(archive.read("images/a/metadata.json"))
    sx = meta["calibration"]["um_per_px_x"]
    sy = meta["calibration"]["um_per_px_y"]

    cells = pd.read_csv(io.BytesIO(archive.read("images/a/cell_measurements.csv")))
    for _, row in cells.iterrows():
        pixels = int((labels == row["cell_id"]).sum())
        assert pixels == pytest.approx(row["area_px2"])
        assert pixels * sx * sy == pytest.approx(row["area_um2"], rel=1e-9)


def test_mixed_calibration_batch_cannot_be_exported(tmp_path):
    """The archive must not ship a summary the guard refuses to compute."""
    from src.export import export_batch

    batch = make_batch([("a.png", 2), ("b.png", 3)])
    batch.images[1].calibration = uncalibrated()
    batch.images[1].results_cache = None

    with pytest.raises(CalibrationMismatch):
        export_batch(batch, str(tmp_path))


# --------------------------------------------------------------------------- #
# Skipping work that is already done
# --------------------------------------------------------------------------- #


def _pending_batch(n=3):
    from src.types import BatchSession

    batch = BatchSession()
    array = fluorescence_phantom(grid=2, size=240, spacing=90)
    for i in range(n):
        session = AnalysisSession()
        session.image = ImageRecord(
            "img{}.png".format(i), "0" * 64, *array.shape[:2], "float32", 1, array
        )
        batch.add(session)
    return batch


def test_only_missing_skips_images_already_segmented():
    """Adding one field to a batch of twelve must not re-run the eleven."""
    from src.pipeline import run_batch_segmentation

    batch = _pending_batch(3)
    first = run_batch_segmentation(batch, engine="threshold_watershed")
    assert first["succeeded"] == 3 and first["skipped"] == 0

    versions = [s.segmentation_version for s in batch.images]
    second = run_batch_segmentation(
        batch, engine="threshold_watershed", only_missing=True
    )
    assert second["total"] == 0
    assert second["skipped"] == 3
    assert [s.segmentation_version for s in batch.images] == versions, (
        "already-segmented images were re-run"
    )


def test_only_missing_still_runs_the_new_image():
    from src.pipeline import run_batch_segmentation

    batch = _pending_batch(2)
    run_batch_segmentation(batch, engine="threshold_watershed")
    versions = [s.segmentation_version for s in batch.images]

    array = fluorescence_phantom(grid=2, size=240, spacing=90)
    added = AnalysisSession()
    added.image = ImageRecord("new.png", "0" * 64, *array.shape[:2], "float32", 1, array)
    batch.add(added)

    report = run_batch_segmentation(
        batch, engine="threshold_watershed", only_missing=True
    )
    assert report["total"] == 1 and report["succeeded"] == 1 and report["skipped"] == 2
    assert added.has_segmentation
    assert [s.segmentation_version for s in batch.images[:2]] == versions


def test_only_missing_retries_a_failed_image():
    """A failure is not 'done'; it must be attempted again."""
    from src.pipeline import run_batch_segmentation

    batch = _pending_batch(1)
    run_batch_segmentation(batch, engine="threshold_watershed")
    batch.images[0].error = "transient failure"

    report = run_batch_segmentation(
        batch, engine="threshold_watershed", only_missing=True
    )
    assert report["total"] == 1
    assert batch.images[0].error == ""


def test_progress_fractions_span_only_the_images_actually_run():
    from src.pipeline import run_batch_segmentation

    batch = _pending_batch(3)
    run_batch_segmentation(batch, engine="threshold_watershed")

    array = fluorescence_phantom(grid=2, size=240, spacing=90)
    added = AnalysisSession()
    added.image = ImageRecord("new.png", "0" * 64, *array.shape[:2], "float32", 1, array)
    batch.add(added)

    seen = []
    run_batch_segmentation(
        batch, engine="threshold_watershed", only_missing=True,
        progress=lambda fraction, desc="": seen.append((fraction, desc)),
    )
    assert seen[0][0] == 0.0 and "new.png" in seen[0][1]
    assert "(1/1)" in seen[0][1], "progress must count the run set, not the whole batch"
    assert seen[-1][0] == 1.0


# --------------------------------------------------------------------------- #
# Analysis resolution
#
# Downscaling is an inference-only optimisation. The invariant that keeps
# calibration, measurement and the reproducibility chain valid is that masks
# always come back on the original pixel grid.
# --------------------------------------------------------------------------- #


def test_scale_is_clamped_to_something_runnable():
    from src.segmentation import _valid_scale

    assert _valid_scale(1.0) == 1.0
    assert _valid_scale(0.5) == 0.5
    assert _valid_scale(1.5) == 1.0        # never upscale
    for bad in (0, -1, None, "x", float("nan"), float("inf")):
        assert _valid_scale(bad) == 1.0


def test_upscaling_returns_the_original_grid():
    from src.segmentation import _upscale_labels

    small = np.array([[0, 3], [5, 5]], dtype=np.int32)
    up = _upscale_labels(small, (8, 8))
    assert up.shape == (8, 8)


def test_upscaling_never_invents_a_label():
    """Nearest-neighbour only: interpolating between IDs 3 and 5 would create a
    label 4 that names no object anyone segmented."""
    from src.segmentation import _upscale_labels

    small = np.array([[0, 3], [5, 5]], dtype=np.int32)
    up = _upscale_labels(small, (16, 16))
    assert set(np.unique(up)) <= set(np.unique(small))
    assert 4 not in set(np.unique(up))


def test_upscaling_preserves_integer_dtype():
    from src.segmentation import _upscale_labels

    up = _upscale_labels(np.array([[1, 2]], dtype=np.int32), (4, 4))
    assert np.issubdtype(up.dtype, np.integer)


def test_downscaled_run_returns_full_resolution_masks():
    """The invariant. Uses the fallback engine so it runs without Cellpose."""
    from src.segmentation import _rescale_stack

    array = fluorescence_phantom(grid=3, size=320, spacing=90)
    stack = np.stack([array, array, array], -1)
    small = _rescale_stack(stack, 0.5)
    assert small.shape[:2] == (160, 160)
    assert small.shape[2] == 3


def test_default_scale_leaves_params_unchanged():
    from src.types import SegmentationParams

    assert SegmentationParams().analysis_scale == 1.0


def test_analysis_scale_reaches_the_export(tmp_path):
    """§18: a factor that changed what was segmented must appear in provenance."""
    import json
    import zipfile

    from src.export import export_batch

    batch = make_batch([("a.png", 2)])
    session = batch.images[0]
    session.engine_info = {**session.engine_info, "analysis_scale": 0.5}

    archive = export_batch(batch, str(tmp_path))
    with zipfile.ZipFile(archive) as z:
        meta = json.loads(z.read("images/a/metadata.json"))
    assert meta["segmentation_engine"]["analysis_scale"] == 0.5


# --------------------------------------------------------------------------- #
# Streaming batch segmentation
# --------------------------------------------------------------------------- #


def test_generator_yields_once_per_image_in_order():
    from src.pipeline import iter_batch_segmentation

    batch = _pending_batch(3)
    seen = list(iter_batch_segmentation(batch, engine="threshold_watershed"))

    assert len(seen) == 3
    assert [position for position, _, _, _ in seen] == [0, 1, 2]
    assert [s.display_name for _, _, s, _ in seen] == ["img0.png", "img1.png", "img2.png"]
    assert all(total == 3 for _, total, _, _ in seen)


def test_each_yield_carries_the_running_tally():
    from src.pipeline import iter_batch_segmentation

    batch = _pending_batch(3)
    counts = [
        report["succeeded"]
        for _, _, _, report in iter_batch_segmentation(batch, engine="threshold_watershed")
    ]
    assert counts == [1, 2, 3]


def test_the_image_is_segmented_by_the_time_it_is_yielded():
    """Streaming is only useful if the yielded image is ready to review."""
    from src.pipeline import iter_batch_segmentation

    batch = _pending_batch(2)
    for _, _, session, _ in iter_batch_segmentation(batch, engine="threshold_watershed"):
        assert session.has_segmentation


def test_wrapper_report_matches_the_generators_final_report():
    from src.pipeline import iter_batch_segmentation, run_batch_segmentation

    streamed = _pending_batch(3)
    final = None
    for _, _, _, final in iter_batch_segmentation(streamed, engine="threshold_watershed"):
        pass

    blocking = _pending_batch(3)
    report = run_batch_segmentation(blocking, engine="threshold_watershed")
    assert report == final


def test_generator_honours_only_missing():
    from src.pipeline import iter_batch_segmentation, run_batch_segmentation

    batch = _pending_batch(3)
    run_batch_segmentation(batch, engine="threshold_watershed")
    seen = list(
        iter_batch_segmentation(batch, engine="threshold_watershed", only_missing=True)
    )
    assert seen == []


def test_skip_count_survives_a_run_that_touches_nothing():
    """The generator never yields when everything is skipped, so the wrapper
    must not learn the skip count from it."""
    from src.pipeline import run_batch_segmentation

    batch = _pending_batch(3)
    run_batch_segmentation(batch, engine="threshold_watershed")
    report = run_batch_segmentation(
        batch, engine="threshold_watershed", only_missing=True
    )
    assert report["total"] == 0
    assert report["skipped"] == 3


def test_a_failure_mid_stream_does_not_stop_the_generator():
    from src.pipeline import iter_batch_segmentation

    batch = _pending_batch(1)
    broken = AnalysisSession()          # no image
    batch.add(broken)
    array = fluorescence_phantom(grid=2, size=240, spacing=90)
    tail = AnalysisSession()
    tail.image = ImageRecord("tail.png", "0" * 64, *array.shape[:2], "float32", 1, array)
    batch.add(tail)

    seen = list(iter_batch_segmentation(batch, engine="threshold_watershed"))
    assert len(seen) == 3
    assert seen[-1][3]["succeeded"] == 2
    assert len(seen[-1][3]["failed"]) == 1
    assert tail.has_segmentation


# --------------------------------------------------------------------------- #
# Model cache concurrency
#
# The app preloads Cellpose weights in a background thread while the user is
# still uploading. Without a lock, pressing Run before that finishes lets both
# threads miss the cache and each build a full model -- measured at 49 s to
# obtain a model, against 10 s once serialised.
# --------------------------------------------------------------------------- #


def test_concurrent_callers_build_the_model_only_once(monkeypatch):
    import threading
    import time

    import src.segmentation as segmentation

    built = []

    class FakeModel:
        def __init__(self, *args, **kwargs):
            built.append(1)
            time.sleep(0.3)          # a load slow enough for a second caller to race

    monkeypatch.setattr(segmentation, "_MODEL_CACHE", {})
    monkeypatch.setattr(
        segmentation, "_MODEL_LOCK", threading.Lock()
    )

    fake_module = type("cp", (), {"CellposeModel": FakeModel})
    monkeypatch.setitem(
        sys.modules, "cellpose", type("m", (), {"models": fake_module})
    )
    monkeypatch.setitem(sys.modules, "cellpose.models", fake_module)

    results = []
    threads = [
        threading.Thread(
            target=lambda: results.append(
                segmentation._get_cellpose_model("fake", False)
            )
        )
        for _ in range(4)
    ]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert len(built) == 1, "model built {} times; the cache lock is not holding".format(
        len(built)
    )
    assert len({id(r) for r in results}) == 1, "callers received different models"


def test_second_caller_receives_the_cached_model(monkeypatch):
    import threading

    import src.segmentation as segmentation

    monkeypatch.setattr(segmentation, "_MODEL_CACHE", {})
    monkeypatch.setattr(segmentation, "_MODEL_LOCK", threading.Lock())

    class FakeModel:
        def __init__(self, *args, **kwargs):
            pass

    fake_module = type("cp", (), {"CellposeModel": FakeModel})
    monkeypatch.setitem(sys.modules, "cellpose", type("m", (), {"models": fake_module}))
    monkeypatch.setitem(sys.modules, "cellpose.models", fake_module)

    first = segmentation._get_cellpose_model("fake", False)
    assert segmentation._get_cellpose_model("fake", False) is first


def test_model_is_loaded_reports_cache_state(monkeypatch):
    import src.segmentation as segmentation

    monkeypatch.setattr(segmentation, "_MODEL_CACHE", {})
    assert segmentation.model_is_loaded("cpsam", use_gpu=False) is False
    segmentation._MODEL_CACHE[("cpsam", False)] = object()
    assert segmentation.model_is_loaded("cpsam", use_gpu=False) is True
