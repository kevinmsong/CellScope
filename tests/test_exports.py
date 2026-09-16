"""Replicate-aware summaries, the export manifest, and atomic archives."""

from __future__ import annotations

import hashlib
import json
import os
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.calibration import calibration_from_resolution
from src.export import export_analysis, export_batch
from src.pipeline import compute_results
from src.quality import design_summary
from src.types import AnalysisSession, BatchSession, ImageRecord, ObjectRecord


def _field(name, sides, well, condition, replicate, calibration=None):
    """An image holding one square cell per entry of ``sides`` (area = side**2)."""
    labels = np.zeros((120, 400), np.int32)
    objects = []
    for index, side in enumerate(sides, start=1):
        left = 10 + (index - 1) * 60
        labels[20 : 20 + side, left : left + side] = index
        objects.append(ObjectRecord(index, index, "resolved"))
    s = AnalysisSession()
    pixels = np.zeros((120, 400, 3), np.uint8)
    pixels[labels > 0, 0] = 200
    pixels.flags.writeable = False
    s.image = ImageRecord(name, hashlib.sha256(name.encode()).hexdigest(), 120, 400, "uint8", 3, pixels)
    labels.flags.writeable = False
    s.raw_labels = labels
    s.object_labels = labels
    s.objects = objects
    s.segmentation_version = 1
    s.segmented_channel = "red"
    s.well, s.condition, s.replicate = well, condition, replicate
    if calibration is not None:
        s.calibration = calibration
    return s


@pytest.fixture
def design_batch():
    batch = BatchSession(label="design")
    fields = [
        # control, replicate R1: well A1 has two fields, A2 one.
        ("c1.png", [10, 10], "A1", "control", "R1"),       # areas 100, 100
        ("c2.png", [20], "A1", "control", "R1"),           # 400 -> A1 mean 200
        ("c3.png", [30], "A2", "control", "R1"),           # 900 -> A2 mean 900
        # control, replicate R2
        ("c4.png", [10], "B1", "control", "R2"),           # 100
        # treated, one replicate
        ("t1.png", [40, 20], "C1", "treated", "R1"),       # 1600, 400 -> 1000
        # not labelled
        ("u1.png", [10], "", "treated", "R9"),
    ]
    for name, sides, well, condition, replicate in fields:
        batch.images.append(_field(name, sides, well, condition, replicate))
    return batch


def test_wells_pool_fields_and_replicates_weight_wells_equally(design_batch):
    summary = design_summary(design_batch, "area")
    wells = summary["wells"].set_index(["condition", "biological_replicate", "well"])
    assert wells.loc[("control", "R1", "A1"), "fields"] == 2
    assert wells.loc[("control", "R1", "A1"), "cells"] == 3
    assert wells.loc[("control", "R1", "A1"), "mean"] == pytest.approx(200)
    assert wells.loc[("control", "R1", "A2"), "mean"] == pytest.approx(900)

    reps = summary["replicates"].set_index(["condition", "biological_replicate"])
    # Equal weight per well: (200 + 900) / 2, not the cell-weighted 375.
    assert reps.loc[("control", "R1"), "mean"] == pytest.approx(550)
    assert reps.loc[("control", "R1"), "wells"] == 2
    assert reps.loc[("control", "R1"), "sd_between_wells"] == pytest.approx(np.std([200, 900], ddof=1))

    conditions = summary["conditions"].set_index("condition")
    assert conditions.loc["control", "biological_replicates"] == 2
    assert conditions.loc["control", "mean"] == pytest.approx((550 + 100) / 2)
    assert conditions.loc["control", "cells"] == 5
    assert conditions.loc["treated", "biological_replicates"] == 1
    assert np.isnan(conditions.loc["treated", "sd_between_replicates"]), "one replicate has no SD"


def test_unlabelled_images_are_listed_not_guessed(design_batch):
    summary = design_summary(design_batch, "area")
    assert list(summary["unassigned"]["image"]) == ["u1.png"]
    assert "u1.png" not in set(summary["images"]["image"])


def test_units_are_never_mixed(design_batch):
    design_batch.images[3].calibration = calibration_from_resolution(0.5)
    design_batch.images[3].results_cache = None
    summary = design_summary(design_batch, "area")
    reps = summary["replicates"]
    assert set(reps["unit"]) == {"px²", "µm²"}
    control_um = reps[(reps.condition == "control") & (reps.unit == "µm²")]
    assert control_um["mean"].iloc[0] == pytest.approx(25.0)
    conditions = summary["conditions"]
    assert len(conditions[conditions.condition == "control"]) == 2


def test_shape_metrics_have_no_unit(design_batch):
    summary = design_summary(design_batch, "aspect_ratio")
    assert set(summary["replicates"]["unit"]) == {"ratio"}
    assert summary["conditions"]["mean"].iloc[0] == pytest.approx(1.0)


def test_unknown_metric_is_rejected(design_batch):
    with pytest.raises(ValueError):
        design_summary(design_batch, "volume")


def test_empty_design_returns_empty_frames():
    summary = design_summary(BatchSession(), "area")
    assert all(frame.empty for frame in summary.values())


def test_replicate_dot_plot_shows_replicate_n(design_batch):
    from src.visualize import replicate_dot_plot

    summary = design_summary(design_batch, "area")
    figure = replicate_dot_plot(summary["replicates"], summary["conditions"], "Cell area")
    ticks = list(figure.layout.xaxis.ticktext)
    assert ticks == ["control (n=2)", "treated (n=1)"]
    assert len(figure.data[0].y) == 3, "one dot per biological replicate"


# --------------------------------------------------------------------------- #
# Batch archive
# --------------------------------------------------------------------------- #


def test_batch_archive_has_a_verifiable_manifest(design_batch, tmp_path):
    path = export_batch(design_batch, str(tmp_path))
    with zipfile.ZipFile(path) as archive:
        names = set(archive.namelist())
        manifest = json.loads(archive.read("manifest.json"))
        for entry in manifest["files"]:
            assert entry["path"] in names
            assert hashlib.sha256(archive.read(entry["path"])).hexdigest() == entry["sha256"]
        assert "README.txt" in names
        assert {"design_wells.csv", "design_replicates.csv", "design_conditions.csv"} <= names
        conditions = pd.read_csv(archive.open("design_conditions.csv"))
    assert manifest["manifest_schema"] == 1
    assert manifest["software"]["cellscope"]
    assert "numpy" in manifest["software"]
    assert "replicates" in manifest["statistics"]["replicate_rule"]
    assert manifest["statistics"]["tests"] == "none"
    assert len(manifest["images"]) == len(design_batch.images)
    first = manifest["images"][0]
    for key in ("source_image", "design", "calibration", "parameters", "inference", "qc",
                "corrections", "review", "segmentation_fingerprint", "folder"):
        assert key in first
    assert first["folder"].startswith("images/")
    assert set(conditions["condition"]) == {"control", "treated"}


def test_a_failed_export_leaves_no_partial_archive(design_batch, tmp_path, monkeypatch):
    first = export_batch(design_batch, str(tmp_path))
    original = Path(first).read_bytes()

    import src.export as export

    def broken(self, *args, **kwargs):
        raise OSError("disk full")

    monkeypatch.setattr(export.zipfile.ZipFile, "write", broken)
    with pytest.raises(OSError):
        export_batch(design_batch, str(tmp_path))
    assert Path(first).read_bytes() == original
    folder = os.path.dirname(first)
    assert not [f for f in os.listdir(folder) if f.endswith(".tmp")]


def test_single_image_export_cleans_up_after_a_failure(design_batch, tmp_path, monkeypatch):
    import tempfile

    import src.export as export

    session = design_batch.images[0]
    results = compute_results(session)
    before = set(os.listdir(tempfile.gettempdir()))

    def broken(*args, **kwargs):
        raise OSError("cannot write")

    monkeypatch.setattr(export, "write_mask", broken)
    with pytest.raises(OSError):
        export_analysis(session, results, str(tmp_path))
    leaked = {f for f in set(os.listdir(tempfile.gettempdir())) - before
              if f.startswith("cellscope_stage_")}
    assert not leaked


def test_per_image_metadata_records_inference_and_annotation(tmp_path):
    from src.perf import image_record, synthetic_field
    from src.pipeline import run_segmentation

    rgb = synthetic_field(800, 800, 10, seed=1, ruler=True)
    session = AnalysisSession()
    session.image = image_record(rgb, "ruler.png")
    run_segmentation(session, rgb[..., 0].astype(np.float32), engine="threshold_watershed")
    path = export_analysis(session, compute_results(session), str(tmp_path))
    with zipfile.ZipFile(path) as archive:
        metadata = json.loads(archive.read("metadata.json"))
        cells = pd.read_csv(archive.open("cell_measurements.csv"))
    assert metadata["burned_in_annotation"]["found"]
    assert metadata["automatic_qc"] == {"exclude_border": True, "exclude_annotations": True}
    assert metadata["software"]["versions"]["cellscope"]
    assert "touches_annotation" in cells.columns
    assert not cells["touches_annotation"].any(), "accepted cells never touch the ruler"
    assert any(o["touches_annotation"] for o in metadata["object_provenance"])


def test_report_includes_the_design_and_status(design_batch, tmp_path):
    from src.report import write_report

    path = write_report(design_batch, tmp_path / "r.pdf")
    data = Path(path).read_bytes()
    assert data.startswith(b"%PDF") and len(data) > 5000
