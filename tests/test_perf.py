"""Smoke tests for the benchmark harness, so it cannot silently rot."""

from __future__ import annotations

import json

import numpy as np

from src import perf


def test_synthetic_field_is_seeded_and_rgb():
    a = perf.synthetic_field(200, 240, n_cells=6, seed=3)
    b = perf.synthetic_field(200, 240, n_cells=6, seed=3)
    assert a.shape == (200, 240, 3) and a.dtype == np.uint8
    np.testing.assert_array_equal(a, b)
    assert a[..., 0].max() > 100 and a[..., 1].max() < 50


def test_ruler_is_drawn_in_the_bottom_right():
    rgb = np.zeros((800, 800, 3), np.uint8)
    top, left, bottom, right = perf.draw_ruler(rgb)
    assert (bottom - top, right - left) == (8, 126)
    assert top > 600 and left > 600
    assert tuple(rgb[top, left]) == perf.RULER_RGB


def test_dense_labels_are_crowded():
    labels = perf.dense_labels(size=240, cell=40)
    assert labels.max() >= 20


def test_pipeline_benchmark_reports_every_stage():
    batch = perf.synthetic_batch(2, size=240, n_cells=6)
    result = perf.bench_pipeline(batch)
    for stage in ("preprocess", "split", "measure_cells", "measure_clusters", "overlay"):
        assert stage in result["stages"]
    assert result["images"] == 2
    json.dumps(result)


def test_bench_cli_quick_run(tmp_path, monkeypatch):
    """The CLI end to end, on tiny fields so it stays fast."""
    import pytest

    pytest.importorskip("gradio")
    import tools.bench as bench

    original = perf.synthetic_batch
    monkeypatch.setattr(
        perf, "synthetic_batch",
        lambda n, **kw: original(min(n, 2), size=240, n_cells=6, ruler_every=0),
    )
    assert bench.main(["--runs", "1", "--out", str(tmp_path)]) == 0
    written = list(tmp_path.glob("*.json"))
    assert len(written) == 1
    report = json.loads(written[0].read_text())
    assert {"batch_1", "batch_10", "ui_10", "persistence_10"} <= set(report["workloads"])
    assert report["workloads"]["ui_10"]["opacity_change"]["median_ms"] >= 0
