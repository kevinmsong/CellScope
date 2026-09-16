"""Pipelined batch segmentation, cancellation, and CUDA out-of-memory recovery."""

from __future__ import annotations

import copy
import threading
import time

import numpy as np
import pytest

import src.pipeline as pipeline
import src.segmentation as segmentation
from src.perf import image_record, synthetic_field
from src.pipeline import iter_batch_segmentation, run_batch_segmentation
from src.types import AnalysisSession, BatchSession, SegmentationParams


def _batch(n=5, size=260):
    batch = BatchSession()
    batch.channel = "red"
    for index in range(n):
        session = AnalysisSession()
        session.image = image_record(synthetic_field(size, size, 10, seed=index), "f{}.png".format(index))
        batch.add(session)
        batch.apply_channels(session)
    return batch


def _labels(batch):
    return [None if s.object_labels is None else s.object_labels.copy() for s in batch.images]


@pytest.mark.parametrize("workers", [1, 2, 4])
def test_pipelined_results_equal_sequential_results(workers):
    sequential, pipelined = _batch(), _batch()
    list(iter_batch_segmentation(sequential, engine="threshold_watershed", workers=0))
    list(iter_batch_segmentation(pipelined, engine="threshold_watershed", workers=workers))
    for a, b in zip(sequential.images, pipelined.images):
        np.testing.assert_array_equal(a.object_labels, b.object_labels)
        np.testing.assert_array_equal(a.raw_labels, b.raw_labels)
        assert a.objects == b.objects
        assert a.qc.excluded_ids == b.qc.excluded_ids
        assert a.segmented_with == b.segmented_with


def test_results_are_yielded_in_input_order_with_progress():
    batch = _batch(4)
    seen = list(iter_batch_segmentation(batch, engine="threshold_watershed", workers=2))
    assert [position for position, *_ in seen] == [0, 1, 2, 3]
    assert [session for _, _, session, _ in seen] == batch.images
    report = seen[-1][3]
    assert [state for _, state in report["status"]] == ["done"] * 4
    assert report["seconds_per_image"] > 0
    assert report["eta_seconds"] == 0
    assert seen[0][3]["eta_seconds"] >= 0


class _SlowEngine:
    """Wraps inference with a delay and records how many images are in flight."""

    def __init__(self, delay=0.05, fail_on=(), oom_on=()):
        self.delay = delay
        self.fail_on = set(fail_on)
        self.live = 0
        self.peak = 0
        self.lock = threading.Lock()
        self.order = []

    def install(self, monkeypatch):
        original_prepare = pipeline.prepare_segmentation
        original_commit = pipeline.commit_segmentation
        original_infer = pipeline.infer_segmentation

        def prepare(session, *args):
            with self.lock:
                self.live += 1
                self.peak = max(self.peak, self.live)
            return original_prepare(session, *args)

        def infer(prepared, engine):
            name = prepared.session.display_name
            self.order.append(name)
            time.sleep(self.delay)
            if name in self.fail_on:
                raise RuntimeError("inference failed for " + name)
            return original_infer(prepared, engine)

        def commit(session, outcome):
            with self.lock:
                self.live -= 1
            return original_commit(session, outcome)

        monkeypatch.setattr(pipeline, "prepare_segmentation", prepare)
        monkeypatch.setattr(pipeline, "infer_segmentation", infer)
        monkeypatch.setattr(pipeline, "commit_segmentation", commit)


def test_inference_is_serial_and_memory_is_bounded(monkeypatch):
    engine = _SlowEngine()
    engine.install(monkeypatch)
    batch = _batch(6)
    list(iter_batch_segmentation(batch, engine="threshold_watershed", workers=2))
    assert engine.order == [s.display_name for s in batch.images]
    assert engine.peak <= 3, "at most three images may be held at once"


def test_a_failed_image_does_not_stop_a_pipelined_run(monkeypatch):
    engine = _SlowEngine(fail_on={"f1.png"})
    engine.install(monkeypatch)
    batch = _batch(4)
    broken = AnalysisSession()               # no image: fails while preparing
    batch.images.insert(3, broken)
    seen = list(iter_batch_segmentation(batch, engine="threshold_watershed", workers=2))
    report = seen[-1][3]
    assert report["succeeded"] == 3
    assert {name for name, _ in report["failed"]} == {"f1.png", "(no image)"}
    assert "inference failed" in batch.images[1].error
    assert batch.images[1].object_labels is None
    assert batch.images[4].has_segmentation
    assert [state for _, state in report["status"]] == ["done", "failed", "done", "failed", "done"]


def test_cancelling_stops_between_images(monkeypatch):
    engine = _SlowEngine(delay=0.1)
    engine.install(monkeypatch)
    batch = _batch(6)
    cancel = threading.Event()
    seen = []
    for item in iter_batch_segmentation(batch, engine="threshold_watershed", cancel=cancel, workers=2):
        seen.append(item)
        cancel.set()
    report = seen[-1][3]
    assert len(seen) == 1
    assert report["cancelled"]
    assert batch.images[0].has_segmentation
    assert not any(s.has_segmentation for s in batch.images[2:])
    assert [state for _, state in report["status"]][2:] == ["cancelled"] * 4


def test_abandoning_the_generator_commits_nothing_more(monkeypatch):
    """Gradio's cancel closes the generator at a yield."""
    engine = _SlowEngine(delay=0.1)
    engine.install(monkeypatch)
    batch = _batch(5)
    run = iter_batch_segmentation(batch, engine="threshold_watershed", workers=2)
    next(run)
    run.close()
    time.sleep(0.5)
    assert batch.images[0].has_segmentation
    assert not any(s.has_segmentation for s in batch.images[1:])


def test_sequential_cancellation_also_stops():
    batch = _batch(3)
    cancel = threading.Event()
    seen = []
    for item in iter_batch_segmentation(batch, engine="threshold_watershed", cancel=cancel, workers=0):
        seen.append(item)
        cancel.set()
    assert len(seen) == 1 and seen[0][3]["cancelled"]


def test_wrapper_uses_the_pipeline_and_reports_the_same():
    a, b = _batch(3), _batch(3)
    streamed = None
    for *_, streamed in iter_batch_segmentation(a, engine="threshold_watershed"):
        pass
    assert run_batch_segmentation(b, engine="threshold_watershed") == streamed


def test_a_batch_with_a_lock_can_still_be_deep_copied():
    batch = _batch(1)
    with batch.lock():
        clone = copy.deepcopy(batch)
    assert clone.lock() is not batch.lock()
    assert clone.images[0].image.filename == "f0.png"


# --------------------------------------------------------------------------- #
# Out-of-memory recovery and the "auto" engine
# --------------------------------------------------------------------------- #


class _OOM(RuntimeError):
    def __init__(self):
        super().__init__("CUDA out of memory. Tried to allocate 1.2 GiB")


def _fake_cellpose(monkeypatch, gpu_behaviour, cpu_calls, tile_batch=4):
    class GPUModel:
        def eval(self, image, batch_size, **kwargs):
            return gpu_behaviour(image, batch_size)

    class CPUModel:
        def eval(self, image, batch_size, **kwargs):
            cpu_calls.append(batch_size)
            return (np.ones(image.shape[:2], np.int32),)

    monkeypatch.setattr(segmentation, "cellpose_available", lambda: True)
    monkeypatch.setattr(segmentation, "_resolve_device", lambda use: True)
    monkeypatch.setattr(segmentation, "_inference_tile_batch", lambda use: tile_batch)
    monkeypatch.setattr(
        segmentation, "_get_cellpose_model",
        lambda name, gpu: GPUModel() if gpu else CPUModel(),
    )


def test_oom_retries_with_a_smaller_tile_batch(monkeypatch):
    tried = []

    def gpu(image, batch_size):
        tried.append(batch_size)
        if batch_size > 1:
            raise _OOM()
        return (np.ones(image.shape[:2], np.int32),)

    _fake_cellpose(monkeypatch, gpu, [])
    labels, info = segmentation.segment_with_cellpose(np.zeros((40, 60)), SegmentationParams())
    assert tried == [4, 1]
    assert info["tile_batch_size"] == 1
    assert info["device"] == "cuda"
    assert "retried with 1" in info["oom_recovery"][0]


def test_persistent_oom_falls_back_to_the_cpu_and_says_so(monkeypatch):
    cpu_calls = []

    def gpu(image, batch_size):
        raise _OOM()

    _fake_cellpose(monkeypatch, gpu, cpu_calls)
    labels, info = segmentation.segment_with_cellpose(np.zeros((40, 60)), SegmentationParams())
    assert cpu_calls == [1]
    assert info["device"] == "cpu"
    assert any("CPU" in step for step in info["oom_recovery"])
    assert labels.shape == (40, 60)


def test_other_gpu_errors_are_not_swallowed(monkeypatch):
    def gpu(image, batch_size):
        raise ValueError("bad input")

    _fake_cellpose(monkeypatch, gpu, [])
    with pytest.raises(ValueError, match="bad input"):
        segmentation.segment_with_cellpose(np.zeros((40, 60)), SegmentationParams())


def test_auto_engine_reports_a_cellpose_failure_instead_of_switching_algorithm(monkeypatch):
    def gpu(image, batch_size):
        raise RuntimeError("cuDNN error")

    _fake_cellpose(monkeypatch, gpu, [])
    with pytest.raises(RuntimeError, match="cuDNN"):
        segmentation.segment_cells(np.zeros((40, 60), np.float32), engine="auto")


def test_auto_engine_falls_back_when_the_model_cannot_load(monkeypatch):
    def unavailable(name, gpu):
        raise segmentation.CellposeUnavailable("weights missing")

    monkeypatch.setattr(segmentation, "cellpose_available", lambda: True)
    monkeypatch.setattr(segmentation, "_get_cellpose_model", unavailable)
    image = np.zeros((60, 60), np.float32)
    image[20:40, 20:40] = 1
    labels, info = segmentation.segment_cells(image, engine="auto")
    assert info["engine"] == "threshold_watershed"
    assert "weights missing" in info["cellpose_error"]
