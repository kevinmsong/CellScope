"""Performance measurement for the scientific pipeline.

Timings are only useful when they are repeatable, so every workload here is
built from seeded synthetic fields rather than lab images (which never enter the
repository). ``tools/bench.py`` is the command-line front end and adds UI
callback latency on top; this module deliberately does not import Gradio.

Nothing here changes a measurement. It calls the same functions the app calls,
in the same order, and records how long each took and how much memory the
process held while doing it.
"""

from __future__ import annotations

import os
import platform
import statistics
import tempfile
import threading
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, Callable, Iterator

import numpy as np
from skimage import draw

from . import __version__

# --------------------------------------------------------------------------- #
# Synthetic fields
# --------------------------------------------------------------------------- #

#: Colour of the burned-in ruler on the reference microscope's exports.
RULER_RGB = (252, 252, 20)


def draw_ruler(rgb: np.ndarray, bar_px: int = 126, colour=RULER_RGB) -> tuple[int, int, int, int]:
    """Burn a scale bar and a blocky "50 um" caption into the bottom-right corner.

    Geometry follows the real exports: an 8 px bar about 80 px above the lower
    edge, with glyphs roughly 34 px tall beneath it. Returns the bar's bbox as
    ``(min_row, min_col, max_row, max_col)``.
    """
    height, width = rgb.shape[:2]
    top = height - 87
    left = width - 28 - bar_px
    rgb[top : top + 8, left : left + bar_px] = colour
    glyph_top = top + 27
    x = left - 6
    # Four glyph-like blocks with holes, so they are not mistaken for a bar.
    for glyph_width in (21, 24, 24, 39):
        rgb[glyph_top : glyph_top + 34, x : x + glyph_width] = colour
        rgb[glyph_top + 6 : glyph_top + 28, x + 5 : x + glyph_width - 5] = 0
        x += glyph_width + 6
    return (top, left, top + 8, left + bar_px)


def synthetic_field(
    height: int = 800,
    width: int = 800,
    n_cells: int = 40,
    seed: int = 0,
    ruler: bool = False,
    touching_fraction: float = 0.3,
) -> np.ndarray:
    """An RGB field of elongated red cells with blue nuclei.

    A fraction of cells is placed against a neighbour so the splitter and the
    contact graph have real work to do.
    """
    rng = np.random.default_rng(seed)
    red = np.zeros((height, width), np.float32)
    blue = np.zeros((height, width), np.float32)
    scale = max(0.35, min(1.0, np.sqrt(height * width / n_cells) / 110))
    placed: list[tuple[float, float, float]] = []
    for _ in range(n_cells):
        major = rng.uniform(30, 45) * scale
        minor = major / rng.uniform(1.8, 2.8)
        angle = rng.uniform(-np.pi, np.pi)
        if placed and rng.random() < touching_fraction:
            py, px, _ = placed[int(rng.integers(len(placed)))]
            cy = py + rng.uniform(-1, 1) * minor * 1.6
            cx = px + rng.uniform(-1, 1) * major * 1.6
        else:
            cy = rng.uniform(0, height)
            cx = rng.uniform(0, width)
        placed.append((cy, cx, major))
        rr, cc = draw.ellipse(cy, cx, minor, major, shape=(height, width), rotation=angle)
        red[rr, cc] = np.maximum(red[rr, cc], rng.uniform(140, 230))
        rr, cc = draw.ellipse(cy, cx, minor * 0.4, minor * 0.5, shape=(height, width))
        blue[rr, cc] = rng.uniform(150, 230)
    noise = rng.normal(0, 6, (height, width)).astype(np.float32)
    rgb = np.zeros((height, width, 3), np.uint8)
    rgb[..., 0] = np.clip(red + noise, 0, 255).astype(np.uint8)
    rgb[..., 2] = np.clip(blue + noise, 0, 255).astype(np.uint8)
    if ruler:
        draw_ruler(rgb)
    return rgb


def dense_labels(size: int = 2048, cell: int = 48, seed: int = 0) -> np.ndarray:
    """A label image packed with ~(size/cell)^2 ellipses, most of them touching.

    Used to stress QC, clustering and measurement without paying for
    segmentation.
    """
    rng = np.random.default_rng(seed)
    labels = np.zeros((size, size), np.int32)
    next_id = 1
    for row in range(cell // 2, size, cell):
        for col in range(cell // 2, size, cell):
            if rng.random() < 0.08:
                continue
            rr, cc = draw.ellipse(
                row + rng.uniform(-3, 3), col + rng.uniform(-3, 3),
                cell * rng.uniform(0.4, 0.55), cell * rng.uniform(0.45, 0.62),
                shape=labels.shape, rotation=rng.uniform(-0.5, 0.5),
            )
            labels[rr, cc] = next_id
            next_id += 1
    return labels


def image_record(rgb: np.ndarray, name: str):
    """Wrap an array as the frozen ImageRecord the app would build on upload."""
    import hashlib

    from .types import ImageRecord

    pixels = np.ascontiguousarray(rgb)
    pixels.flags.writeable = False
    channels = 1 if pixels.ndim == 2 else int(pixels.shape[2])
    return ImageRecord(
        name, hashlib.sha256(pixels.tobytes()).hexdigest(),
        int(pixels.shape[0]), int(pixels.shape[1]), str(pixels.dtype), channels, pixels,
    )


def synthetic_batch(n_images: int, size: int = 800, n_cells: int = 40, ruler_every: int = 0,
                    seed: int = 0):
    """A batch of synthetic RGB fields with red as the cell channel."""
    from .image_io import make_display
    from .types import AnalysisSession, BatchSession

    batch = BatchSession(label="benchmark")
    batch.channel = "red"
    for index in range(n_images):
        ruler = bool(ruler_every) and index % ruler_every == 0
        rgb = synthetic_field(size, size, n_cells, seed=seed + index, ruler=ruler)
        session = AnalysisSession()
        session.image = image_record(rgb, "field_{:03d}.png".format(index + 1))
        session.display_rgb, session.display_scale = make_display(session.image)
        batch.add(session)
        batch.apply_channels(session)
    return batch


# --------------------------------------------------------------------------- #
# Measurement plumbing
# --------------------------------------------------------------------------- #


class PeakMemory:
    """Samples resident memory on a background thread.

    ``psutil`` is optional; without it only the start and end values from
    ``tracemalloc``-free sources are unavailable and ``peak_mb`` is ``None``.
    """

    def __init__(self, interval: float = 0.02):
        self.interval = interval
        self.peak = 0
        self.start = 0
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        try:
            import psutil

            self._process = psutil.Process()
        except Exception:
            self._process = None

    def _rss(self) -> int:
        return int(self._process.memory_info().rss) if self._process else 0

    def _run(self) -> None:
        while not self._stop.is_set():
            self.peak = max(self.peak, self._rss())
            self._stop.wait(self.interval)

    def __enter__(self):
        self.start = self.peak = self._rss()
        if self._process is not None:
            self._thread = threading.Thread(target=self._run, daemon=True)
            self._thread.start()
        return self

    def __exit__(self, *exc) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join()
        self.peak = max(self.peak, self._rss())

    @property
    def peak_mb(self) -> float | None:
        return round(self.peak / 2**20, 1) if self._process else None

    @property
    def growth_mb(self) -> float | None:
        return round((self.peak - self.start) / 2**20, 1) if self._process else None


def _cuda():
    try:
        import torch

        return torch if torch.cuda.is_available() else None
    except Exception:
        return None


@contextmanager
def peak_vram() -> Iterator[dict[str, Any]]:
    """Peak CUDA memory for the enclosed block, or empty when there is no GPU."""
    torch = _cuda()
    stats: dict[str, Any] = {}
    if torch is not None:
        torch.cuda.reset_peak_memory_stats()
    try:
        yield stats
    finally:
        if torch is not None:
            stats["vram_peak_allocated_mb"] = round(torch.cuda.max_memory_allocated() / 2**20, 1)
            stats["vram_peak_reserved_mb"] = round(torch.cuda.max_memory_reserved() / 2**20, 1)


@dataclass
class Timings:
    """Named wall-clock samples, in seconds."""

    samples: dict[str, list[float]] = field(default_factory=dict)

    @contextmanager
    def time(self, name: str) -> Iterator[None]:
        started = time.perf_counter()
        try:
            yield
        finally:
            self.samples.setdefault(name, []).append(time.perf_counter() - started)

    def summary(self) -> dict[str, dict[str, float]]:
        return {
            name: {
                "n": len(values),
                "median_s": round(statistics.median(values), 5),
                "total_s": round(sum(values), 5),
                "max_s": round(max(values), 5),
            }
            for name, values in self.samples.items()
        }


def repeat(fn: Callable[[], Any], runs: int = 5, warmup: int = 1) -> dict[str, float]:
    """Median and spread of ``fn`` over several runs after a warm-up."""
    for _ in range(warmup):
        fn()
    values = []
    for _ in range(runs):
        started = time.perf_counter()
        fn()
        values.append(time.perf_counter() - started)
    return {
        "median_ms": round(statistics.median(values) * 1000, 2),
        "min_ms": round(min(values) * 1000, 2),
        "max_ms": round(max(values) * 1000, 2),
    }


def environment() -> dict[str, Any]:
    """What the numbers were measured on."""
    from .segmentation import cellpose_version

    info: dict[str, Any] = {
        "cellscope_version": __version__,
        "python": platform.python_version(),
        "platform": platform.platform(),
        "processor": platform.processor(),
        "cpu_count": os.cpu_count(),
        "numpy": np.__version__,
        "cellpose": cellpose_version(),
    }
    torch = _cuda()
    if torch is not None:
        info["gpu"] = torch.cuda.get_device_name(0)
        info["gpu_total_mb"] = round(torch.cuda.get_device_properties(0).total_memory / 2**20)
        info["torch"] = torch.__version__
    return info


# --------------------------------------------------------------------------- #
# Workloads
# --------------------------------------------------------------------------- #


def bench_pipeline(batch, engine: str = "threshold_watershed") -> dict[str, Any]:
    """Segment and measure every image, timing each stage separately."""
    from .clustering import build_clusters
    from .image_io import extract_channel
    from .morphometry import measure_cells, measure_clusters
    from .pipeline import compute_results, iter_batch_segmentation, nuclear_array_for
    from .preprocessing import preprocess_image
    from .qc import apply_qc
    from .segmentation import segment_cells, split_touching_cells
    from .visualize import make_overlay

    timings = Timings()
    per_image: list[float] = []
    with PeakMemory() as memory, peak_vram() as vram:
        # Stage breakdown on the first image, then the real batch loop.
        first = batch.images[0]
        with timings.time("extract_channel"):
            channel = extract_channel(first.image, first.channel)
        with timings.time("preprocess"):
            prepared = preprocess_image(channel, first.preprocess_params)
        nuclear = nuclear_array_for(first)
        prepared_nuclear = (
            preprocess_image(nuclear, first.preprocess_params) if nuclear is not None else None
        )
        with timings.time("segment_first_image"):
            raw, _ = segment_cells(prepared, first.segmentation_params, engine=engine,
                                   nuclear=prepared_nuclear)
        with timings.time("split"):
            split_touching_cells(raw, first.split_params, first.segmentation_params.min_object_area)

        started = time.perf_counter()
        with timings.time("batch_segmentation_total"):
            for _ in iter_batch_segmentation(batch, engine=engine):
                now = time.perf_counter()
                per_image.append(now - started)
                started = now

        for session in batch.images:
            if not session.has_segmentation:
                continue
            with timings.time("apply_qc"):
                qc_labels = apply_qc(session.object_labels, session.qc)
            included = session.included_objects()
            with timings.time("build_clusters"):
                clusters, _ = build_clusters(
                    qc_labels, included, session.cluster_params.contact_distance_px
                )
            with timings.time("measure_cells"):
                cells = measure_cells(qc_labels, included, clusters, session.calibration)
            with timings.time("measure_clusters"):
                measure_clusters(qc_labels, included, clusters, session.calibration, cells,
                                 session.cluster_params.contact_distance_px)
            with timings.time("compute_results_cold"):
                results = compute_results(session, use_cache=False)
            with timings.time("compute_results_cached"):
                compute_results(session)
            with timings.time("overlay"):
                make_overlay(session.display_rgb, results.qc_labels, results.objects,
                             results.clusters, excluded_ids=session.qc.excluded_ids)

    objects = sum(len(s.objects) for s in batch.images)
    return {
        "images": len(batch.images),
        "objects": objects,
        "engine": engine,
        "stages": timings.summary(),
        "per_image_s": [round(v, 4) for v in per_image],
        "images_per_minute": round(60 * len(per_image) / max(sum(per_image), 1e-9), 2),
        "peak_rss_mb": memory.peak_mb,
        "rss_growth_mb": memory.growth_mb,
        **vram,
    }


def bench_dense(size: int = 2048, cell: int = 48) -> dict[str, Any]:
    """QC, clustering and measurement on a crowded label image."""
    from .pipeline import compute_results
    from .segmentation import split_touching_cells
    from .types import AnalysisSession

    labels = dense_labels(size, cell)
    timings = Timings()
    with PeakMemory() as memory:
        with timings.time("split"):
            object_labels, objects = split_touching_cells(labels)
        session = AnalysisSession()
        session.object_labels = object_labels
        session.objects = objects
        for _ in range(2):
            with timings.time("compute_results_cold"):
                results = compute_results(session, use_cache=False)
    return {
        "size": size,
        "objects": len(objects),
        "clusters": len(results.clusters),
        "stages": timings.summary(),
        "peak_rss_mb": memory.peak_mb,
    }


def bench_persistence(batch) -> dict[str, Any]:
    """Project save/load, batch export and the PDF report."""
    from .export import export_batch
    from .project import load_project, save_project
    from .report import write_report

    timings = Timings()
    sizes: dict[str, int] = {}
    with tempfile.TemporaryDirectory(prefix="cellscope_bench_") as folder:
        path = os.path.join(folder, "bench.cellscope")
        for level, name in ((1, "save_project_autosave"), (6, "save_project_manual")):
            with timings.time(name):
                save_project(batch, path, compression_level=level)
            sizes[name + "_bytes"] = os.path.getsize(path)
        with timings.time("load_project"):
            load_project(path)
        with timings.time("write_report"):
            write_report(batch, os.path.join(folder, "report.pdf"))
        with timings.time("export_batch"):
            archive = export_batch(batch, folder)
        sizes["export_bytes"] = os.path.getsize(archive)
    return {"stages": timings.summary(), **sizes}
