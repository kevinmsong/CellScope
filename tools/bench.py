"""CellScope benchmark runner.

    python tools/bench.py                       # quick CPU suite
    python tools/bench.py --suite full          # every workload
    python tools/bench.py --engine cellpose     # GPU inference
    python tools/bench.py --images images/      # local lab images (never committed)

Results are written as JSON to ``output/performance/`` (gitignored) and printed
as a Markdown table for ``docs/PERFORMANCE.md``.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src import perf

# --------------------------------------------------------------------------- #
# UI callback latency
#
# Each entry reproduces what one user gesture costs on the server: the handler
# plus every chained follow-up that Gradio runs for it. Keeping the table in one
# place means only this block changes when callbacks move.
# --------------------------------------------------------------------------- #


def _legacy_gestures(click_event):
    """Gesture costs for the pre-refactor ``app.py`` (the recorded baseline)."""
    import app
    import professional_ui
    from src.review import history

    def review(batch):
        views = app.on_toggle_labels(True, False, batch)
        professional_ui.review_table(batch)
        return views

    def opacity(batch):
        batch.active.review_alpha = 0.3 if batch.active.review_alpha != 0.3 else 0.4
        return review(batch)

    def zoom(batch):
        batch.active.review_zoom = 2.0 if batch.active.review_zoom != 2.0 else 3.0
        return review(batch)

    def inspect_click(batch):
        batch.active.review_mode = "Inspect"
        app.on_review_click(True, False, batch, click_event(batch))
        professional_ui.review_table(batch)

    def toggle_click(batch):
        batch.active.review_mode = "Toggle inclusion"
        app.on_review_click(True, False, batch, click_event(batch))
        professional_ui.review_table(batch)

    def undo(batch):
        history(batch.active)
        return review(batch)

    def next_image(batch):
        app.on_next(batch)
        return review(batch)

    return {
        "next_image": next_image,
        "opacity_change": opacity,
        "zoom_change": zoom,
        "inspect_click": inspect_click,
        "exclude_click": toggle_click,
        "undo": undo,
        "results_tab": app.on_refresh_results,
        "status_strip": app.status_html,
    }


def _ui_gestures():
    import gradio as gr

    try:
        from cellscope.ui.gestures import gestures
    except ImportError:
        gestures = _legacy_gestures

    def click_event(batch):
        session = batch.active
        obj = next(o for o in session.objects if session.qc.is_included(o.object_id))
        import numpy as np

        rows, cols = np.nonzero(session.object_labels == obj.object_id)
        scale = session.display_scale
        point = [int(cols[len(cols) // 2] / scale), int(rows[len(rows) // 2] / scale)]
        return gr.SelectData(None, {"index": point, "value": None})

    return gestures(click_event)


def bench_ui(batch, runs: int) -> dict:
    results = {}
    for name, action in _ui_gestures().items():
        results[name] = perf.repeat(lambda action=action: action(batch), runs=runs)
    return results


# --------------------------------------------------------------------------- #


def _git_sha() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, text=True
        ).strip()
    except Exception:
        return "unknown"


def _local_batch(folder: str, limit: int):
    from src.image_io import load_image, make_display
    from src.types import AnalysisSession, BatchSession

    paths = sorted(glob.glob(os.path.join(folder, "**", "*.jpg"), recursive=True))
    paths = [p for p in paths if "ruler" not in os.path.basename(p).lower()][:limit]
    batch = BatchSession(label="local images")
    batch.channel = "red"
    batch.nuclear_channel = "blue"
    for path in paths:
        session = AnalysisSession()
        session.image = load_image(path)
        session.display_rgb, session.display_scale = make_display(session.image)
        batch.add(session)
        batch.apply_channels(session)
    return batch


def run(args) -> dict:
    report: dict = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "git": _git_sha(),
        "environment": perf.environment(),
        "engine": args.engine,
        "workloads": {},
    }
    workloads = report["workloads"]

    def section(name, fn):
        print("running", name, "...", flush=True)
        started = time.perf_counter()
        workloads[name] = fn()
        workloads[name]["wall_s"] = round(time.perf_counter() - started, 2)

    if args.engine == "cellpose":
        from src.segmentation import _get_cellpose_model

        started = time.perf_counter()
        _get_cellpose_model("cpsam", True)
        report["model_load_s"] = round(time.perf_counter() - started, 2)

    if args.images:
        batch = _local_batch(args.images, args.limit)
        section("local", lambda: perf.bench_pipeline(batch, args.engine))
        section("local_ui", lambda: bench_ui(batch, args.runs))
        section("local_persistence", lambda: perf.bench_persistence(batch))
        return report

    sizes = [1, 10] if args.suite == "quick" else [1, 10, 50]
    ui_batch = None
    for n in sizes:
        batch = perf.synthetic_batch(n, ruler_every=3)
        section("batch_{}".format(n), lambda b=batch: perf.bench_pipeline(b, args.engine))
        if n == 10:
            ui_batch = batch
            section("persistence_10", lambda b=batch: perf.bench_persistence(b))
    if ui_batch is not None:
        section("ui_10", lambda: bench_ui(ui_batch, args.runs))
    if args.suite == "full":
        large = perf.synthetic_batch(1, size=4096, n_cells=400)
        section("large_4096", lambda: perf.bench_pipeline(large, args.engine))
        section("dense_2048", lambda: perf.bench_dense())
    return report


def markdown(report: dict) -> str:
    lines = ["| Workload | Measure | Value |", "|---|---|---|"]
    for name, data in report["workloads"].items():
        if name.startswith("ui") or name.endswith("_ui"):
            for gesture, stats in data.items():
                if isinstance(stats, dict):
                    lines.append("| {} | {} | {} ms |".format(name, gesture, stats["median_ms"]))
            continue
        for stage, stats in data.get("stages", {}).items():
            lines.append("| {} | {} | {} s (median of {}) |".format(
                name, stage, stats["median_s"], stats["n"]))
        for key in ("images_per_minute", "peak_rss_mb", "rss_growth_mb",
                    "vram_peak_reserved_mb", "objects", "wall_s"):
            if data.get(key) is not None:
                lines.append("| {} | {} | {} |".format(name, key, data[key]))
    return "\n".join(lines)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("--suite", choices=["quick", "full"], default="quick")
    parser.add_argument("--engine", default="threshold_watershed",
                        choices=["threshold_watershed", "cellpose", "auto"])
    parser.add_argument("--images", help="folder of local images (never committed)")
    parser.add_argument("--limit", type=int, default=10, help="max local images")
    parser.add_argument("--runs", type=int, default=5, help="repeats per UI gesture")
    parser.add_argument("--label", default="", help="suffix for the output file")
    parser.add_argument("--out", default=str(ROOT / "output" / "performance"))
    args = parser.parse_args(argv)

    report = run(args)
    os.makedirs(args.out, exist_ok=True)
    name = "{}-{}{}.json".format(
        time.strftime("%Y%m%d-%H%M%S"), report["git"], "-" + args.label if args.label else ""
    )
    path = os.path.join(args.out, name)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)
    print(markdown(report))
    print("\nwrote", path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
