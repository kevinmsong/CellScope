"""Command-line launcher: ``cellscope`` or ``python -m cellscope``."""

from __future__ import annotations

import argparse
import os
import sys
import threading
import time


def preload_model(use_gpu: bool) -> None:
    """Load the Cellpose weights in the background while the user uploads.

    Failures are printed, not raised: this is an optimisation, and a real run
    reports any genuine problem.
    """
    def warm():
        try:
            from src.segmentation import _get_cellpose_model, _resolve_device, cellpose_available

            if cellpose_available():
                started = time.perf_counter()
                _get_cellpose_model("cpsam", _resolve_device(use_gpu))
                print("Cellpose weights ready ({:.1f} s)".format(time.perf_counter() - started))
        except Exception as error:
            print("Model preload skipped: {}: {}".format(type(error).__name__, error))

    threading.Thread(target=warm, daemon=True, name="cellscope-preload").start()


def environment_summary() -> str:
    from src import __version__
    from src.segmentation import cellpose_version

    lines = ["CellScope {}".format(__version__),
             "Python {}".format(sys.version.split()[0]),
             "Cellpose {}".format(cellpose_version() or "not installed")]
    try:
        import torch

        if torch.cuda.is_available():
            lines.append("GPU {} (torch {}, CUDA {})".format(
                torch.cuda.get_device_name(0), torch.__version__, torch.version.cuda))
        else:
            lines.append("GPU not available (torch {}); inference will use the CPU".format(
                torch.__version__))
    except ImportError:
        lines.append("PyTorch not installed")
    return " | ".join(lines)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        prog="cellscope",
        description="Fluorescence cell morphometry with explicit single-cell vs cluster resolution.",
    )
    parser.add_argument("--host", default="127.0.0.1",
                        help="address to serve on (default: this computer only)")
    parser.add_argument("--port", type=int, default=int(os.environ.get("CELLSCOPE_PORT", "7860")))
    parser.add_argument("--projects", help="folder for autosaved projects (default: ./projects)")
    parser.add_argument("--autosave-interval", type=float, default=30.0, metavar="SECONDS",
                        help="how often to check for unsaved changes")
    parser.add_argument("--cpu", action="store_true", help="never use the GPU")
    parser.add_argument("--no-preload", action="store_true",
                        help="do not load the Cellpose model at start-up")
    parser.add_argument("--inbrowser", action="store_true", help="open a browser tab")
    parser.add_argument("--share", action="store_true",
                        help="create a public Gradio link (off by default; CellScope has no "
                             "authentication, so only use this on trusted networks)")
    parser.add_argument("--version", action="store_true", help="print versions and exit")
    args = parser.parse_args(argv)

    if args.version:
        print(environment_summary())
        return 0
    if args.projects:
        os.environ["CELLSCOPE_PROJECTS"] = os.path.abspath(args.projects)
    if args.cpu:
        os.environ["CELLSCOPE_FORCE_CPU"] = "1"

    print(environment_summary())
    from cellscope.ui import build_interface

    if not args.no_preload:
        preload_model(use_gpu=not args.cpu)
    demo = build_interface(autosave_interval=args.autosave_interval)
    demo.queue(default_concurrency_limit=1)
    demo.launch(server_name=args.host, server_port=args.port, share=args.share,
                inbrowser=args.inbrowser, show_api=False)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
