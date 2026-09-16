"""Export tab: the self-documenting analysis archive."""

from __future__ import annotations

import os
import tempfile
import traceback
from types import SimpleNamespace

import gradio as gr

from src.batch import CalibrationMismatch, pooled_cells
from src.export import export_batch
from src.pipeline import compute_results
from src.visualize import make_overlay
from src.workflow import change_token, is_stale

from .views import display_base, error_text, status_html

OUTPUT_DIR = os.path.join(tempfile.gettempdir(), "cellscope_exports")

ARCHIVE_GUIDE = """
| File | Contents |
|---|---|
| `manifest.json` | How every number was produced: versions, device, parameters, calibration, QC, corrections, review, design, and a checksum for each file |
| `batch_summary.csv` | All accepted cells pooled (descriptive) |
| `design_*.csv` | Image → well → biological replicate → condition summaries |
| `per_image_summary.csv` | One row per image |
| `all_cells.csv`, `all_clusters.csv` | Every cell and cluster, tagged with image and well |
| `qc_log.csv` | Every exclusion and restoration, with its reason |
| `report.pdf` | A readable report |
| `images/<name>/` | Masks (raw and QC-approved), overlay, measurements and metadata per image |
"""


def on_export(batch):
    if batch is None or batch.is_empty:
        return None, "Add images first.", status_html(batch)
    if not batch.segmented:
        return None, "Segment at least one image before exporting.", status_html(batch)

    overlays = {}
    for session in batch.images:
        if session.has_segmentation:
            results = compute_results(session)
            overlays[session.analysis_id] = make_overlay(
                display_base(session), results.qc_labels, results.objects, results.clusters,
                excluded_ids=session.qc.excluded_ids, annotation=session.annotation,
            )
    try:
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        with batch.lock():
            token = change_token(batch)
        path = export_batch(batch, OUTPUT_DIR, overlays=overlays)
    except CalibrationMismatch as exc:
        return None, error_text(exc), status_html(batch)
    except Exception as exc:
        traceback.print_exc()
        return None, error_text(exc), status_html(batch)
    batch.exported_token = token

    warnings = []
    if batch.n_unreviewed:
        warnings.append("**{} image(s) were not reviewed**; this is recorded in the manifest.".format(
            batch.n_unreviewed))
    stale = [s.display_name for s in batch.images if is_stale(s)]
    if stale:
        warnings.append("**{} image(s) were segmented with settings that have since changed**; "
                        "the manifest records the settings actually used.".format(len(stale)))
    return path, "Wrote **{}** ({:.0f} KB): {} image(s), {} pooled cell(s).{}".format(
        os.path.basename(path), os.path.getsize(path) / 1024, len(batch.segmented),
        len(pooled_cells(batch)), "".join("  \n" + w for w in warnings),
    ), status_html(batch)


def build(batch, shared) -> SimpleNamespace:
    c = SimpleNamespace()
    with gr.Tab("Export", id="tab_export") as c.tab:
        gr.Markdown("Everything needed to reproduce and audit the batch, in one archive.",
                    elem_classes="cs-note")
        c.button = gr.Button("Build analysis archive", variant="primary", size="lg")
        c.file = gr.File(label="Download")
        c.note = gr.Markdown(elem_classes="cs-note")
        with gr.Accordion("What is in the archive", open=False):
            gr.Markdown(ARCHIVE_GUIDE)
    return c


def wire(c, batch, shared) -> None:
    c.button.click(on_export, [batch], [c.file, c.note, shared.status], show_progress_on=[c.file])
