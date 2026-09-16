"""Results tab: pooled, descriptive summaries and distributions for the batch.

Everything here is memoised on the batch key, so returning to the tab after
reviewing costs nothing unless a result actually changed.
"""

from __future__ import annotations

from types import SimpleNamespace

import gradio as gr
import pandas as pd

from src.batch import (
    CalibrationMismatch,
    batch_summary,
    calibration_consistency,
    memoised,
    per_image_summary,
    pooled_cells,
    pooled_clusters,
)
from src.calibration import uncalibrated
from src.visualize import build_all_figures

from .theme import callout, empty_state
from .views import DASH, empty_frame, status_html

PSEUDOREPLICATION_NOTE = (
    "<strong>Descriptive, not inferential.</strong> These figures pool every accepted "
    "cell across every image in this batch, weighted by cell count. Cells are "
    "<em>not</em> independent biological replicates: compare conditions on the "
    "Design tab, where the well and the biological replicate are the units. CellScope "
    "deliberately computes no p-values."
)

HEADLINE = (
    ("n_cells", "Cells pooled"),
    ("n_images_segmented", "Images"),
    ("isolated_cells", "Isolated"),
    ("multicell_clusters", "Clusters"),
    ("unresolved_clusters", "Unresolved groups"),
)
FIGURE_KEYS = (
    "area_distribution", "major_axis_distribution", "area_vs_major",
    "length_vs_width", "cluster_size_distribution",
)


def headline_html(row, calibrated) -> str:
    cells = []
    for key, label in HEADLINE:
        cells.append('<div class="cs-stat"><span class="cs-k">{}</span>'
                     '<span class="cs-v">{}</span></div>'.format(label, row.get(key, DASH)))
    area_key = "mean_area_um2" if calibrated else "mean_area_px2"
    unit = "µm²" if calibrated else "px²"
    area = row.get(area_key)
    text = "{:.0f} {}".format(area, unit) if area is not None and not pd.isna(area) else DASH
    cells.append('<div class="cs-stat"><span class="cs-k">Mean cell area</span>'
                 '<span class="cs-v">{}</span></div>'.format(text))
    return '<div class="cs-status">{}</div>'.format("".join(cells))


def round_frame(frame, digits=3):
    if frame is None or not len(frame):
        return frame
    out = frame.copy()
    for column in out.select_dtypes("number").columns:
        out[column] = out[column].round(digits)
    return out


_EMPTY_FIGURES = None


def _empty_figures():
    global _EMPTY_FIGURES
    if _EMPTY_FIGURES is None:
        blank = empty_frame("cell_id")
        _EMPTY_FIGURES = build_all_figures(blank, blank, uncalibrated())
    return _EMPTY_FIGURES


def on_refresh_results(batch):
    blank = empty_frame("cell_id")
    figures = _empty_figures()

    def early(note, message=""):
        return (batch, "", note, blank, blank, blank,
                *(figures[k] for k in FIGURE_KEYS), status_html(batch))

    if batch is None or batch.is_empty:
        return early(empty_state("No results yet", "Add images on the Import tab, then segment them."))
    if not batch.segmented:
        return early(empty_state(
            "Nothing segmented yet",
            "{} image(s) are loaded. Segment them to see results.".format(len(batch.images))))
    verdict = calibration_consistency(batch)
    if not verdict["consistent"]:
        return early(callout(verdict["reason"], tone="error"))
    try:
        summary = memoised(batch, "results_summary", lambda: batch_summary(batch))
    except CalibrationMismatch as exc:
        return early(callout(str(exc), tone="error"))

    cells = pooled_cells(batch)
    figures = memoised(batch, "results_figures",
                       lambda: build_all_figures(cells, pooled_clusters(batch), batch.calibration))
    tables = memoised(batch, "results_tables", lambda: (
        round_frame(summary.T.reset_index().rename(columns={"index": "metric", 0: "value"})),
        round_frame(per_image_summary(batch)),
        round_frame(cells),
    ))

    banner = PSEUDOREPLICATION_NOTE
    tone = "info"
    if batch.n_unreviewed:
        banner = ("<strong>{} of {} image(s) not reviewed yet:</strong> {}. They are already "
                  "included below.<br><br>".format(
                      batch.n_unreviewed, len(batch.images),
                      ", ".join(batch.unreviewed_names[:5]) + ("…" if batch.n_unreviewed > 5 else ""))
                  + PSEUDOREPLICATION_NOTE)
        tone = "warn"
    return (
        batch,
        headline_html(summary.iloc[0], verdict["calibrated"]),
        callout(banner, tone=tone),
        *tables,
        *(figures[k] for k in FIGURE_KEYS),
        status_html(batch),
    )


def build(batch, shared) -> SimpleNamespace:
    c = SimpleNamespace()
    with gr.Tab("Results", id="tab_results", elem_id="cs-results") as c.tab:
        c.headline = gr.HTML()
        with gr.Row():
            c.refresh = gr.Button("Refresh", size="sm", scale=0)
            c.note = gr.HTML()
        with gr.Tabs():
            with gr.Tab("Batch summary"):
                gr.Markdown("Pooled across every accepted cell in the batch.", elem_classes="cs-note")
                c.summary = gr.Dataframe(interactive=False, wrap=True)
            with gr.Tab("Per image"):
                gr.Markdown("One row per image, for spotting a field unlike its neighbours.",
                            elem_classes="cs-note")
                c.per_image = gr.Dataframe(interactive=False, wrap=True)
            with gr.Tab("All cells"):
                gr.Markdown("Every accepted, resolved cell. Unresolved groups are absent by design.",
                            elem_classes="cs-note")
                c.cells = gr.Dataframe(interactive=False, wrap=True, max_height=500)
            with gr.Tab("Distributions"):
                with gr.Row():
                    c.fig_area = gr.Plot(label="Cell area")
                    c.fig_major = gr.Plot(label="Major axis")
                with gr.Row():
                    c.fig_area_major = gr.Plot(label="Area vs major axis")
                    c.fig_length_width = gr.Plot(label="Length vs width")
                c.fig_cluster = gr.Plot(label="Cluster size")
    return c


def wire(c, batch, shared) -> None:
    outputs = [batch, c.headline, c.note, c.summary, c.per_image, c.cells, c.fig_area,
               c.fig_major, c.fig_area_major, c.fig_length_width, c.fig_cluster, shared.status]
    shared.results_outputs = outputs
    c.refresh.click(on_refresh_results, [batch], outputs, show_progress_on=[c.summary])
    c.tab.select(on_refresh_results, [batch], outputs, show_progress="hidden")
