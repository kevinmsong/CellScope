"""Design tab: readiness, experimental design, and replicate-level comparison."""

from __future__ import annotations

import tempfile
from pathlib import Path
from types import SimpleNamespace

import gradio as gr
import pandas as pd

from src.quality import AGGREGATION_RULE, DESIGN_METRICS, design_summary, readiness
from src.report import write_report
from src.visualize import replicate_dot_plot

from .results_tab import round_frame
from .theme import callout, section
from .views import locked, status_html

DESIGN_COLUMNS = ["Image", "Well", "Condition", "Biological replicate"]
METRIC_CHOICES = [(label, key) for key, (label, _, _) in DESIGN_METRICS.items()]


def design_table(batch) -> pd.DataFrame:
    if batch is None:
        return pd.DataFrame(columns=DESIGN_COLUMNS)
    return pd.DataFrame([[s.display_name, s.well, s.condition, s.replicate] for s in batch.images],
                        columns=DESIGN_COLUMNS)


def _status_note(summary, batch) -> str:
    notes = []
    unassigned = summary["unassigned"]
    if len(unassigned):
        notes.append("{} image(s) are not in the comparison: {}.".format(
            len(unassigned), ", ".join("{} ({})".format(r.image, r.reason)
                                       for r in unassigned.head(6).itertuples())))
    conditions = summary["conditions"]
    if len(conditions):
        few = conditions[conditions["biological_replicates"] < 3]
        if len(few):
            notes.append("Fewer than three biological replicates for: {}. Treat those means "
                         "with caution.".format(", ".join(sorted(set(few["condition"])))))
        if not conditions["all_reviewed"].all():
            notes.append("Some images in the comparison are not reviewed yet.")
        if len(set(conditions["unit"])) > 1:
            notes.append("Calibrated and uncalibrated images are shown in separate rows; "
                         "they are never combined.")
    tone = "warn" if notes else "info"
    body = AGGREGATION_RULE + ("<br><br>" + "<br>".join(notes) if notes else "")
    return callout(body, tone=tone)


def on_quality(metric, batch):
    if batch is None:
        batch_frame = pd.DataFrame(columns=["Image", "Status", "Detail"])
        return (batch_frame, design_table(batch), None, None, None, None, "", status_html(batch))
    summary = design_summary(batch, metric or "area")
    label = DESIGN_METRICS[metric or "area"][0]
    plot = replicate_dot_plot(summary["replicates"], summary["conditions"], label)
    return (
        readiness(batch),
        design_table(batch),
        round_frame(summary["conditions"]),
        round_frame(summary["replicates"]),
        round_frame(summary["wells"]),
        plot,
        _status_note(summary, batch),
        status_html(batch),
    )


@locked
def on_save_design(table, metric, batch):
    frame = pd.DataFrame(table)
    if batch is None or len(frame) != len(batch.images) or \
            list(frame.iloc[:, 0]) != [s.display_name for s in batch.images]:
        return (*on_quality(metric, batch), "The image list changed; refresh before editing the design.")
    for session, row in zip(batch.images, frame.fillna("").itertuples(index=False, name=None)):
        session.well, session.condition, session.replicate = [str(v).strip() for v in row[1:4]]
    return (*on_quality(metric, batch), "Experimental design saved.")


def on_quality_view(metric, batch):
    return (*on_quality(metric, batch), gr.update())


def on_pdf(batch):
    try:
        if not batch or batch.is_empty:
            raise ValueError("Add images first.")
        path = Path(tempfile.mkdtemp(prefix="cellscope_report_")) / "CellScope_report.pdf"
        return write_report(batch, path), "Report generated from the current QC-approved measurements."
    except Exception as exc:
        return None, "Report failed: " + str(exc)


def build(batch, shared) -> SimpleNamespace:
    c = SimpleNamespace()
    with gr.Tab("Design & compare", id="tab_design") as c.tab:
        gr.HTML(section("Experimental design"))
        gr.Markdown(
            "Give each image its **well**, **condition** and **biological replicate**. Reuse a "
            "replicate label for technical wells of the same biological sample. Images missing "
            "any label are left out of the comparison, never guessed.", elem_classes="cs-note")
        c.design = gr.Dataframe(headers=DESIGN_COLUMNS, interactive=True, wrap=True)
        with gr.Row():
            c.save_design = gr.Button("Save design", variant="primary")
            c.design_note = gr.Markdown(elem_classes="cs-note")
        gr.HTML(section("Comparison"))
        with gr.Row():
            c.metric = gr.Dropdown(METRIC_CHOICES, value="area", label="Measurement", scale=1)
            c.refresh = gr.Button("Refresh", size="sm", scale=0)
        c.rule = gr.HTML()
        with gr.Row(equal_height=False):
            with gr.Column(scale=3):
                c.conditions = gr.Dataframe(interactive=False, label="Conditions (n = biological replicates)",
                                            wrap=True)
                c.replicates = gr.Dataframe(interactive=False, label="Biological replicates", wrap=True)
            with gr.Column(scale=2):
                c.plot = gr.Plot(label="Replicate means")
        with gr.Accordion("Wells", open=False):
            c.wells = gr.Dataframe(interactive=False, wrap=True)
        gr.HTML(section("Readiness"))
        c.checks = gr.Dataframe(interactive=False, label="Outstanding checks", wrap=True)
        gr.HTML(section("Report"))
        with gr.Row():
            c.pdf = gr.Button("Build PDF report")
            c.pdf_file = gr.File(label="Report")
        c.pdf_note = gr.Markdown(elem_classes="cs-note")
        gr.HTML(callout("Statistics are descriptive. CellScope runs no significance tests; "
                        "choose one that matches your design, with biological replicates as n."))
    return c


def wire(c, batch, shared) -> None:
    outputs = [c.checks, c.design, c.conditions, c.replicates, c.wells, c.plot, c.rule, shared.status]
    c.tab.select(on_quality_view, [c.metric, batch], outputs + [c.design_note], show_progress="hidden")
    c.refresh.click(on_quality_view, [c.metric, batch], outputs + [c.design_note])
    c.metric.input(on_quality_view, [c.metric, batch], outputs + [c.design_note], show_progress="hidden")
    c.save_design.click(on_save_design, [c.design, c.metric, batch], outputs + [c.design_note])
    c.pdf.click(on_pdf, [batch], [c.pdf_file, c.pdf_note], show_progress_on=[c.pdf_file])
