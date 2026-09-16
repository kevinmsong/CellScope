"""Readable PDF reports generated from current, QC-approved results."""
from io import BytesIO
from xml.sax.saxutils import escape

import pandas as pd
from PIL import Image as PILImage
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.platypus import (
    Image,
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

from . import __version__
from .batch import calibration_consistency
from .image_io import make_display
from .pipeline import compute_results
from .qc import exclusion_counts
from .quality import AGGREGATION_RULE, design_summary, readiness
from .types import utc_now
from .visualize import make_overlay
from .workflow import is_stale


def _dot_plot(replicates, conditions, width=468, height=170):
    """Replicate means per condition, drawn directly with ReportLab graphics."""
    from reportlab.graphics.shapes import Circle, Drawing, Line, String

    drawing = Drawing(width, height)
    frame = replicates[replicates["unit"] == replicates["unit"].iloc[0]]
    means = conditions[conditions["unit"] == frame["unit"].iloc[0]]
    order = sorted(set(frame["condition"]))
    values = list(frame["mean"].dropna())
    if not values:
        return drawing
    low, high = min(values), max(values)
    span = (high - low) or abs(high) or 1.0
    low, high = low - 0.1 * span, high + 0.1 * span
    left, bottom, top = 50, 30, height - 20
    step = (width - left - 10) / max(len(order), 1)

    def y(value):
        return bottom + (value - low) / (high - low) * (top - bottom)

    drawing.add(Line(left, bottom, left, top, strokeColor=colors.grey))
    for fraction in (0.0, 0.5, 1.0):
        value = low + fraction * (high - low)
        drawing.add(String(4, y(value) - 3, "{:.3g}".format(value), fontSize=7))
    for index, condition in enumerate(order):
        centre = left + step * (index + 0.5)
        subset = frame[frame["condition"] == condition]["mean"].dropna()
        for k, value in enumerate(subset):
            offset = (k - (len(subset) - 1) / 2) * 6
            drawing.add(Circle(centre + offset, y(value), 3,
                               fillColor=colors.HexColor("#1f5fb0"), strokeColor=colors.white))
        mean = means[means["condition"] == condition]["mean"]
        if len(mean) and pd.notna(mean.iloc[0]):
            drawing.add(Line(centre - 18, y(mean.iloc[0]), centre + 18, y(mean.iloc[0]),
                             strokeColor=colors.black, strokeWidth=2))
        drawing.add(String(centre, 12, "{} (n={})".format(condition, len(subset)),
                           fontSize=7, textAnchor="middle"))
    unit = frame["unit"].iloc[0]
    drawing.add(String(left, height - 10, "Replicate mean cell area ({}); bar = condition mean".format(unit),
                       fontSize=8))
    return drawing


def write_report(batch, path):
    styles = getSampleStyleSheet()
    story = []
    def paragraph(text, style="BodyText"):
        story.append(Paragraph(escape(str(text)), styles[style]))
        story.append(Spacer(1, 7))
    def table(frame):
        if frame.empty:
            paragraph("No data available.")
            return
        labels = {"biological_replicates": "Biological replicates", "sd_between_replicates": "Between-replicate SD",
                  "nuclei_included": "Included nuclei", "touching_nuclei": "Touching nuclei",
                  "touching_groups": "Touching groups", "nuclei_excluded": "Excluded nuclei"}
        data = [[Paragraph(escape(labels.get(str(c), str(c).replace("_", " ").capitalize())), styles["BodyText"]) for c in frame.columns]]
        data += [[Paragraph(escape("NA" if pd.isna(v) else str(v)), styles["BodyText"]) for v in row] for row in frame.itertuples(index=False, name=None)]
        t = Table(data, repeatRows=1, hAlign="LEFT", colWidths=[468 / len(frame.columns)] * len(frame.columns))
        t.setStyle(TableStyle([("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#E9F5ED")),
                               ("VALIGN", (0, 0), (-1, -1), "TOP"),
                               ("LINEBELOW", (0, 0), (-1, -1), .3, colors.lightgrey),
                               ("BOTTOMPADDING", (0, 0), (-1, -1), 7)]))
        story.append(t)
        story.append(Spacer(1, 12))
    paragraph("CellScope analysis report", "Title")
    paragraph(batch.label or "Untitled experiment", "Heading2")
    paragraph(f"Software {__version__} | Batch {batch.batch_id} | Generated {utc_now()}")

    paragraph("Status", "Heading2")
    verdict = calibration_consistency(batch)
    segmented = [s for s in batch.images if s.has_segmentation]
    stale = [s.display_name for s in segmented if is_stale(s)]
    paragraph(
        f"{len(batch.images)} image(s); {len(segmented)} segmented; "
        f"{batch.n_reviewed} reviewed. Calibration: "
        + (batch.calibration.summary() if verdict["consistent"] else verdict["reason"])
    )
    if stale:
        paragraph("Settings changed after segmentation for: " + ", ".join(stale)
                  + ". Their masks reflect the earlier settings recorded in manifest.json.")
    excluded = {"excluded_border": 0, "excluded_annotation": 0, "excluded_other": 0}
    for s in segmented:
        for key, value in exclusion_counts(s.objects, s.qc).items():
            excluded[key] += value
    paragraph(
        "Excluded objects: {excluded_border} touching the image border, "
        "{excluded_annotation} touching a burned-in scale bar, {excluded_other} by "
        "review or filters.".format(**excluded)
    )

    paragraph("Readiness checks", "Heading2")
    table(readiness(batch))
    paragraph("Experimental comparisons", "Heading2")
    paragraph(AGGREGATION_RULE + " Missing design labels are omitted, not guessed.")
    design = design_summary(batch, "area")
    if len(design["conditions"]):
        conditions = design["conditions"][["condition", "unit", "biological_replicates", "wells",
                                           "cells", "mean", "sd_between_replicates"]]
        table(conditions.round(3))
        story.append(_dot_plot(design["replicates"], design["conditions"]))
        story.append(Spacer(1, 10))
        paragraph("Biological replicates", "Heading3")
        table(design["replicates"][["condition", "biological_replicate", "unit", "wells",
                                    "cells", "mean", "sd_between_wells"]].round(3))
        paragraph("Wells", "Heading3")
        table(design["wells"][["condition", "biological_replicate", "well", "unit", "fields",
                               "cells", "mean", "median"]].round(3))
    else:
        paragraph("No image has a complete well, condition and biological replicate "
                  "assignment yet, so no comparison is shown.")
    if len(design["unassigned"]):
        paragraph("Not in the comparison: " + ", ".join(
            "{} ({})".format(r.image, r.reason) for r in design["unassigned"].itertuples()))
    from .nuclei import batch_nuclear_counts
    paragraph("Independent DAPI nuclei counts", "Heading2")
    nuclei = batch_nuclear_counts(batch)
    if not nuclei.empty:
        table(nuclei[["image", "nuclei_included", "touching_nuclei", "touching_groups", "nuclei_excluded"]])
    else:
        paragraph("DAPI analysis has not been run.")
    paragraph("Touching means direct mask contact, including shared corners. Nuclear counts are separate segmentation estimates and are not cell counts.")
    for s in batch.images:
        story.append(PageBreak())
        paragraph(s.display_name, "Heading1")
        paragraph(f"Well: {s.well or 'unset'} | Condition: {s.condition or 'unset'} | Biological replicate: {s.replicate or 'unset'}")
        paragraph(f"Calibration: {s.calibration.summary()} | Reviewed: {s.reviewed}")
        paragraph("Review note: " + (s.review_note or "None"))
        if not s.has_segmentation:
            paragraph("No segmentation results.")
            continue
        results = compute_results(s)
        paragraph(f"Accepted cells: {results.counts['cells_accepted']} | Excluded objects: {results.counts['objects_excluded']} | Unresolved objects: {results.counts['unresolved_accepted']}")
        if s.annotation is not None and s.annotation.found:
            paragraph("Burned-in scale bar detected: " + s.annotation.note
                      + ". Objects touching it are excluded (dotted outline).")
        display, _ = make_display(s.image)
        overlay = make_overlay(display, results.qc_labels, results.objects, results.clusters,
                               excluded_ids=s.qc.excluded_ids, annotation=s.annotation)
        buffer = BytesIO()
        PILImage.fromarray(overlay).save(buffer, format="PNG")
        buffer.seek(0)
        ratio = min(468 / overlay.shape[1], 190 / overlay.shape[0])
        story.append(Image(buffer, width=overlay.shape[1] * ratio, height=overlay.shape[0] * ratio))
        paragraph("Blue: isolated cell. Orange: clustered cell. Aqua: unresolved group. IDs link the overlay to CSV measurements.")
        paragraph("Accepted-cell measurements", "Heading2")
        units = "um" if s.calibration.is_calibrated else "px"
        metrics = ["area_" + units + "2", "major_axis_" + units, "minor_axis_" + units,
                   "aspect_ratio", "circularity", "solidity"]
        table(pd.DataFrame([{"Metric": name.replace("_", " "),
                             "Mean": round(float(results.cells[name].mean()), 3),
                             "Median": round(float(results.cells[name].median()), 3)}
                            for name in metrics if name in results.cells]))
        paragraph("Measurement definitions", "Heading2")
        paragraph("Area is mask pixel count scaled by pixel area. Major/minor axes are equivalent-ellipse axes. Feret max is maximum caliper diameter. Aspect ratio is major/minor axis; circularity is 4*pi*area/perimeter squared; solidity is area/convex area. Physical lengths use um and areas use um2; uncalibrated data use px and px2.")
        story.append(PageBreak())
        paragraph("Methods and QC: " + s.display_name, "Heading1")
        paragraph("Recorded settings", "Heading2")
        if s.nuclei_settings:
            paragraph("Independent nuclear settings: " + str(s.nuclei_settings))
        for name in ("preprocess_params", "segmentation_params", "split_params",
                     "cluster_params", "qc_params"):
            paragraph(name + ": " + str(getattr(s, name).describe()))
        engine = {k: v for k, v in (s.engine_info or {}).items() if k != "parameters"}
        paragraph("Inference: " + str(engine))
        paragraph("QC history", "Heading2")
        if not s.qc.log and not s.edit_log:
            paragraph("No manual corrections or exclusions recorded.")
        for entry in s.qc.log:
            paragraph(f"{entry.timestamp}: {entry.action} object {entry.object_id}: {entry.reason}")
        for entry in s.edit_log:
            paragraph(str(entry))
    def footer(canvas, doc):
        canvas.setFont("Helvetica", 8)
        canvas.drawString(72, 30, "CellScope | Descriptive analysis | " + str(doc.page))
    SimpleDocTemplate(str(path), pagesize=(612, 792), rightMargin=72, leftMargin=72,
                      topMargin=48, bottomMargin=48).build(story, onFirstPage=footer, onLaterPages=footer)
    return str(path)
