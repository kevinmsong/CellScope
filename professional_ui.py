"""Project management and review controls kept separate from the main app wiring."""
import json
import tempfile
import time
from pathlib import Path
from types import SimpleNamespace

import gradio as gr
import pandas as pd

from src.autosave import Autosaver
from src.autosave import recovery_path as _recovery_path
from src.pipeline import compute_results, iter_batch_segmentation
from src.project import apply_preset, load_project, preset_dict, save_project
from src.quality import experimental_summary, readiness
from src.report import write_report
from src.review import correct_mask, history

PROJECTS = Path(__file__).resolve().parent / "projects"


AUTOSAVER = Autosaver(PROJECTS)


def recovery_path(batch):
    # Derived from a hash of the batch ID, never from user-supplied text.
    return _recovery_path(PROJECTS, batch)


def autosave(batch, wait=False):
    """Save in the background if anything changed; returns a status line."""
    return AUTOSAVER.request(batch, wait=wait)


def review_table(batch):
    s = batch.active if batch else None
    if s is None or not s.has_segmentation:
        return pd.DataFrame(columns=["cell_id", "included", "status", "reason"]), "No cell selected.", "", ""
    cells = compute_results(s).cells
    # Excluded cells retain diagnostic measurements, without entering accepted summaries.
    from src.pipeline import excluded_cell_measurements
    measurements = excluded_cell_measurements(s)
    if len(measurements):
        cells = pd.concat([cells, measurements], ignore_index=True)
    reasons = {a.object_id: a.reason for a in s.qc.log}
    rows = pd.DataFrame([{"cell_id": o.object_id, "included": s.qc.is_included(o.object_id),
                          "status": o.status, "reason": reasons.get(o.object_id, "")}
                         for o in s.objects])
    if not cells.empty:
        rows = rows.merge(cells, on="cell_id", how="left")
    selected = rows[rows.cell_id == s.selected_object]
    detail = "Choose Inspect and click a cell, or click its table row."
    if not selected.empty:
        row = selected.iloc[0]
        def safe(value):
            return str(value).replace("|", "\\|").replace("\n", " ")
        detail = f"**Cell {int(row.cell_id)} — {'included' if row.included else 'excluded'}**\n\n"
        detail += "| Measurement | Value |\n|---|---|\n"
        wanted = ["status", "reason", "area_px2", "area_um2", "major_axis_px", "major_axis_um",
                  "minor_axis_px", "minor_axis_um", "aspect_ratio", "circularity", "solidity"]
        for key in wanted:
            if key in row and pd.notna(row[key]) and str(row[key]):
                value = round(float(row[key]), 3) if isinstance(row[key], (int, float)) else row[key]
                detail += f"| {key.replace('_', ' ')} | {safe(value)} |\n"
        if not row.included:
            detail += "\nExcluded measurements are diagnostic only and do not enter accepted-cell summaries."
    return rows, detail, json.dumps(s.correction_points), s.review_note


def build_review_controls():
    c = SimpleNamespace()
    with gr.Accordion("View and correction tools", open=False):
        c.mode = gr.Radio(["Toggle inclusion", "Inspect", "Collect correction points"],
                          value="Toggle inclusion", label="Click action")
        with gr.Row():
            c.undo = gr.Button("Undo", elem_id="cs-undo")
            c.redo = gr.Button("Redo", elem_id="cs-redo")
        c.zoom = gr.Slider(1, 8, value=1, step=.25, label="Zoom")
        c.pan_x = gr.Slider(0, 100, value=50, label="Horizontal position (%)")
        c.pan_y = gr.Slider(0, 100, value=50, label="Vertical position (%)")
        c.alpha = gr.Slider(0, .8, value=.28, step=.02, label="Overlay opacity")
        c.original = gr.Checkbox(label="Show original image", value=False)
        c.fit = gr.Button("Fit image")
        gr.Markdown("Keyboard: Ctrl+Z undo, Ctrl+Shift+Z redo (outside text fields). Use the image fullscreen control for a larger view.")
    with gr.Accordion("Correct segmentation", open=False):
        gr.Markdown("Merge touching IDs, split one ID using two interior seeds, or replace one boundary with a polygon. Choose **Collect correction points**, then click the image. Coordinates are original-image pixels.")
        c.operation = gr.Radio(["Merge", "Split", "Replace boundary"], value="Merge", label="Correction")
        c.ids = gr.Textbox(label="Cell IDs", placeholder="12, 15")
        c.points = gr.Textbox(label="Points (JSON)", value="[]")
        c.clear = gr.Button("Clear points")
        c.correct = gr.Button("Apply correction")
    return c


def build_tools(batch, review):
    c = SimpleNamespace(review=review)
    with gr.Tab("Projects & presets"):
        gr.Markdown("Save a portable project to resume later. Autosaves are local to this computer in `projects/`; they include your images and are excluded from Git.")
        c.save = gr.Button("Save project", variant="primary")
        c.download = gr.File(label="Project download")
        c.upload = gr.File(label="Open project", file_types=[".cellscope"], type="filepath")
        c.open = gr.Button("Open selected project")
        c.recoveries = gr.Dropdown(label="Local recovery", choices=[r["name"] for r in AUTOSAVER.recoveries()])
        c.scan = gr.Button("Refresh recovery list")
        c.recover = gr.Button("Recover selected autosave")
        c.auto = gr.Checkbox(value=True, label="Autosave every 60 seconds")
        c.auto_note = gr.Markdown()
        c.timer = gr.Timer(60)
        c.note = gr.Markdown()
        with gr.Accordion("Analysis presets", open=False):
            gr.Markdown("Presets include channel and segmentation settings. Calibration stays specific to the images. Apply a preset, then return to Segment to inspect the settings before running.")
            c.preset_name = gr.Textbox(label="Preset name", value="My analysis")
            c.save_preset = gr.Button("Download current preset")
            c.preset_download = gr.File(label="Preset download")
            c.preset_upload = gr.File(label="Load preset", file_types=[".json"], type="filepath")
            c.load_preset = gr.Button("Apply preset to batch")
            c.preset_note = gr.Markdown()
    with gr.Tab("Quality & design") as quality_tab:
        c.check = gr.Button("Refresh readiness checks", variant="primary")
        c.checks = gr.Dataframe(interactive=False, label="Readiness")
        gr.Markdown("Set the well, condition, and biological replicate for each image. Reuse a replicate label for technical wells from the same biological sample. Rows missing any design label are omitted from experimental summaries.")
        c.design = gr.Dataframe(headers=["Image", "Well", "Condition", "Biological replicate"], interactive=True)
        c.save_design = gr.Button("Apply experimental design")
        c.design_note = gr.Markdown()
        c.wells = gr.Dataframe(interactive=False, label="Well summaries")
        c.reps = gr.Dataframe(interactive=False, label="Biological replicate summaries")
        c.conditions = gr.Dataframe(interactive=False, label="Condition comparison")
        gr.Markdown("Fields pool within wells. Well means receive equal weight within each replicate; replicate means receive equal weight within conditions. SD is unavailable with only one replicate. No cell-level p-values are calculated.")
        c.pdf = gr.Button("Build PDF report")
        c.pdf_file = gr.File(label="Report")
        c.pdf_note = gr.Markdown()
    with gr.Tab("Processing"):
        gr.Markdown("Resume processes missing or failed images using their saved settings. Retry processes failures only. Cancel finishes the current inference call before stopping; completed images remain available.")
        c.engine = gr.Dropdown(["auto", "cellpose", "threshold_watershed"], value="auto", label="Engine")
        with gr.Row():
            c.resume = gr.Button("Resume pending images")
            c.retry = gr.Button("Retry failed images")
            c.cancel = gr.Button("Cancel processing", variant="stop")
        c.progress = gr.Markdown()
    def wire(demo, batch, nav_outputs, review_outputs, results_outputs, show_ids, show_clusters, preview, note_box, param_widgets):
        import app
        r = c.review
        def save(b):
            try:
                if not b or b.is_empty:
                    raise ValueError("Load images first.")
                path = save_project(b, recovery_path(b))
                AUTOSAVER.mark_clean(b, path)
                return path, "Project saved. Download it for a portable copy."
            except Exception as exc:
                return None, "Save failed: " + str(exc)
        def open_file(path, b):
            try:
                if not path:
                    raise ValueError("Choose a project first.")
                notes = []
                loaded = load_project(path, notes)
                # Keep the currently open work recoverable before replacing it.
                if b and not b.is_empty:
                    autosave(b, wait=True)
                AUTOSAVER.mark_clean(loaded)
                return loaded, "Project opened." + "".join("  \n" + n for n in notes)
            except Exception as exc:
                return b, "Open failed: " + str(exc)
        def recover(name, b):
            if not name or Path(name).name != name:
                return b, "Select a recovery file."
            return open_file(PROJECTS / name, b)
        def sync(b):
            return app._blank_views(b, "Project loaded.") + app._active_controls(b)
        # Use the same existing navigation callback to refresh all original widgets.
        def sync_nav(b):
            return (b, app.image_table(b), app.nav_html(b), app.strip_html(b), "Project loaded.", app.status_html(b)) + tuple(app._active_controls(b))
        def params(b):
            target = b.active or b
            seg, pre, split, cluster = (target.segmentation_params, target.preprocess_params,
                                        target.split_params, target.cluster_params)
            return (seg.model, seg.diameter or 0, seg.cellprob_threshold, seg.flow_threshold,
                    seg.min_object_area, seg.use_gpu, pre.background_subtract, pre.background_radius_px,
                    pre.gaussian_sigma, pre.normalize_contrast, split.enabled, split.min_saddle_depth_frac,
                    split.solidity_max, split.min_fragment_area_frac, cluster.contact_distance_px,
                    seg.nuclear_channel or "none", seg.analysis_scale)
        c.save.click(save, [batch], [c.download, c.note])
        for button, fn, source in [(c.open, open_file, c.upload), (c.recover, recover, c.recoveries)]:
            button.click(fn, [source, batch], [batch, c.note]).then(sync_nav, [batch], nav_outputs).then(
                app.on_toggle_labels, [show_ids, show_clusters, batch], review_outputs).then(
                app.on_refresh_results, [batch], results_outputs).then(params, [batch], param_widgets)
        c.scan.click(lambda: gr.update(choices=[r["name"] for r in AUTOSAVER.recoveries()]), outputs=c.recoveries)
        def tick(enabled, b):
            try:
                return autosave(b) if enabled else "Autosave paused."
            except Exception as exc:
                return "Autosave failed: " + str(exc) + ". Save a project manually."
        c.timer.tick(tick, [c.auto, batch], [c.auto_note])
        def save_preset(name, b):
            import re
            folder = Path(tempfile.mkdtemp(prefix="cellscope_preset_"))
            path = folder / ((re.sub(r"[^\w-]", "_", name or "preset")[:80] or "preset") + ".json")
            data = preset_dict(b)
            data["name"] = name
            path.write_text(json.dumps(data, indent=2), encoding="utf-8")
            return str(path)
        def load_preset(path, b):
            try:
                data = json.loads(Path(path).read_text(encoding="utf-8"))
                return apply_preset(b, data), "Preset applied; images need review after re-segmentation."
            except Exception as exc:
                return b, "Preset failed: " + str(exc)
        c.save_preset.click(save_preset, [c.preset_name, batch], c.preset_download)
        c.load_preset.click(load_preset, [c.preset_upload, batch], [batch, c.preset_note]).then(params, [batch], param_widgets).then(sync_nav, [batch], nav_outputs)
        def quality(b):
            design = pd.DataFrame([[s.display_name, s.well, s.condition, s.replicate] for s in b.images],
                                  columns=["Image", "Well", "Condition", "Biological replicate"])
            return readiness(b), design, *experimental_summary(b)
        c.check.click(quality, [batch], [c.checks, c.design, c.wells, c.reps, c.conditions])
        quality_tab.select(quality, [batch], [c.checks, c.design, c.wells, c.reps, c.conditions])
        def design(table, b):
            if len(table) != len(b.images) or list(table.iloc[:, 0]) != [s.display_name for s in b.images]:
                return b, "Image list changed. Refresh checks before editing the design."
            for s, row in zip(b.images, table.fillna("").itertuples(index=False, name=None)):
                s.well, s.condition, s.replicate = [str(v).strip() for v in row[1:4]]
            return b, "Experimental design saved."
        c.save_design.click(design, [c.design, batch], [batch, c.design_note]).then(
            quality, [batch], [c.checks, c.design, c.wells, c.reps, c.conditions])
        def pdf(b):
            try:
                if not b or b.is_empty:
                    raise ValueError("Load images first.")
                path = Path(tempfile.mkdtemp(prefix="cellscope_report_")) / "CellScope_report.pdf"
                return write_report(b, path), "Report generated from current QC-approved measurements."
            except Exception as exc:
                return None, "Report failed: " + str(exc)
        c.pdf.click(pdf, [batch], [c.pdf_file, c.pdf_note])
        def process(engine, b, failed_only=False):
            import copy
            target = copy.copy(b)
            target.images = [s for s in b.images if s.error] if failed_only else b.images
            start = time.monotonic()
            count = 0
            for position, total, session, report in iter_batch_segmentation(target, engine=engine, only_missing=True):
                count += 1
                b.active_index = b.images.index(session)
                remaining = (time.monotonic() - start) / count * (total - count)
                save_note = tick(True, b)
                yield b, f"Completed {count}/{total}; estimated remaining {remaining:.0f}s. {save_note}"
            if count == 0:
                yield b, "No images need processing."
        def retry(engine, b):
            yield from process(engine, b, True)
        runs = []
        for button, fn in [(c.resume, process), (c.retry, retry)]:
            event = button.click(fn, [c.engine, batch], [batch, c.progress])
            runs.append(event)
            event.then(app.on_toggle_labels, [show_ids, show_clusters, batch], review_outputs)
        c.cancel.click(None, cancels=runs, queue=False)
        # Review controls share the active image and existing review output contract.
        def view(mode, zoom, x, y, alpha, original, b):
            if b.active:
                s = b.active
                s.review_mode, s.review_zoom = mode, zoom
                s.review_pan_x, s.review_pan_y = x, y
                s.review_alpha, s.review_original = alpha, original
            return b
        view_inputs = [r.mode, r.zoom, r.pan_x, r.pan_y, r.alpha, r.original, batch]
        for control in view_inputs[:-1]:
            control.input(view, view_inputs, [batch]).then(app.on_toggle_labels, [show_ids, show_clusters, batch], review_outputs)
        r.fit.click(lambda: (1, 50, 50), outputs=[r.zoom, r.pan_x, r.pan_y]).then(
            view, view_inputs, [batch]).then(app.on_toggle_labels, [show_ids, show_clusters, batch], review_outputs)
        def undo(b):
            if b.active:
                history(b.active)
            return b
        def redo(b):
            if b.active:
                history(b.active, True)
            return b
        for button, fn in [(r.undo, undo), (r.redo, redo)]:
            button.click(fn, [batch], [batch]).then(app.on_toggle_labels, [show_ids, show_clusters, batch], review_outputs)
        def correct(operation, ids, points, b, show, clusters):
            try:
                if not b.active:
                    raise ValueError("Load an image first.")
                parsed, unknown = app._parse_ids(ids, b.active)
                if unknown:
                    raise ValueError("Unknown IDs: " + str(unknown))
                vertices = [tuple(map(int, p)) for p in json.loads(points)]
                correct_mask(b.active, operation, parsed, vertices)
                b.active.correction_points.clear()
                message = "Correction applied. Review the new boundaries before marking reviewed."
            except Exception as exc:
                message = "Correction failed: " + str(exc)
            return app._review_views(b, show, clusters, message)
        r.correct.click(correct, [r.operation, r.ids, r.points, batch, show_ids, show_clusters], review_outputs)
        def clear(b):
            if b.active:
                b.active.correction_points.clear()
            return b, "[]"
        r.clear.click(clear, [batch], [batch, r.points])
        def store_note(note, b):
            if b.active:
                b.active.review_note = str(note or "")
            return b
        note_box.input(store_note, [note_box, batch], [batch], show_progress="hidden")
        preview.change(review_table, [batch], [r.table, r.detail, r.points, note_box], show_progress="hidden")
        def view_state(b):
            s = b.active
            if s is None:
                return ("Toggle inclusion", 1, 50, 50, .28, False)
            return s.review_mode, s.review_zoom, s.review_pan_x, s.review_pan_y, s.review_alpha, s.review_original
        preview.change(view_state, [batch], view_inputs[:-1], show_progress="hidden")
        def table_click(table, b, show, clusters, evt: gr.SelectData):
            if b.active and len(table):
                b.active.selected_object = int(table.iloc[evt.index[0]]["cell_id"])
            return app._review_views(b, show, clusters)
        r.table.select(table_click, [r.table, batch, show_ids, show_clusters], review_outputs)
        demo.load(None, js="""() => { if (window.cellscopeKeys) return; window.cellscopeKeys = true;
          document.addEventListener('keydown', e => {
            if (/INPUT|TEXTAREA|SELECT/.test(e.target.tagName) || e.target.isContentEditable) return;
            if ((e.ctrlKey || e.metaKey) && e.key.toLowerCase() === 'z') {
              const button = document.querySelector(e.shiftKey ? '#cs-redo' : '#cs-undo');
              if (button && button.getClientRects().length) { e.preventDefault(); button.click(); }
            }
          }); }""")
    c.wire = wire
    return c
