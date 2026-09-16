"""Review tab: inspect, exclude and restore cells, correct boundaries, mark reviewed.

Every review action returns the same set of views -- viewer, measurements,
navigator, status -- in one response, so there are no chained follow-up events.
Rendering reads cached results, and zoom, pan and opacity never reach the server.
"""

from __future__ import annotations

import json
from types import SimpleNamespace

import gradio as gr
import pandas as pd

from src.pipeline import compute_results, excluded_cell_measurements
from src.qc import (
    REASON_BORDER,
    exclude_border_touching,
    exclude_by_area,
    exclude_cluster,
    exclude_object,
    exclusion_reasons,
    qc_log_dataframe,
    restore_object,
)
from src.review import checkpoint, correct_mask, history, signature, validate_review
from src.visualize import _resize_labels

from .theme import legend_html, section
from .viewer import SHORTCUTS_MD, parse_click, viewer_html
from .views import active, display_base, image_table, locked, nav_html, status_html, strip_html

MODES = ["Toggle inclusion", "Inspect", "Collect correction points"]
TABLE_COLUMNS = ["cell_id", "included", "type", "reason"]
DETAIL_KEYS = ["area_px2", "area_um2", "major_axis_px", "major_axis_um", "minor_axis_px",
               "minor_axis_um", "aspect_ratio", "circularity", "solidity", "cluster_id",
               "cluster_status"]


# --------------------------------------------------------------------------- #
# Measurements linked to the selection
# --------------------------------------------------------------------------- #


def cell_rows(session) -> pd.DataFrame:
    """One row per object: its QC state and, when it has them, its measurements."""
    cells = compute_results(session).cells
    excluded = excluded_cell_measurements(session)
    measured = pd.concat([f for f in (cells, excluded) if len(f)], ignore_index=True) \
        if len(cells) or len(excluded) else pd.DataFrame(columns=["cell_id"])
    reasons = exclusion_reasons(session.qc)
    rows = pd.DataFrame([
        {"cell_id": o.object_id, "included": session.qc.is_included(o.object_id),
         "type": "cell" if o.is_resolved else "unresolved group",
         "reason": reasons.get(o.object_id, "")}
        for o in session.objects
    ], columns=TABLE_COLUMNS)
    if len(measured):
        keep = ["cell_id"] + [k for k in DETAIL_KEYS if k in measured.columns]
        rows = rows.merge(measured[keep], on="cell_id", how="left")
    return rows


def _display_table(rows: pd.DataFrame, calibrated: bool, selected: int):
    unit = "um" if calibrated else "px"
    wanted = TABLE_COLUMNS + [c for c in ("area_" + unit + "2", "major_axis_" + unit,
                                          "aspect_ratio", "circularity", "solidity", "cluster_id")
                              if c in rows.columns]
    table = rows[wanted].copy()
    numeric = [c for c in table.select_dtypes("number").columns if c not in ("cell_id", "cluster_id")]
    if "cluster_id" in table:
        table["cluster_id"] = table["cluster_id"].astype("Int64")

    def highlight(row):
        colour = "background-color: rgba(34,197,94,.22); font-weight: 600"
        return [colour if selected and row["cell_id"] == selected else "" for _ in row]

    # A Styler keeps numbers numeric (so columns still sort) while showing
    # missing values as a dash and three decimals, highlighted or not.
    return (table.style.apply(highlight, axis=1)
            .format(precision=3, na_rep="—", subset=numeric)
            .format(na_rep="—", subset=[c for c in table.columns if c not in numeric]))


def selection_detail(session, rows: pd.DataFrame) -> str:
    selected = int(session.selected_object or 0)
    if not selected:
        return ("Select a cell: click it in **Inspect** mode, click its row below, or type its "
                "number. <kbd>X</kbd> then excludes or restores it.")
    match = rows[rows.cell_id == selected]
    if match.empty:
        return "Cell {} no longer exists; select another.".format(selected)
    row = match.iloc[0]
    state = "included" if row.included else "excluded"
    text = "**Cell {} — {}** · {}".format(selected, state, row.type)
    if not row.included and row.reason:
        text += "  \nReason: {}".format(str(row.reason).replace("|", "/"))
    values = []
    for key in DETAIL_KEYS:
        if key in row and pd.notna(row[key]) and str(row[key]) != "":
            value = row[key]
            value = round(float(value), 3) if isinstance(value, (int, float)) else value
            values.append("{} **{}**".format(key.replace("_", " "), value))
    if values:
        text += "  \n" + " · ".join(values)
    if not row.included:
        text += "  \n*Measurements of excluded cells are diagnostic only.*"
    if row.type != "cell":
        text += ("  \n*An unresolved group has no per-cell measurements and no cell count; "
                 "split it on the right if the boundary is clear.*")
    return text


def review_table(batch):
    """Linked measurements for the active image: (table, detail, points json, note)."""
    s = active(batch)
    if s is None or not s.has_segmentation:
        return (pd.DataFrame(columns=TABLE_COLUMNS), "No cell selected.", "", "")
    rows = cell_rows(s)
    detail = selection_detail(s, rows)
    return rows, detail, json.dumps(s.correction_points), s.review_note


# --------------------------------------------------------------------------- #
# The shared response
# --------------------------------------------------------------------------- #


def review_views(batch, show_ids=True, show_clusters=False, message=""):
    """Everything the Review tab shows, in one response."""
    session = active(batch)
    if session is None or not session.has_segmentation:
        if not message:
            if session is None:
                message = "Add images on the Import tab."
            elif session.error:
                message = "**This image failed to segment:** {}".format(session.error)
            else:
                message = "Not segmented yet. Segment it on the Segment tab to review it."
        empty = pd.DataFrame(columns=TABLE_COLUMNS)
        return (
            batch,
            viewer_html(session, show_ids, show_clusters, message) if session else viewer_html(None),
            pd.DataFrame(columns=["object_id"]), message, nav_html(batch), strip_html(batch),
            image_table(batch), status_html(batch), empty, "", "[]",
            session.review_note if session else "", gr.update(value=None),
        )
    validate_review(session)
    counts = compute_results(session).counts
    summary = "**{} included · {} excluded**{}.".format(
        counts["objects_accepted"], counts["objects_excluded"],
        " · reviewed ✓" if session.reviewed else " · not reviewed yet",
    )
    rows = cell_rows(session)
    return (
        batch,
        viewer_html(session, show_ids, show_clusters),
        qc_log_dataframe(session.qc),
        summary + ("  \n" + message if message else ""),
        nav_html(batch), strip_html(batch), image_table(batch), status_html(batch),
        _display_table(rows, session.calibration.is_calibrated, int(session.selected_object or 0)),
        selection_detail(session, rows),
        json.dumps(session.correction_points),
        session.review_note,
        gr.update(value=int(session.selected_object) if session.selected_object else None),
    )


def on_toggle_labels(show_ids, show_clusters, batch):
    return review_views(batch, show_ids, show_clusters)


# --------------------------------------------------------------------------- #
# Clicks
# --------------------------------------------------------------------------- #


def _click(batch, point, show_ids, show_clusters):
    """Handle a click at display-pixel ``point`` according to the click mode."""
    session = active(batch)
    if session is None or not session.has_segmentation or point is None:
        return review_views(batch, show_ids, show_clusters)
    base = display_base(session)
    x, y = int(point[0]), int(point[1])
    if not (0 <= y < base.shape[0] and 0 <= x < base.shape[1]):
        return review_views(batch, show_ids, show_clusters)
    full_h, full_w = session.object_labels.shape
    if session.review_mode == "Collect correction points":
        ox = min(full_w - 1, int((x + .5) * full_w / base.shape[1]))
        oy = min(full_h - 1, int((y + .5) * full_h / base.shape[0]))
        session.correction_points.append((ox, oy))
        return review_views(batch, show_ids, show_clusters,
                            "Point {} at ({}, {}).".format(len(session.correction_points), ox, oy))
    key = ("small_labels", session.segmentation_version, id(session.object_labels), base.shape[:2])
    if session.view_cache.get("small_labels_key") != key:
        session.view_cache["small_labels"] = _resize_labels(session.object_labels, base.shape[:2])
        session.view_cache["small_labels_key"] = key
    object_id = int(session.view_cache["small_labels"][y, x])
    if object_id == 0 or session.object_by_id(object_id) is None:
        return review_views(batch, show_ids, show_clusters, "No cell there.")
    session.selected_object = object_id
    if session.review_mode == "Inspect":
        return review_views(batch, show_ids, show_clusters, "Selected cell {}.".format(object_id))
    return _toggle(batch, session, object_id, show_ids, show_clusters)


def _toggle(batch, session, object_id, show_ids, show_clusters):
    checkpoint(session)
    if session.qc.is_included(object_id):
        exclude_object(session.qc, object_id, "deselected during review")
        action = "Excluded"
    else:
        restore_object(session.qc, object_id, "selected during review")
        action = "Restored"
    session.reviewed = False
    return review_views(batch, show_ids, show_clusters, "{} cell {}.".format(action, object_id))


@locked
def on_review_click(show_ids, show_clusters, batch, evt: gr.SelectData):
    """A click on a display-sized image, as Gradio reports it."""
    return _click(batch, evt.index[:2] if evt.index is not None else None, show_ids, show_clusters)


@locked
def on_canvas_click(payload, show_ids, show_clusters, batch):
    """A click from the viewer, relayed through the hidden bridge textbox."""
    return _click(batch, parse_click(payload), show_ids, show_clusters)


@locked
def on_mode(mode, show_ids, show_clusters, batch):
    session = active(batch)
    if session is not None:
        session.review_mode = mode if mode in MODES else MODES[0]
    return review_views(batch, show_ids, show_clusters)


@locked
def on_toggle_selected(show_ids, show_clusters, batch):
    session = active(batch)
    if session is None or not session.has_segmentation or not session.selected_object:
        return review_views(batch, show_ids, show_clusters, "Select a cell first.")
    if session.object_by_id(session.selected_object) is None:
        return review_views(batch, show_ids, show_clusters, "The selected cell no longer exists.")
    return _toggle(batch, session, int(session.selected_object), show_ids, show_clusters)


@locked
def on_select_cell(number, show_ids, show_clusters, batch):
    session = active(batch)
    if session is None or not session.has_segmentation or number in (None, ""):
        return review_views(batch, show_ids, show_clusters)
    object_id = int(number)
    if session.object_by_id(object_id) is None:
        return review_views(batch, show_ids, show_clusters, "No cell {}.".format(object_id))
    session.selected_object = object_id
    return review_views(batch, show_ids, show_clusters, "Selected cell {}.".format(object_id))


@locked
def on_table_select(show_ids, show_clusters, batch, evt: gr.SelectData):
    """Select the cell whose row was clicked, by its ID rather than the row position."""
    session = active(batch)
    if session is None or not session.has_segmentation:
        return review_views(batch, show_ids, show_clusters)
    row = getattr(evt, "row_value", None)
    try:
        object_id = int(row[0]) if row else None
    except (TypeError, ValueError):
        object_id = None
    if object_id is None or session.object_by_id(object_id) is None:
        return review_views(batch, show_ids, show_clusters)
    session.selected_object = object_id
    return review_views(batch, show_ids, show_clusters, "Selected cell {}.".format(object_id))


# --------------------------------------------------------------------------- #
# Review state and bulk actions
# --------------------------------------------------------------------------- #


@locked
def on_mark_reviewed(note, show_ids, show_clusters, batch):
    session = active(batch)
    if session is None:
        return review_views(batch, show_ids, show_clusters)
    if not session.has_segmentation:
        return review_views(batch, show_ids, show_clusters,
                            "Segment this image before marking it reviewed.")
    session.reviewed = True
    session.reviewed_signature = signature(session)
    session.review_note = str(note or "").strip()
    remaining = [s.display_name for s in batch.images if s.has_segmentation and not s.reviewed]
    message = "Marked **{}** reviewed. {} of {} done.".format(
        session.display_name, batch.n_reviewed, len(batch.images))
    if remaining:
        message += " Next unreviewed: {} (press <kbd>→</kbd>).".format(remaining[0])
    return review_views(batch, show_ids, show_clusters, message)


@locked
def on_unmark_reviewed(show_ids, show_clusters, batch):
    session = active(batch)
    if session is not None:
        session.reviewed = False
    return review_views(batch, show_ids, show_clusters, "Marked unreviewed.")


@locked
def on_store_note(note, batch):
    session = active(batch)
    if session is not None:
        session.review_note = str(note or "")
    return batch


def parse_ids(text, session):
    """Whitespace/comma-separated IDs; anything that is not a real object is reported.

    Excluding a cell that does not exist would put a phantom entry in the QC log.
    """
    known = {o.object_id for o in session.objects}
    ids, unknown = [], []
    for token in str(text or "").replace(",", " ").split():
        try:
            value = int(token)
        except ValueError:
            unknown.append(token)
            continue
        (ids if value in known else unknown).append(value)
    return ids, unknown


@locked
def on_exclude(object_ids, reason, show_ids, show_clusters, batch):
    session = active(batch)
    if session is None or not session.has_segmentation:
        return review_views(batch, show_ids, show_clusters)
    ids, unknown = parse_ids(object_ids, session)
    checkpoint(session)
    changed = sum(exclude_object(session.qc, i, reason or "manual exclusion") for i in ids)
    message = "Excluded {} object(s).".format(changed)
    if unknown:
        message += "  No such cell: {}.".format(", ".join(str(u) for u in unknown))
    return review_views(batch, show_ids, show_clusters, message)


@locked
def on_restore(object_ids, show_ids, show_clusters, batch):
    session = active(batch)
    if session is None or not session.has_segmentation:
        return review_views(batch, show_ids, show_clusters)
    ids, unknown = parse_ids(object_ids, session)
    checkpoint(session)
    changed = sum(restore_object(session.qc, i) for i in ids)
    message = "Restored {} object(s).".format(changed)
    if unknown:
        message += "  No such cell: {}.".format(", ".join(str(u) for u in unknown))
    return review_views(batch, show_ids, show_clusters, message)


@locked
def on_exclude_cluster(cluster_id, show_ids, show_clusters, batch):
    session = active(batch)
    if session is None or not session.has_segmentation:
        return review_views(batch, show_ids, show_clusters)
    results = compute_results(session)
    target = next((c for c in results.clusters if c.cluster_id == int(cluster_id or 0)), None)
    if target is None:
        return review_views(batch, show_ids, show_clusters, "No cluster {}.".format(cluster_id))
    checkpoint(session)
    removed = exclude_cluster(session.qc, target)
    return review_views(batch, show_ids, show_clusters,
                        "Excluded cluster {} ({} object(s)).".format(int(cluster_id), removed))


@locked
def on_exclude_border(show_ids, show_clusters, batch):
    session = active(batch)
    if session is None or not session.has_segmentation:
        return review_views(batch, show_ids, show_clusters)
    checkpoint(session)
    removed = exclude_border_touching(session.qc, session.objects, reason=REASON_BORDER)
    return review_views(batch, show_ids, show_clusters,
                        "Excluded {} border object(s).".format(removed))


@locked
def on_filter_area(min_area, max_area, show_ids, show_clusters, batch):
    session = active(batch)
    if session is None or not session.has_segmentation:
        return review_views(batch, show_ids, show_clusters)
    checkpoint(session)
    removed = exclude_by_area(
        session.qc, session.object_labels,
        min_area=float(min_area) if min_area and float(min_area) > 0 else None,
        max_area=float(max_area) if max_area and float(max_area) > 0 else None,
    )
    return review_views(batch, show_ids, show_clusters,
                        "Excluded {} object(s) by area.".format(removed))


@locked
def on_reset_qc(show_ids, show_clusters, batch):
    session = active(batch)
    if session is not None:
        checkpoint(session)
        for oid in list(session.qc.excluded_ids):
            session.qc.restore(oid, "reset all QC")
    return review_views(batch, show_ids, show_clusters,
                        "All objects restored, including automatic exclusions. Undo to revert.")


@locked
def on_undo(show_ids, show_clusters, batch):
    session = active(batch)
    done = session is not None and history(session)
    return review_views(batch, show_ids, show_clusters, "Undone." if done else "Nothing to undo.")


@locked
def on_redo(show_ids, show_clusters, batch):
    session = active(batch)
    done = session is not None and history(session, redo=True)
    return review_views(batch, show_ids, show_clusters, "Redone." if done else "Nothing to redo.")


@locked
def on_correct(operation, ids, points, show_ids, show_clusters, batch):
    session = active(batch)
    try:
        if session is None:
            raise ValueError("Load an image first.")
        parsed, unknown = parse_ids(ids, session)
        if unknown:
            raise ValueError("Unknown IDs: " + ", ".join(map(str, unknown)))
        if not parsed and session.selected_object:
            parsed = [int(session.selected_object)]
        vertices = [tuple(map(int, p)) for p in json.loads(points or "[]")]
        correct_mask(session, operation, parsed, vertices)
        session.correction_points.clear()
        message = "{} applied. Check the new boundaries before marking reviewed.".format(operation)
    except Exception as exc:
        message = "Correction failed: " + str(exc)
    return review_views(batch, show_ids, show_clusters, message)


@locked
def on_clear_correction_points(show_ids, show_clusters, batch):
    session = active(batch)
    if session is not None:
        session.correction_points.clear()
    return review_views(batch, show_ids, show_clusters, "Correction points cleared.")


def _step(batch, delta, unreviewed_only):
    if batch is None or batch.is_empty:
        return
    count = len(batch.images)
    index = batch.active_index
    for _ in range(count):
        index = (index + delta) % count
        session = batch.images[index]
        if not unreviewed_only or (session.has_segmentation and not session.reviewed):
            batch.active_index = index
            return
    batch.active_index = (batch.active_index + delta) % count


@locked
def on_review_prev(show_ids, show_clusters, batch):
    _step(batch, -1, False)
    return review_views(batch, show_ids, show_clusters)


@locked
def on_review_next(show_ids, show_clusters, batch):
    _step(batch, 1, False)
    return review_views(batch, show_ids, show_clusters)


@locked
def on_next_unreviewed(show_ids, show_clusters, batch):
    _step(batch, 1, True)
    return review_views(batch, show_ids, show_clusters)


# --------------------------------------------------------------------------- #
# Layout
# --------------------------------------------------------------------------- #


def build(batch, shared) -> SimpleNamespace:
    c = SimpleNamespace()
    with gr.Tab("Review", id="tab_review") as c.tab:
        c.nav = gr.HTML()
        c.strip = shared.review_strip = gr.HTML()
        with gr.Row(equal_height=False):
            with gr.Column(scale=5, min_width=460):
                with gr.Row():
                    c.prev = gr.Button("◀ Previous", size="sm", elem_id="cs-prev")
                    c.next = gr.Button("Next ▶", size="sm", elem_id="cs-next")
                    c.next_unreviewed = gr.Button("Next unreviewed ▶▶", size="sm")
                    c.mode = gr.Radio(MODES, value=MODES[0], label="Click mode", elem_id="cs-mode",
                                      scale=3)
                c.viewer = gr.HTML(elem_id="cs-review-viewer")
                c.bridge = gr.Textbox(elem_id="cs-bridge", visible="hidden", show_label=False)
                gr.HTML(legend_html())
                c.note = gr.Markdown(elem_classes="cs-note", elem_id="cs-review-status")
                with gr.Row():
                    c.show_ids = gr.Checkbox(value=True, label="Cell IDs")
                    c.show_clusters = gr.Checkbox(value=False, label="Cluster IDs")
                with gr.Accordion("Keyboard shortcuts", open=False, elem_id="cs-shortcuts"):
                    gr.Markdown(SHORTCUTS_MD, elem_classes="cs-shortcuts")
            with gr.Column(scale=3, min_width=340):
                gr.HTML(section("Selected cell"))
                with gr.Row():
                    c.cell_number = gr.Number(label="Cell", precision=0, minimum=1, scale=1)
                    c.toggle_selected = gr.Button("Exclude / restore (X)", elem_id="cs-toggle-selected",
                                                  scale=2)
                c.detail = gr.Markdown(elem_classes="cs-note", elem_id="cs-cell-detail")
                gr.HTML(section("This image"))
                c.review_note = gr.Textbox(label="Review note (optional)",
                                           placeholder="out of focus at the left edge")
                with gr.Row():
                    c.mark = gr.Button("Mark reviewed (R)", variant="primary", elem_id="cs-mark-reviewed")
                    c.unmark = gr.Button("Unmark")
                with gr.Row():
                    c.undo = gr.Button("Undo", size="sm", elem_id="cs-undo")
                    c.redo = gr.Button("Redo", size="sm", elem_id="cs-redo")
                    c.reset = gr.Button("Restore all", size="sm", variant="stop")
                with gr.Accordion("Exclude or restore by ID", open=False):
                    c.exclude_ids = gr.Textbox(label="Cell IDs", placeholder="12 15 23")
                    c.exclude_reason = gr.Textbox(label="Reason", value="segmentation error",
                                                  info="Recorded in qc_log.csv")
                    with gr.Row():
                        c.exclude = gr.Button("Exclude", variant="stop")
                        c.restore = gr.Button("Restore")
                with gr.Accordion("Bulk filters", open=False):
                    with gr.Row():
                        c.cluster_id = gr.Number(value=1, label="Cluster ID", precision=0)
                        c.exclude_cluster = gr.Button("Exclude cluster", variant="stop")
                    c.exclude_border = gr.Button("Exclude border-touching cells", variant="stop")
                    with gr.Row():
                        c.filter_min = gr.Number(value=0, label="Min area (px)")
                        c.filter_max = gr.Number(value=0, label="Max area (px)")
                    c.filter = gr.Button("Apply area filter", variant="stop")
                with gr.Accordion("Correct segmentation", open=False):
                    gr.Markdown(
                        "**Merge** touching cells by ID. **Split** one cell: switch to "
                        "*Collect correction points* (<kbd>P</kbd>) and click two seeds inside it. "
                        "**Replace boundary**: click three or more polygon vertices. The IDs box "
                        "defaults to the selected cell. The raw segmentation is never changed.",
                        elem_classes="cs-note")
                    c.operation = gr.Radio(["Merge", "Split", "Replace boundary"], value="Split",
                                           label="Correction")
                    c.ids = gr.Textbox(label="Cell IDs", placeholder="selected cell")
                    c.points = gr.Textbox(label="Points (original pixels)", value="[]", max_lines=2)
                    with gr.Row():
                        c.correct = gr.Button("Apply correction", variant="primary")
                        c.clear_points = gr.Button("Clear points")
        with gr.Accordion("Measurements for this image", open=True):
            c.table = gr.Dataframe(interactive=False, wrap=False, max_height=380)
        with gr.Accordion("QC log for this image", open=False):
            c.qc_log = gr.Dataframe(interactive=False, wrap=True)
    return c


def wire(c, batch, shared) -> None:
    labels = [c.show_ids, c.show_clusters]
    outputs = [batch, c.viewer, c.qc_log, c.note, c.nav, c.strip, shared.image_grid,
               shared.status, c.table, c.detail, c.points, c.review_note, c.cell_number]
    shared.review_outputs = outputs
    quiet = {"show_progress": "hidden"}

    for control in labels:
        control.input(on_toggle_labels, labels + [batch], outputs, **quiet)
    c.bridge.input(on_canvas_click, [c.bridge] + labels + [batch], outputs, **quiet,
                   trigger_mode="always_last")
    c.mode.input(on_mode, [c.mode] + labels + [batch], outputs, **quiet)
    c.prev.click(on_review_prev, labels + [batch], outputs, **quiet)
    c.next.click(on_review_next, labels + [batch], outputs, **quiet)
    c.next_unreviewed.click(on_next_unreviewed, labels + [batch], outputs, **quiet)
    c.toggle_selected.click(on_toggle_selected, labels + [batch], outputs, **quiet)
    c.cell_number.submit(on_select_cell, [c.cell_number] + labels + [batch], outputs, **quiet)
    c.table.select(on_table_select, labels + [batch], outputs, **quiet)
    c.mark.click(on_mark_reviewed, [c.review_note] + labels + [batch], outputs, **quiet)
    c.unmark.click(on_unmark_reviewed, labels + [batch], outputs, **quiet)
    c.review_note.blur(on_store_note, [c.review_note, batch], [batch], **quiet)
    c.undo.click(on_undo, labels + [batch], outputs, **quiet)
    c.redo.click(on_redo, labels + [batch], outputs, **quiet)
    c.reset.click(on_reset_qc, labels + [batch], outputs, **quiet)
    c.exclude.click(on_exclude, [c.exclude_ids, c.exclude_reason] + labels + [batch], outputs, **quiet)
    c.restore.click(on_restore, [c.exclude_ids] + labels + [batch], outputs, **quiet)
    c.exclude_cluster.click(on_exclude_cluster, [c.cluster_id] + labels + [batch], outputs, **quiet)
    c.exclude_border.click(on_exclude_border, labels + [batch], outputs, **quiet)
    c.filter.click(on_filter_area, [c.filter_min, c.filter_max] + labels + [batch], outputs, **quiet)
    c.correct.click(on_correct, [c.operation, c.ids, c.points] + labels + [batch], outputs,
                    show_progress_on=[c.viewer])
    c.clear_points.click(on_clear_correction_points, labels + [batch], outputs, **quiet)
    c.tab.select(on_toggle_labels, labels + [batch], outputs, **quiet)
    shared.review_labels = labels
