"""Batch tab: import images, choose channels, calibrate."""

from __future__ import annotations

import os
from types import SimpleNamespace

import gradio as gr
import pandas as pd

from src.calibration import (
    calibration_from_reference_bar,
    calibration_from_resolution,
    calibration_from_scale_bar,
    display_to_original,
    uncalibrated,
)
from src.image_io import available_channels, describe_channel_signal, load_image, make_display
from src.pipeline import nuclear_array_for
from src.scalebar import detect_annotation, detect_scale_bar
from src.types import AnalysisSession, BatchSession, SegmentationParams

from .theme import section
from .views import (
    DASH,
    IMAGE_TABLE_COLUMNS,
    NO_NUCLEAR,
    active,
    batch_views,
    display_base,
    empty_frame,
    error_text,
    image_table,
    locked,
    nav_html,
    status_html,
    strip_html,
)

CALIBRATION_METHODS = (
    "Scale bar in image",
    "Scale bar on a reference frame",
    "Enter resolution",
    "Work in pixels",
)
SCOPE_CHOICES = ["Whole batch", "This image only"]

#: Set on a batch when the title has been clicked once and is awaiting
#: confirmation. Held on the batch, not a module global, so two users cannot
#: arm each other's reset.
RESET_ARMED = "_reset_armed"


# --------------------------------------------------------------------------- #
# Active image
# --------------------------------------------------------------------------- #


def active_controls(batch):
    """Widget updates that follow whichever image is active."""
    session = active(batch)
    if session is None or session.image is None:
        return (
            gr.update(choices=["grayscale"], value="grayscale"),
            gr.update(choices=[NO_NUCLEAR], value=NO_NUCLEAR),
            None,
            "",
        )
    choices = available_channels(session.image)
    signal = describe_channel_signal(session.image)
    report = "  \n".join(
        "- **{}** mean {:.1f}, max {:.0f}".format(name, values["mean"], values["max"])
        for name, values in signal.items()
    )
    note = (
        "**{}** ({} × {} px).  \nPer-channel signal:  \n{}\n\n"
        "*JPEG chroma subsampling leaks a little into the other channels, so a "
        "non-zero mean does not mean a channel carries your stain.*"
    ).format(session.image.filename, session.image.width, session.image.height, report)
    return (
        gr.update(choices=choices, value=session.channel),
        gr.update(
            choices=[NO_NUCLEAR] + choices,
            value=session.segmentation_params.nuclear_channel or NO_NUCLEAR,
        ),
        display_base(session),
        note,
    )


def on_home(batch):
    """Clear everything and return to the Batch tab, after a confirming click."""
    def views(target, note, tabs_update):
        return (
            target, note, tabs_update,
            image_table(target), nav_html(target), strip_html(target), status_html(target),
            *active_controls(target),
        )

    if batch is not None and batch.images and not getattr(batch, RESET_ARMED, False):
        setattr(batch, RESET_ARMED, True)
        from src.batch import headline_counts

        counts = headline_counts(batch)
        return views(
            batch,
            "**Click CellScope again to discard {} image(s) and {} cell(s).** "
            "Anything not saved or exported will be lost; the last autosave stays "
            "under Projects.".format(counts["n_images"], counts["n_cells"]),
            gr.Tabs(),
        )
    return views(BatchSession(), "New batch started.", gr.Tabs(selected="tab_batch"))


@locked
def on_batch_label(label, batch):
    if batch is not None:
        setattr(batch, RESET_ARMED, False)
        batch.label = str(label or "").strip()
    return batch, status_html(batch)


@locked
def on_upload(file_paths, batch):
    """Add images to the batch, seeded with the shared defaults."""
    batch = batch or BatchSession()
    if not file_paths:
        return (*batch_views(batch), gr.update(), gr.update(), None, "")

    # Content hashes, not filenames: two condition folders may both number
    # their fields 1.jpg..10.jpg, and those are genuinely different images.
    seen = {s.image.sha256 for s in batch.images if s.image is not None}
    added, failed, duplicates = 0, [], []
    for path in file_paths:
        try:
            record = load_image(path)
        except Exception as exc:
            failed.append("{}: {}".format(os.path.basename(str(path)), exc))
            continue
        if record.sha256 in seen:
            # The same field twice would count its cells twice when pooled.
            duplicates.append(record.filename)
            continue
        seen.add(record.sha256)
        session = AnalysisSession()
        session.image = record
        session.display_rgb, session.display_scale = make_display(record)
        session.channel = available_channels(record)[0]
        batch.add(session)
        batch.apply_channels(session)
        added += 1

    message = "Added **{}** image(s). Batch now holds {}.".format(added, len(batch.images))
    rulers = sum(
        1 for s in batch.images[-added:] if added and detect_annotation(s.image.pixels).found
    ) if added else 0
    if rulers:
        message += ("  \n{} image(s) carry a burned-in scale bar. Cells touching it will be "
                    "excluded automatically, and you can calibrate from it below.".format(rulers))
    if duplicates:
        message += "\n\n**Skipped {} already in this batch** ({}). The same field " \
                   "counted twice would double its cells in the pooled result.".format(
                       len(duplicates),
                       ", ".join(duplicates[:5]) + ("..." if len(duplicates) > 5 else ""),
                   )
    if failed:
        message += "\n\n**Could not load:** " + "; ".join(failed)
    return (*batch_views(batch, message), *active_controls(batch))


@locked
def on_select_image(batch, evt: gr.SelectData):
    """Clicking a table row makes that image active."""
    if batch is None or batch.is_empty:
        return (*batch_views(batch), *active_controls(batch))
    row = evt.index[0] if isinstance(evt.index, (list, tuple)) else evt.index
    batch.active_index = max(0, min(int(row), len(batch.images) - 1))
    return (*batch_views(batch, ""), *active_controls(batch))


@locked
def on_table_edited(table, batch):
    """Write the editable columns (Well, Channel, Nuclear) back onto the images.

    The rest is derived and regenerated, so a stray edit cannot corrupt state.
    """
    if batch is None or batch.is_empty or table is None:
        return batch, image_table(batch), status_html(batch)
    frame = pd.DataFrame(table)
    for index, session in enumerate(batch.images):
        if index >= len(frame):
            break
        row = frame.iloc[index]
        if "Well" in frame.columns:
            value = row["Well"]
            session.well = "" if pd.isna(value) else str(value).strip()
        if "Channel" in frame.columns and session.image is not None:
            value = str(row["Channel"]).strip().lower()
            if value in available_channels(session.image) and value != session.channel:
                session.channel = value
                session.results_cache = None
        if "Nuclear" in frame.columns and session.image is not None and session.nuclear_image is None:
            value = str(row["Nuclear"]).strip().lower()
            choice = None if value in ("", DASH.lower(), "none", "nan") else value
            known = choice is None or choice in available_channels(session.image)
            if known and session.segmentation_params.nuclear_channel != choice:
                session.segmentation_params = SegmentationParams(
                    **{**session.segmentation_params.describe(), "nuclear_channel": choice}
                )
                session.results_cache = None
    return batch, image_table(batch), status_html(batch)


@locked
def on_remove_image(batch):
    if batch is None or batch.is_empty:
        return (*batch_views(batch, "Nothing to remove."), *active_controls(batch))
    removed = batch.remove(batch.active_index)
    name = removed.display_name if removed else "?"
    return (*batch_views(batch, "Removed **{}** from the batch.".format(name)),
            *active_controls(batch))


@locked
def on_prev(batch):
    if batch is not None and not batch.is_empty:
        batch.active_index = (batch.active_index - 1) % len(batch.images)
    return (*batch_views(batch), *active_controls(batch))


@locked
def on_next(batch):
    if batch is not None and not batch.is_empty:
        batch.active_index = (batch.active_index + 1) % len(batch.images)
    return (*batch_views(batch), *active_controls(batch))


def refresh_batch_view(batch):
    return (*batch_views(batch), *active_controls(batch))


# --------------------------------------------------------------------------- #
# Channels
# --------------------------------------------------------------------------- #


def _set_cell_channel(batch, channel) -> int:
    """Apply a cell channel to every image that has it. Returns how many changed."""
    changed = 0
    for session in batch.images:
        if session.image is None or channel not in available_channels(session.image):
            continue
        if session.channel != channel:
            session.channel = channel
            session.results_cache = None
            changed += 1
    return changed


def _set_nuclear_channel(batch, choice) -> int:
    value = None if choice in (None, "", NO_NUCLEAR) else choice
    changed = 0
    for session in batch.images:
        if session.image is not None and value is not None and value not in available_channels(session.image):
            continue
        if session.segmentation_params.nuclear_channel != value:
            session.segmentation_params = SegmentationParams(
                **{**session.segmentation_params.describe(), "nuclear_channel": value}
            )
            session.results_cache = None
            changed += 1
    return changed


@locked
def on_channel_change(channel, batch):
    """The cell channel is a batch setting: choosing it once covers every image.

    Per-image selection once made it easy to segment eleven of twelve images on
    the wrong channel. Per-image overrides still exist, in the image table.
    """
    if batch is None:
        return batch, image_table(batch), nav_html(batch)
    batch.channel = channel
    _set_cell_channel(batch, channel)
    return batch, image_table(batch), nav_html(batch)


@locked
def on_nuclear_change(choice, batch):
    if batch is None:
        return batch, image_table(batch), nav_html(batch)
    batch.nuclear_channel = choice
    _set_nuclear_channel(batch, choice)
    return batch, image_table(batch), nav_html(batch)


@locked
def on_nuclear_file(file_path, batch):
    """Attach a separate nuclear file to the active image, validated now."""
    session = active(batch)
    if session is None or not session.has_image:
        return batch, "Load an image first.", image_table(batch)
    if not file_path:
        session.nuclear_image = None
        return batch, "Nuclear file cleared.", image_table(batch)
    try:
        record = load_image(file_path)
    except Exception as exc:
        return batch, error_text(exc), image_table(batch)
    session.nuclear_image = record
    try:
        nuclear_array_for(session)
    except ValueError as exc:
        session.nuclear_image = None
        return batch, error_text(exc), image_table(batch)
    return batch, "Attached **{}** as the nuclear guide for {}.".format(
        record.filename, session.display_name
    ), image_table(batch)


# --------------------------------------------------------------------------- #
# Calibration
# --------------------------------------------------------------------------- #


def _calibrated(batch, message):
    return batch, message, status_html(batch), image_table(batch)


@locked
def on_image_click(batch, evt: gr.SelectData):
    """Collect scale-bar endpoints, converting display to original pixels."""
    session = active(batch)
    if session is None or not session.has_image:
        return batch, "Load an image first."
    point = display_to_original(evt.index, session.display_scale)
    session.scale_bar_points.append(point)
    session.scale_bar_points = session.scale_bar_points[-2:]
    if len(session.scale_bar_points) < 2:
        return batch, "First endpoint at ({:.0f}, {:.0f}). Click the other end.".format(*point)
    (x1, y1), (x2, y2) = session.scale_bar_points
    distance = ((x2 - x1) ** 2 + (y2 - y1) ** 2) ** 0.5
    return batch, (
        "Two endpoints captured: ({:.0f}, {:.0f}) to ({:.0f}, {:.0f}) = "
        "**{:.1f} px**. Enter the bar's length in µm and apply.".format(x1, y1, x2, y2, distance)
    )


@locked
def on_detect_bar(batch):
    """Propose the burned-in bar's endpoints on the active image. Never applied silently."""
    session = active(batch)
    if session is None or not session.has_image:
        return batch, "Load an image first."
    found = detect_annotation(session.image.pixels)
    if not found.found:
        return batch, ("No burned-in scale bar found on this image. Click the bar's two "
                       "ends instead, or use a reference frame.")
    r0, c0, r1, c1 = found.bar_bbox
    y = (r0 + r1 - 1) / 2.0
    # The bar covers c1 - c0 pixel widths, the same length the reference-frame
    # detector reports, so both calibration routes agree for the same ruler.
    session.scale_bar_points = [(float(c0), y), (float(c1), y)]
    return batch, (
        "Detected a **{}** bar **{:.0f} px** long ({}). Enter the length its caption "
        "states (for example 50 µm) and apply."
    ).format(found.colour, found.bar_length_px, found.note)


@locked
def on_clear_points(batch):
    session = active(batch)
    if session is not None:
        session.scale_bar_points = []
    return batch, "Endpoints cleared."


def _apply_calibration(batch, calibration, scope):
    session = active(batch)
    if scope == "This image only":
        if session is None:
            return "Load an image first."
        session.calibration = calibration
        session.results_cache = None
        return "**{}** calibrated: {}".format(session.display_name, calibration.summary())
    batch.calibration = calibration
    batch.calibration_confirmed = True
    changed = batch.apply_shared("calibration")
    return "Whole batch calibrated: **{}** ({} image(s) updated).".format(
        calibration.summary(), changed
    )


@locked
def on_apply_scale_bar(known_length_um, scope, batch):
    session = active(batch)
    if session is None or not session.has_image:
        return _calibrated(batch, "Load an image first.")
    if len(session.scale_bar_points) < 2:
        return _calibrated(batch, "Click both ends of the scale bar first, or detect it.")
    try:
        p1, p2 = session.scale_bar_points
        calibration = calibration_from_scale_bar(p1, p2, known_length_um)
    except Exception as exc:
        return _calibrated(batch, error_text(exc))
    return _calibrated(batch, _apply_calibration(batch, calibration, scope))


@locked
def on_reference_upload(file_path, batch):
    """Load a reference frame and propose its bar length. The researcher confirms it."""
    session = active(batch)
    if session is None or not session.has_image:
        return batch, gr.update(), "Load images first.", None
    if not file_path:
        return batch, gr.update(), "", None
    try:
        reference = load_image(file_path)
    except Exception as exc:
        return batch, gr.update(), error_text(exc), None
    session.reference_image = reference
    detection = detect_scale_bar(reference.pixels)
    session.reference_detection = detection
    preview, _ = make_display(reference)
    if not detection.found:
        return batch, gr.update(), "Loaded **{}**. {}".format(reference.filename, detection.note), preview
    message = (
        "Loaded **{}**. Detected a **{}** bar **{:.0f} px** long. Enter what it "
        "represents in µm, correct the pixel length if the detection is wrong, then apply."
    ).format(reference.filename, detection.colour, detection.length_px)
    return batch, gr.update(value=float(detection.length_px)), message, preview


@locked
def on_apply_reference(bar_length_px, known_length_um, scope, batch):
    session = active(batch)
    if session is None or not session.has_image:
        return _calibrated(batch, "Load images first.")
    try:
        calibration = calibration_from_reference_bar(
            bar_length_px, known_length_um, session.reference_image, session.image,
            session.reference_detection.describe() if session.reference_detection else None,
        )
    except Exception as exc:
        return _calibrated(batch, error_text(exc))
    return _calibrated(batch, _apply_calibration(batch, calibration, scope))


@locked
def on_apply_resolution(sx, sy, scope, batch):
    if batch is None:
        return _calibrated(batch, "Load images first.")
    try:
        calibration = calibration_from_resolution(sx, sy)
    except Exception as exc:
        return _calibrated(batch, error_text(exc))
    return _calibrated(batch, _apply_calibration(batch, calibration, scope))


@locked
def on_set_uncalibrated(scope, batch):
    if batch is None:
        return _calibrated(batch, "")
    message = _apply_calibration(batch, uncalibrated(), scope)
    return _calibrated(
        batch,
        message + "\n\nMeasurements will be reported in **px and px² only**; "
        "no physical columns will be produced.",
    )


def on_calibration_method(choice):
    """Show one calibration panel at a time."""
    return [gr.update(visible=choice == name) for name in CALIBRATION_METHODS]


# --------------------------------------------------------------------------- #
# Layout
# --------------------------------------------------------------------------- #


def build(batch, shared) -> SimpleNamespace:
    c = SimpleNamespace()
    with gr.Tab("Import & calibrate", id="tab_batch") as c.tab:
        with gr.Row(equal_height=False):
            with gr.Column(scale=1, min_width=320):
                gr.HTML(section("Experiment"))
                c.label = gr.Textbox(
                    label="Experiment name", placeholder="shRNA FST, plate 2",
                    info="Recorded in the export and the report.",
                )
                c.upload = gr.File(
                    label="Add images (JPEG, PNG or TIFF)",
                    file_types=["image", ".tif", ".tiff"], file_count="multiple", type="filepath",
                    elem_id="cs-upload",
                )
                gr.HTML(section("Channels · apply to the whole batch"))
                c.channel = gr.Radio(
                    ["grayscale"], value="grayscale", label="Cell channel",
                    info="What gets measured. Never picked for you; recorded in metadata.",
                )
                c.nuclear_channel = gr.Radio(
                    [NO_NUCLEAR], value=NO_NUCLEAR, label="Nuclear guide channel (optional)",
                    info="Helps separate touching cells. Guides segmentation only; never measured.",
                )
                with gr.Accordion("Separate nuclear file for this image", open=False):
                    c.nuclear_file = gr.File(
                        label="Nuclear file (same size as the image)",
                        file_types=["image", ".tif", ".tiff"], type="filepath",
                    )
                c.channel_note = gr.Markdown(elem_classes="cs-note")
            with gr.Column(scale=2, min_width=420):
                c.preview = gr.Image(
                    label="Active image · click two points to measure a scale bar",
                    type="numpy", show_download_button=False, interactive=False,
                )
                with gr.Row():
                    c.prev = gr.Button("◀ Previous", size="sm")
                    c.next = gr.Button("Next ▶", size="sm")
                    c.remove = gr.Button("Remove image", size="sm", variant="stop")
                c.note = gr.Markdown(elem_classes="cs-note")

        gr.HTML(section("Images in this batch"))
        gr.Markdown(
            "Click a row to make it active. **Well**, **Channel** and **Nuclear** can be "
            "edited here to override a single image.", elem_classes="cs-note",
        )
        c.grid = gr.Dataframe(value=empty_frame(*IMAGE_TABLE_COLUMNS), interactive=True, wrap=True)

        gr.HTML(section("Calibration"))
        with gr.Group():
            with gr.Row():
                c.method = gr.Radio(CALIBRATION_METHODS, value=CALIBRATION_METHODS[0],
                                    label="Method", scale=3)
                c.scope = gr.Radio(SCOPE_CHOICES, value="Whole batch", label="Apply to", scale=1,
                                   info="Images in one batch normally share a magnification.")
            c.calibration_note = gr.Markdown(
                "Not calibrated yet: results would be in px and px².", elem_classes="cs-note")
            with gr.Group(visible=True) as c.group_bar:
                gr.Markdown(
                    "**Detect** finds a scale bar drawn on the active image, or click its two "
                    "ends on the image above.", elem_classes="cs-note")
                c.click_note = gr.Markdown(elem_classes="cs-note")
                with gr.Row():
                    c.detect_bar = gr.Button("Detect bar in this image", size="sm")
                    c.clear_points = gr.Button("Clear clicks", size="sm")
                with gr.Row():
                    c.bar_um = gr.Number(value=50.0, label="Bar represents (µm)", precision=4)
                    c.apply_bar = gr.Button("Apply calibration", variant="primary")
            with gr.Group(visible=False) as c.group_reference:
                gr.Markdown(
                    "A separate frame with the ruler, taken at the same magnification. "
                    "Different image sizes are fine.", elem_classes="cs-note")
                with gr.Row():
                    c.reference_upload = gr.File(label="Reference frame",
                                                 file_types=["image", ".tif", ".tiff"], type="filepath")
                    c.reference_preview = gr.Image(label="Detected bar", type="numpy", height=150,
                                                   show_download_button=False, interactive=False)
                c.reference_note = gr.Markdown(elem_classes="cs-note")
                with gr.Row():
                    c.detected_px = gr.Number(label="Bar length (px)", precision=2)
                    c.reference_um = gr.Number(value=50.0, label="Represents (µm)", precision=4)
                    c.apply_reference = gr.Button("Apply calibration", variant="primary")
            with gr.Group(visible=False) as c.group_resolution:
                gr.Markdown("Anisotropic pixels are supported.", elem_classes="cs-note")
                with gr.Row():
                    c.res_x = gr.Number(value=1.0, label="µm per pixel (X)", precision=6)
                    c.res_y = gr.Number(value=1.0, label="µm per pixel (Y)", precision=6)
                    c.apply_resolution = gr.Button("Apply calibration", variant="primary")
            with gr.Group(visible=False) as c.group_pixels:
                gr.Markdown(
                    "Analysis proceeds in pixels and no physical columns are produced. "
                    "CellScope will not invent a micron value.", elem_classes="cs-note")
                c.set_pixels = gr.Button("Confirm pixel units", variant="primary")
    return c


def wire(c, batch, shared) -> None:
    active_outputs = [c.channel, c.nuclear_channel, c.preview, c.channel_note]
    views = [batch, c.grid, shared.segment_nav, shared.review_strip, c.note, shared.status]
    nav_outputs = views + active_outputs
    shared.nav_outputs = nav_outputs

    c.method.change(on_calibration_method, c.method,
                    [c.group_bar, c.group_reference, c.group_resolution, c.group_pixels],
                    show_progress="hidden")
    c.label.change(on_batch_label, [c.label, batch], [batch, shared.status], show_progress="hidden")
    # .upload, not .change: change also fires when the component's value is reset.
    c.upload.upload(on_upload, [c.upload, batch], nav_outputs, show_progress_on=[c.grid])
    c.grid.select(on_select_image, [batch], nav_outputs, show_progress="hidden")
    c.grid.input(on_table_edited, [c.grid, batch], [batch, c.grid, shared.status],
                 show_progress="hidden")
    c.remove.click(on_remove_image, [batch], nav_outputs, show_progress_on=[c.grid])
    c.prev.click(on_prev, [batch], nav_outputs, show_progress="hidden")
    c.next.click(on_next, [batch], nav_outputs, show_progress="hidden")
    c.channel.input(on_channel_change, [c.channel, batch], [batch, c.grid, shared.segment_nav],
                    show_progress="hidden")
    c.nuclear_channel.input(on_nuclear_change, [c.nuclear_channel, batch],
                            [batch, c.grid, shared.segment_nav], show_progress="hidden")
    c.nuclear_file.change(on_nuclear_file, [c.nuclear_file, batch],
                          [batch, c.channel_note, c.grid], show_progress_on=[c.channel_note])

    c.preview.select(on_image_click, [batch], [batch, c.click_note], show_progress="hidden")
    c.detect_bar.click(on_detect_bar, [batch], [batch, c.click_note], show_progress="hidden")
    c.clear_points.click(on_clear_points, [batch], [batch, c.click_note], show_progress="hidden")
    calibration_outputs = [batch, c.calibration_note, shared.status, c.grid]
    c.apply_bar.click(on_apply_scale_bar, [c.bar_um, c.scope, batch], calibration_outputs,
                      show_progress_on=[c.calibration_note])
    c.reference_upload.change(on_reference_upload, [c.reference_upload, batch],
                              [batch, c.detected_px, c.reference_note, c.reference_preview],
                              show_progress_on=[c.reference_preview])
    c.apply_reference.click(on_apply_reference, [c.detected_px, c.reference_um, c.scope, batch],
                            calibration_outputs, show_progress_on=[c.calibration_note])
    c.apply_resolution.click(on_apply_resolution, [c.res_x, c.res_y, c.scope, batch],
                             calibration_outputs, show_progress_on=[c.calibration_note])
    c.set_pixels.click(on_set_uncalibrated, [c.scope, batch], calibration_outputs,
                       show_progress_on=[c.calibration_note])
    c.tab.select(refresh_batch_view, [batch], nav_outputs, show_progress="hidden")
