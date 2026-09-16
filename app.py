"""CellScope -- research-grade fluorescence cell morphometry.

Gradio wiring only. Every scientific calculation lives in ``src/`` and is
unit-tested without a browser; this file moves values between widgets and those
functions and does no arithmetic of its own beyond formatting.

The unit of work is a **batch**: the images of one experiment, across its wells.
Per-session data lives in a ``gr.State`` holding a :class:`BatchSession`, never a
module global, so two people using the same server cannot see each other's work.
"""

from __future__ import annotations

import os
import tempfile
import traceback

import gradio as gr
import pandas as pd

from src import __version__
from src.batch import (
    CalibrationMismatch,
    batch_summary,
    calibration_consistency,
    headline_counts,
    per_image_summary,
    pooled_cells,
    pooled_clusters,
)
from src.calibration import (
    calibration_from_reference_bar,
    calibration_from_resolution,
    calibration_from_scale_bar,
    display_to_original,
    uncalibrated,
)
from src.export import export_batch
from src.image_io import (
    available_channels,
    describe_channel_signal,
    load_image,
    make_display,
)
from src.pipeline import (
    compute_results,
    iter_batch_segmentation,
    nuclear_array_for,
    segment_session,
)
from src.qc import (
    exclude_border_touching,
    exclude_by_area,
    exclude_cluster,
    exclude_object,
    qc_log_dataframe,
    restore_object,
)
from src.review import checkpoint, click_position, display_view, signature, validate_review
from src.scalebar import detect_scale_bar
from src.segmentation import cellpose_available, cellpose_version
from src.types import (
    AnalysisSession,
    BatchSession,
    ClusterParams,
    PreprocessParams,
    SegmentationParams,
    SplitParams,
)
from src.visualize import build_all_figures, make_overlay
from ui_theme import CSS, THEME_JS, build_theme, callout, empty_state, legend_html, section

OUTPUT_DIR = os.path.join(tempfile.gettempdir(), "cellscope_exports")

DASH = "—"
_NO_NUCLEAR = "none"

CALIBRATION_METHODS = (
    "Scale bar in image",
    "Scale bar on a reference frame",
    "Enter resolution",
    "Work in pixels",
)


# --------------------------------------------------------------------------- #
# Formatting helpers
# --------------------------------------------------------------------------- #


def _empty(*columns):
    return pd.DataFrame(columns=list(columns))


def _error(exc: Exception) -> str:
    return "**{}**: {}".format(type(exc).__name__, exc)


def _active(batch: BatchSession | None) -> AnalysisSession | None:
    return batch.active if batch is not None else None


STATUS_FIELDS = ("Batch", "Images", "Reviewed", "Calibration", "Segmented", "Cells")


def status_fields(batch: BatchSession | None) -> dict:
    """The persistent header §21 requires, at batch level."""
    if batch is None or batch.is_empty:
        return dict.fromkeys(STATUS_FIELDS, DASH)

    counts = headline_counts(batch)
    return {
        "Batch": batch.label or "(unnamed)",
        "Images": str(len(batch.images)),
        "Reviewed": "{} / {}".format(counts["n_reviewed"], counts["n_images"]),
        "Calibration": batch.calibration.summary(),
        "Segmented": "{} / {}".format(counts["n_segmented"], counts["n_images"]),
        "Cells": str(counts["n_cells"]) if counts["consistent"] else "mixed units",
    }


def status_html(batch: BatchSession | None) -> str:
    """The status strip as labelled cells rather than a run-on sentence."""
    fields = status_fields(batch)
    stale = fields["Cells"] == "mixed units"
    cells = "".join(
        '<div class="cs-stat{}"><span class="cs-k">{}</span>'
        '<span class="cs-v">{}</span></div>'.format(
            " cs-stale" if (key == "Cells" and stale) else "", key, fields[key]
        )
        for key in STATUS_FIELDS
    )
    return '<div class="cs-status">{}</div>'.format(cells)


def nav_html(batch: BatchSession | None) -> str:
    """Which image the current tab is acting on."""
    session = _active(batch)
    if session is None:
        return '<div class="cs-nav"><span class="cs-pos">No images loaded</span></div>'

    badges = []
    if session.error:
        badges.append('<span class="cs-badge warn">failed</span>')
    elif session.has_segmentation:
        results = compute_results(session)
        badges.append(
            '<span class="cs-badge">{} cells</span>'.format(results.counts["cells_accepted"])
        )
    else:
        badges.append('<span class="cs-badge">not segmented</span>')
    badges.append(
        '<span class="cs-badge ok">reviewed</span>'
        if session.reviewed
        else '<span class="cs-badge">unreviewed</span>'
    )
    if session.well:
        badges.append('<span class="cs-badge">well {}</span>'.format(session.well))
    scale = session.engine_info.get("analysis_scale")
    if scale is not None and scale < 1.0:
        badges.append(
            '<span class="cs-badge warn">segmented at {:.2f}x</span>'.format(scale)
        )

    return (
        '<div class="cs-nav"><span class="cs-pos">{} / {}</span>'
        '<span class="cs-name">{}</span><span class="cs-sep">|</span>{}</div>'
    ).format(
        batch.active_index + 1, len(batch.images), session.display_name, "".join(badges)
    )


def strip_html(batch: BatchSession | None) -> str:
    """Every image's review state at a glance."""
    if batch is None or batch.is_empty:
        return ""
    tiles = []
    for index, session in enumerate(batch.images):
        classes = ["cs-tile"]
        if session.reviewed:
            classes.append("done")
        if session.error:
            classes.append("err")
        if index == batch.active_index:
            classes.append("active")
        tiles.append(
            '<span class="{}"><span class="n">{}</span>{}{}</span>'.format(
                " ".join(classes), index + 1, session.display_name,
                " ✓" if session.reviewed else "",
            )
        )
    return '<div class="cs-strip">{}</div>'.format("".join(tiles))


IMAGE_TABLE_COLUMNS = ["#", "File", "Well", "Channel", "Nuclear", "Status", "Cells", "Reviewed"]


def image_table(batch: BatchSession | None) -> pd.DataFrame:
    """One row per image. ``Well`` is the only column meant to be edited."""
    if batch is None or batch.is_empty:
        return _empty(*IMAGE_TABLE_COLUMNS)

    rows = []
    for index, session in enumerate(batch.images):
        if session.error:
            status, cells = "failed", DASH
        elif session.has_segmentation:
            status = "segmented"
            cells = str(compute_results(session).counts["cells_accepted"])
        else:
            status, cells = "pending", DASH
        nuclear = (
            session.nuclear_image.filename
            if session.nuclear_image
            else (session.segmentation_params.nuclear_channel or DASH)
        )
        rows.append(
            [
                index + 1,
                session.display_name,
                session.well,
                session.channel,
                nuclear,
                status,
                cells,
                "yes" if session.reviewed else "no",
            ]
        )
    return pd.DataFrame(rows, columns=IMAGE_TABLE_COLUMNS)


# --------------------------------------------------------------------------- #
# Tab 1 -- Batch
# --------------------------------------------------------------------------- #


def _blank_views(batch, message=""):
    """Everything the batch tab shows, recomputed."""
    return (
        batch,
        image_table(batch),
        nav_html(batch),
        strip_html(batch),
        message,
        status_html(batch),
    )


#: Set on a batch when the title has been clicked once and is awaiting
#: confirmation. Held on the session, not a module global, so two users cannot
#: arm each other's reset.
_RESET_ARMED = "_reset_armed"


def on_home(batch):
    """Clear everything and return to the Batch tab.

    Guarded, because losing a twenty-minute segmentation run to a stray click on
    the logo would be worse than having no reset at all. The first click on a
    batch with work in it says what would be lost; the second confirms. An empty
    batch resets immediately -- there is nothing to protect.
    """
    def views(target, note, tabs_update):
        """The 11 values this callback is wired to, in order."""
        return (
            target, note, tabs_update,
            image_table(target), nav_html(target), strip_html(target),
            status_html(target),
            *_active_controls(target),
        )

    if batch is not None and batch.images and not getattr(batch, _RESET_ARMED, False):
        setattr(batch, _RESET_ARMED, True)
        counts = headline_counts(batch)
        return views(
            batch,
            "**Click CellScope again to discard {} image(s) and {} cell(s).** "
            "Anything not exported will be lost.".format(
                counts["n_images"], counts["n_cells"]
            ),
            gr.Tabs(),          # stay put until confirmed
        )

    return views(BatchSession(), "New batch started.", gr.Tabs(selected="tab_batch"))


def on_batch_label(label, batch):
    if batch is not None:
        setattr(batch, _RESET_ARMED, False)
        batch.label = str(label or "").strip()
    return batch, status_html(batch)


def on_upload(file_paths, batch):
    """Add images to the batch, seeded with the shared defaults."""
    batch = batch or BatchSession()
    if not file_paths:
        return (*_blank_views(batch), gr.update(), gr.update(), None, "")

    # Content hashes, not filenames: both of a pair of condition folders may
    # number their fields 1.jpg..10.jpg, and those are genuinely different
    # images that must both be kept.
    seen = {s.image.sha256 for s in batch.images if s.image is not None}

    added, failed, duplicates = 0, [], []
    for path in file_paths:
        try:
            record = load_image(path)
        except Exception as exc:
            failed.append("{}: {}".format(os.path.basename(str(path)), exc))
            continue
        if record.sha256 in seen:
            # Adding the same field twice would count its cells twice in the
            # pooled batch summary, silently inflating n_cells.
            duplicates.append(record.filename)
            continue
        seen.add(record.sha256)

        session = AnalysisSession()
        session.image = record
        display, scale = make_display(record)
        session.display_rgb = display
        session.display_scale = scale
        session.channel = available_channels(record)[0]
        batch.add(session)
        batch.apply_channels(session)     # inherit the batch's channel choices
        added += 1

    message = "Added **{}** image(s). Batch now holds {}.".format(added, len(batch.images))
    if duplicates:
        message += "\n\n**Skipped {} already in this batch** ({}). The same field " \
                   "counted twice would double its cells in the pooled result.".format(
                       len(duplicates),
                       ", ".join(duplicates[:5]) + ("..." if len(duplicates) > 5 else ""),
                   )
    if failed:
        message += "\n\nCould not load: " + "; ".join(failed)

    views = _blank_views(batch, message)
    return (*views, *_active_controls(batch))


def _active_controls(batch):
    """Widget updates that follow whichever image is active."""
    session = _active(batch)
    if session is None or session.image is None:
        return (
            gr.update(choices=["grayscale"], value="grayscale"),
            gr.update(choices=[_NO_NUCLEAR], value=_NO_NUCLEAR),
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
        "**{}** ({} x {} px).  \nPer-channel signal:  \n{}\n\n"
        "*JPEG chroma subsampling leaks a little into the other channels, so a "
        "non-zero mean does not mean a channel carries your stain.*"
    ).format(session.image.filename, session.image.width, session.image.height, report)
    return (
        gr.update(choices=choices, value=session.channel),
        gr.update(
            choices=[_NO_NUCLEAR] + choices,
            value=session.segmentation_params.nuclear_channel or _NO_NUCLEAR,
        ),
        _display_base(session),
        note,
    )


def on_select_image(batch, evt: gr.SelectData):
    """Clicking a table row makes that image active."""
    if batch is None or batch.is_empty:
        return (*_blank_views(batch), *_active_controls(batch))
    row = evt.index[0] if isinstance(evt.index, (list, tuple)) else evt.index
    batch.active_index = max(0, min(int(row), len(batch.images) - 1))
    views = _blank_views(batch, "")
    return (*views, *_active_controls(batch))


def on_table_edited(table, batch):
    """Write the editable columns back onto the images.

    Only **Well**, **Channel** and **Nuclear** are read back; the rest is derived
    and regenerated on the next refresh, so a stray edit elsewhere cannot corrupt
    state. Channel edits here are the per-image override — the radios above set
    the whole batch.
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

        # A filename in Nuclear means a separate file is attached; leave it be.
        if (
            "Nuclear" in frame.columns
            and session.image is not None
            and session.nuclear_image is None
        ):
            value = str(row["Nuclear"]).strip().lower()
            choice = None if value in ("", DASH.lower(), "none", "nan") else value
            known = choice is None or choice in available_channels(session.image)
            if known and session.segmentation_params.nuclear_channel != choice:
                session.segmentation_params = SegmentationParams(
                    **{**session.segmentation_params.describe(),
                       "nuclear_channel": choice}
                )
                session.results_cache = None

    return batch, image_table(batch), status_html(batch)


def on_remove_image(batch):
    if batch is None or batch.is_empty:
        return (*_blank_views(batch, "Nothing to remove."), *_active_controls(batch))
    removed = batch.remove(batch.active_index)
    name = removed.display_name if removed else "?"
    views = _blank_views(batch, "Removed **{}** from the batch.".format(name))
    return (*views, *_active_controls(batch))


def on_prev(batch):
    if batch is not None and not batch.is_empty:
        batch.active_index = (batch.active_index - 1) % len(batch.images)
    return (*_blank_views(batch), *_active_controls(batch))


def on_next(batch):
    if batch is not None and not batch.is_empty:
        batch.active_index = (batch.active_index + 1) % len(batch.images)
    return (*_blank_views(batch), *_active_controls(batch))


# --------------------------------------------------------------------------- #
# Channels
# --------------------------------------------------------------------------- #


def _set_cell_channel(batch, channel) -> int:
    """Apply a cell channel to every image in the batch. Returns how many changed.

    An image that lacks the channel keeps its own -- a greyscale TIFF in an RGB
    batch must not be handed a red plane that does not exist.
    """
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
    """Apply a nuclear channel to every image in the batch."""
    value = None if choice in (None, "", _NO_NUCLEAR) else choice
    changed = 0
    for session in batch.images:
        if (
            session.image is not None
            and value is not None
            and value not in available_channels(session.image)
        ):
            continue
        if session.segmentation_params.nuclear_channel != value:
            session.segmentation_params = SegmentationParams(
                **{**session.segmentation_params.describe(), "nuclear_channel": value}
            )
            session.results_cache = None
            changed += 1
    return changed


def on_channel_change(channel, batch):
    """The cell channel is a batch setting: choosing it once covers every image.

    It used to apply only to the active image, with a separate *Apply to all*
    button to propagate. That put the burden of remembering a second step on the
    researcher, and quietly segmented the rest of the batch on whatever they
    happened to have -- usually greyscale. Per-image overrides still exist, in
    the image table.
    """
    if batch is None:
        return batch, image_table(batch), nav_html(batch)
    batch.channel = channel
    _set_cell_channel(batch, channel)
    return batch, image_table(batch), nav_html(batch)


def on_nuclear_change(choice, batch):
    """The nuclear channel is a batch setting, for the same reason."""
    if batch is None:
        return batch, image_table(batch), nav_html(batch)
    batch.nuclear_channel = choice
    _set_nuclear_channel(batch, choice)
    return batch, image_table(batch), nav_html(batch)


def on_nuclear_file(file_path, batch):
    """Attach a separate nuclear file to the active image.

    Validated here, at attach time. A mismatched file caught mid-batch would
    waste a long run and, worse, could be missed.
    """
    session = _active(batch)
    if session is None or not session.has_image:
        return batch, "Load an image first.", image_table(batch)
    if not file_path:
        session.nuclear_image = None
        return batch, "Nuclear file cleared.", image_table(batch)

    try:
        record = load_image(file_path)
    except Exception as exc:
        return batch, _error(exc), image_table(batch)

    session.nuclear_image = record
    try:
        nuclear_array_for(session)
    except ValueError as exc:
        session.nuclear_image = None
        return batch, _error(exc), image_table(batch)

    return batch, "Attached **{}** as the nuclear guide for {}.".format(
        record.filename, session.display_name
    ), image_table(batch)


# --------------------------------------------------------------------------- #
# Calibration (batch default, with per-image override)
# --------------------------------------------------------------------------- #


def _calibrated(batch, message):
    return batch, message, status_html(batch), image_table(batch)


def on_image_click(batch, evt: gr.SelectData):
    """Collect scale-bar endpoints, converting display -> original pixels.

    The conversion happens here and nowhere else; a silent error in it would
    rescale every physical measurement in the export.
    """
    session = _active(batch)
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


def on_clear_points(batch):
    session = _active(batch)
    if session is not None:
        session.scale_bar_points = []
    return batch, "Endpoints cleared."


def _apply_calibration(batch, calibration, scope):
    """Set the calibration on the batch, on the active image, or both."""
    session = _active(batch)
    if scope == "This image only":
        if session is None:
            return "Load an image first."
        session.calibration = calibration
        session.results_cache = None
        return "**{}** calibrated: {}".format(session.display_name, calibration.summary())

    batch.calibration = calibration
    changed = batch.apply_shared("calibration")
    return "Whole batch calibrated: **{}** ({} image(s) updated).".format(
        calibration.summary(), changed
    )


def on_apply_scale_bar(known_length_um, scope, batch):
    session = _active(batch)
    if session is None or not session.has_image:
        return _calibrated(batch, "Load an image first.")
    if len(session.scale_bar_points) < 2:
        return _calibrated(batch, "Click both ends of the scale bar first.")
    try:
        p1, p2 = session.scale_bar_points
        calibration = calibration_from_scale_bar(p1, p2, known_length_um)
    except Exception as exc:
        return _calibrated(batch, _error(exc))
    return _calibrated(batch, _apply_calibration(batch, calibration, scope))


def on_reference_upload(file_path, batch):
    """Load a reference frame and look for a burned-in scale bar.

    Detection only proposes a length. The researcher confirms or corrects it
    before it becomes a calibration.
    """
    session = _active(batch)
    if session is None or not session.has_image:
        return batch, gr.update(), "Load images first.", None
    if not file_path:
        return batch, gr.update(), "", None

    try:
        reference = load_image(file_path)
    except Exception as exc:
        return batch, gr.update(), _error(exc), None

    session.reference_image = reference
    detection = detect_scale_bar(reference.pixels)
    session.reference_detection = detection
    preview, _ = make_display(reference)

    if not detection.found:
        return batch, gr.update(), "Loaded **{}**. {}".format(
            reference.filename, detection.note
        ), preview

    message = (
        "Loaded **{}**. Detected a **{}** bar **{:.0f} px** long. Enter what it "
        "represents in µm, correct the pixel length if the detection is wrong, "
        "then apply."
    ).format(reference.filename, detection.colour, detection.length_px)
    return batch, gr.update(value=float(detection.length_px)), message, preview


def on_apply_reference(bar_length_px, known_length_um, scope, batch):
    session = _active(batch)
    if session is None or not session.has_image:
        return _calibrated(batch, "Load images first.")
    try:
        calibration = calibration_from_reference_bar(
            bar_length_px,
            known_length_um,
            session.reference_image,
            session.image,
            session.reference_detection.describe() if session.reference_detection else None,
        )
    except Exception as exc:
        return _calibrated(batch, _error(exc))
    return _calibrated(batch, _apply_calibration(batch, calibration, scope))


def on_apply_resolution(sx, sy, scope, batch):
    if batch is None:
        return _calibrated(batch, "Load images first.")
    try:
        calibration = calibration_from_resolution(sx, sy)
    except Exception as exc:
        return _calibrated(batch, _error(exc))
    return _calibrated(batch, _apply_calibration(batch, calibration, scope))


def on_set_uncalibrated(scope, batch):
    if batch is None:
        return _calibrated(batch, "")
    message = _apply_calibration(batch, uncalibrated(), scope)
    return _calibrated(
        batch,
        message + "\n\nMeasurements will be reported in **px and px² only** -- "
        "no physical columns will be produced.",
    )


# --------------------------------------------------------------------------- #
# Tab 2 -- Segment
# --------------------------------------------------------------------------- #


def _collect_params(
    model, diameter, cellprob, flow, min_area, use_gpu,
    background_subtract, background_radius, gaussian_sigma, normalize,
    split_enabled, saddle_depth, solidity_max, min_fragment_frac, contact_distance,
    nuclear_channel, analysis_scale,
):
    """Build the parameter objects from the widget values, once."""
    nuclear = None if nuclear_channel in (None, "", _NO_NUCLEAR) else nuclear_channel
    return (
        SegmentationParams(
            model=model,
            nuclear_channel=nuclear,
            diameter=float(diameter) if diameter and float(diameter) > 0 else None,
            cellprob_threshold=float(cellprob),
            flow_threshold=float(flow),
            min_object_area=int(min_area),
            use_gpu=bool(use_gpu),
            analysis_scale=float(analysis_scale),
        ),
        PreprocessParams(
            background_subtract=bool(background_subtract),
            background_radius_px=float(background_radius),
            gaussian_sigma=float(gaussian_sigma),
            normalize_contrast=bool(normalize),
        ),
        SplitParams(
            enabled=bool(split_enabled),
            min_saddle_depth_frac=float(saddle_depth),
            solidity_max=float(solidity_max),
            min_fragment_area_frac=float(min_fragment_frac),
            min_fragment_area=int(min_area),
        ),
        ClusterParams(contact_distance_px=int(contact_distance)),
    )


def _push_params(batch, params, to_all):
    """Apply parameters to the batch, or only to the active image."""
    segmentation, preprocess, split, cluster = params
    targets = batch.images if to_all else [batch.active]
    if to_all:
        batch.segmentation_params = segmentation
        batch.preprocess_params = preprocess
        batch.split_params = split
        batch.cluster_params = cluster
    for session in targets:
        if session is None:
            continue
        # The nuclear channel is chosen per image, so a batch-wide parameter
        # push must not overwrite an image's own choice with the widget's.
        own_nuclear = session.segmentation_params.nuclear_channel
        session.segmentation_params = SegmentationParams(
            **{**segmentation.describe(),
               "nuclear_channel": own_nuclear if to_all else segmentation.nuclear_channel}
        )
        session.preprocess_params = preprocess
        session.split_params = split
        session.cluster_params = cluster
        session.results_cache = None


def _display_base(session):
    """The RGB the overlay is drawn on, built on demand if it is missing.

    Upload sets this, but a batch run must not die partway through because one
    image arrived by some other route without a preview.
    """
    if session.display_rgb is not None:
        return session.display_rgb
    if session.image is None:
        return None
    display, scale = make_display(session.image)
    session.display_rgb = display
    session.display_scale = scale
    return display


def _segment_views(batch, message):
    session = _active(batch)
    overlay = None
    if session is not None and session.has_segmentation:
        base = _display_base(session)
        if base is not None:
            results = compute_results(session)
            overlay = make_overlay(
                base, results.qc_labels, results.objects, results.clusters
            )
    return (
        batch, overlay, message, image_table(batch), nav_html(batch),
        strip_html(batch), status_html(batch),
    )


def _describe_run(session):
    results = compute_results(session)
    unresolved = [o for o in results.objects if not o.is_resolved]
    timing = session.engine_info.get("timings_seconds", {})
    note = ""
    if timing:
        note += "\n\nProcessing: **{:.1f}s** (segmentation {:.1f}s; splitting {:.1f}s).".format(
            timing["total"], timing["segmentation"], timing["splitting"])
    if session.engine_info.get("device"):
        note += " Device: **{}**; tile batch: **{}**.".format(
            session.engine_info["device"], session.engine_info.get("tile_batch_size", "default"))
    if session.engine_info.get("cellpose_error"):
        note += "\n\n**Cellpose failed and the fallback ran instead:** `{}`".format(
            session.engine_info["cellpose_error"]
        )
    if unresolved:
        note += (
            "\n\n**{} object(s) could not be separated into cells** and are recorded "
            "as `unresolved_cluster`. They contribute group-level measurements only, "
            "and their cell counts are reported as NA.".format(len(unresolved))
        )
    return "Segmented with **{}**: {} object(s), {} resolved cell(s), {} cluster(s).{}".format(
        session.engine_info.get("engine", "?"),
        len(session.objects),
        results.counts["cells_accepted"],
        len(results.clusters),
        note,
    )


def _tally(sessions, attribute) -> str:
    """``red x3, grayscale x1`` — how a setting is distributed across a batch."""
    counts: dict[str, int] = {}
    for session in sessions:
        counts[attribute(session)] = counts.get(attribute(session), 0) + 1
    return ", ".join(
        "{} x{}".format(name, n)
        for name, n in sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))
    )


def _channel_divergence_warning(batch) -> str:
    """Report when the images in one run did not all use the same channels.

    Channels are per image by design, and **Apply to all images** is how they
    propagate -- but that makes a specific mistake easy: set the channel on the
    active image, press *Run on all*, and every other image quietly runs on
    whatever it had. The cell channel is what gets measured, and the nuclear
    channel is the largest accuracy lever there is (9 objects versus 18 on the
    reference field), so a divergence is stated rather than left to be noticed.

    Deliberately factual, not accusatory: a mixed batch can be entirely correct
    -- a DAPI frame legitimately uses blue where its merge uses red.
    """
    segmented = [s for s in batch.images if s.has_segmentation]
    if len(segmented) < 2:
        return ""

    lines = []
    if len({s.segmented_channel for s in segmented}) > 1:
        lines.append(
            "cell channel — {}".format(_tally(segmented, lambda s: str(s.segmented_channel)))
        )
    # The nuclear guide has two routes -- a channel of the same image, or an
    # attached file -- and both are per image, so both can diverge.
    def nuclear_of(session):
        if session.nuclear_image is not None:
            return session.nuclear_image.filename
        return session.segmentation_params.nuclear_channel or "none"

    if len({nuclear_of(s) for s in segmented}) > 1:
        lines.append("nuclear guide — {}".format(_tally(segmented, nuclear_of)))
    if not lines:
        return ""

    return (
        "\n\n**These {} images were not all segmented the same way:**\n"
        "{}\n\nThat is legitimate if the fields genuinely differ. If they should "
        "match, set the channels on one image, press **Apply to all images** on "
        "the Batch tab, and run again."
    ).format(len(segmented), "\n".join("- " + line for line in lines))


def on_segment_one(*args, progress=gr.Progress()):
    """Segment the active image only."""
    *widget_values, engine_choice, batch = args
    session = _active(batch)
    if session is None or not session.has_image:
        return _segment_views(batch, "Load images first.")

    _push_params(batch, _collect_params(*widget_values), to_all=False)
    try:
        progress(0.2, desc="Segmenting {}".format(session.display_name))
        segment_session(session, engine=engine_choice)
        progress(0.8, desc="Measuring")
        message = _describe_run(session)
    except Exception as exc:
        traceback.print_exc()
        session.error = "{}: {}".format(type(exc).__name__, exc)
        return _segment_views(batch, _error(exc))
    return _segment_views(batch, message)


def _run_summary(report, batch, final: bool) -> str:
    """The note under the segmentation preview, mid-run or at the end."""
    verb = "Segmented" if final else "Segmenting:"
    message = "{} **{} of {}** image(s).".format(
        verb, report["succeeded"], report["total"]
    )
    if report["skipped"]:
        message += "  Skipped {} already segmented.".format(report["skipped"])
    if report["failed"]:
        message += "\n\n**Failed:** " + "; ".join(
            "{} ({})".format(name, error) for name, error in report["failed"]
        )
    if final:
        session = _active(batch)
        if session is not None and session.has_segmentation:
            message += "\n\n" + _describe_run(session)
        message += _channel_divergence_warning(batch)
    return message


def on_segment_all(*args, progress=gr.Progress()):
    """Segment every image in the batch, streaming each result as it lands.

    A generator: Gradio pushes every yield to the widgets, so the table, overlay
    and review strip fill in one image at a time and the Review tab is usable
    from the moment the first image finishes. The total GPU time is unchanged --
    inference is 95% of it and does not batch -- but the wait now overlaps with
    the reviewing that has to happen anyway.
    """
    *widget_values, engine_choice, skip_done, batch = args
    if batch is None or batch.is_empty:
        yield _segment_views(batch, "Load images first.")
        return

    _push_params(batch, _collect_params(*widget_values), to_all=True)

    report = None
    for position, total, session, report in iter_batch_segmentation(
        batch, engine=engine_choice, only_missing=bool(skip_done)
    ):
        progress(
            (position + 1) / max(total, 1),
            desc="Segmented {} ({}/{})".format(session.display_name, position + 1, total),
        )
        # Follow the run so the preview always shows the image just finished.
        batch.active_index = batch.images.index(session)
        from professional_ui import autosave
        try:
            autosave(batch)
        except OSError as exc:
            gr.Warning("Autosave failed: " + str(exc))
        yield _segment_views(batch, _run_summary(report, batch, final=False))

    if report is None:
        yield _segment_views(
            batch,
            "Every image is already segmented. Untick *skip already-segmented* "
            "to re-run them.",
        )
        return

    yield _segment_views(batch, _run_summary(report, batch, final=True))


# --------------------------------------------------------------------------- #
# Tab 3 -- Review
# --------------------------------------------------------------------------- #


def _review_views(batch, show_ids=True, show_clusters=False, message=""):
    """Recompute the active image from its QC-approved mask and redraw."""
    session = _active(batch)
    if session is None or not session.has_segmentation:
        # Show the image itself rather than an empty frame. A field you have not
        # segmented yet -- or one that failed -- is still worth looking at, and a
        # blank panel reads as "the image did not load".
        note = message
        if not note:
            if session is None:
                note = "Add images on the Batch tab."
            elif session.error:
                note = "**This image failed to segment:** {}".format(session.error)
            else:
                note = "Not segmented yet. Showing the original image."
        return (
            batch,
            _display_base(session) if session is not None else None,
            _empty("object_id"),
            note,
            nav_html(batch), strip_html(batch), image_table(batch), status_html(batch),
        )

    validate_review(session)
    results = compute_results(session)
    overlay = make_overlay(
        _display_base(session),
        session.object_labels,
        results.objects,
        results.clusters,
        show_cell_ids=show_ids,
        show_cluster_ids=show_clusters,
        excluded_ids=session.qc.excluded_ids,
        alpha=session.review_alpha,
    )
    if session.review_original:
        overlay = _display_base(session).copy()
    elif session.selected_object:
        import numpy as np
        from skimage.segmentation import find_boundaries

        from src.visualize import _resize_labels
        small = _resize_labels(session.object_labels, overlay.shape[:2])
        overlay = overlay.copy()
        overlay[find_boundaries(small == session.selected_object, mode="inner")] = (255, 255, 255)
    if session.correction_points:
        from PIL import Image, ImageDraw
        canvas = Image.fromarray(overlay)
        draw = ImageDraw.Draw(canvas)
        points = [(x * overlay.shape[1] / session.object_labels.shape[1],
                   y * overlay.shape[0] / session.object_labels.shape[0])
                  for x, y in session.correction_points]
        if len(points) > 1:
            draw.line(points, fill="yellow", width=2)
        for index, (x, y) in enumerate(points, 1):
            draw.ellipse((x-3, y-3, x+3, y+3), fill="yellow")
            draw.text((x+4, y+4), str(index), fill="yellow", stroke_width=1, stroke_fill="black")
        import numpy as np
        overlay = np.asarray(canvas)
    overlay = display_view(session, overlay)
    counts = results.counts
    summary = "**{} included · {} excluded**. Gray outlines = excluded; click any cell to toggle.".format(
        counts["objects_accepted"], counts["objects_excluded"]
    )
    message = summary + ("\n\n" + message if message else "")
    return (
        batch, overlay, qc_log_dataframe(session.qc), message,
        nav_html(batch), strip_html(batch), image_table(batch), status_html(batch),
    )


def on_review_click(show_ids, show_clusters, batch, evt: gr.SelectData):
    """Hit-test the exact display label grid, including excluded cells."""
    from src.visualize import _resize_labels

    session = _active(batch)
    if session is None or not session.has_segmentation:
        return _review_views(batch, show_ids, show_clusters)
    labels = _resize_labels(session.object_labels, _display_base(session).shape[:2])
    point = click_position(session, evt.index, labels.shape)
    if point is None:
        return _review_views(batch, show_ids, show_clusters)
    x, y = point
    if session.review_mode == "Collect correction points":
        ox = min(session.object_labels.shape[1]-1, int((x + .5) * session.object_labels.shape[1] / labels.shape[1]))
        oy = min(session.object_labels.shape[0]-1, int((y + .5) * session.object_labels.shape[0] / labels.shape[0]))
        session.correction_points.append((ox, oy))
        return _review_views(batch, show_ids, show_clusters, f"Point {len(session.correction_points)}: ({ox}, {oy})")
    if not (0 <= y < labels.shape[0] and 0 <= x < labels.shape[1]):
        return _review_views(batch, show_ids, show_clusters)
    object_id = int(labels[y, x])
    if object_id == 0 or session.object_by_id(object_id) is None:
        return _review_views(batch, show_ids, show_clusters, "Click inside a segmented cell.")
    session.selected_object = object_id
    if session.review_mode == "Inspect":
        return _review_views(batch, show_ids, show_clusters, f"Selected cell {object_id}.")
    checkpoint(session)
    if session.qc.is_included(object_id):
        exclude_object(session.qc, object_id, "deselected during review")
        action = "Excluded"
    else:
        restore_object(session.qc, object_id, "selected during review")
        action = "Restored"
    session.reviewed = False
    return _review_views(batch, show_ids, show_clusters, f"{action} cell {object_id}.")


def on_toggle_labels(show_ids, show_clusters, batch):
    return _review_views(batch, show_ids, show_clusters)


def on_mark_reviewed(note, show_ids, show_clusters, batch):
    session = _active(batch)
    if session is None:
        return _review_views(batch, show_ids, show_clusters)
    if not session.has_segmentation:
        return _review_views(batch, show_ids, show_clusters, "Segment this image before marking it reviewed.")
    session.reviewed = True
    session.reviewed_signature = signature(session)
    session.review_note = str(note or "").strip()
    return _review_views(
        batch, show_ids, show_clusters,
        "Marked **{}** reviewed. {} of {} done.".format(
            session.display_name, batch.n_reviewed, len(batch.images)
        ),
    )


def on_unmark_reviewed(show_ids, show_clusters, batch):
    session = _active(batch)
    if session is not None:
        session.reviewed = False
    return _review_views(batch, show_ids, show_clusters, "Marked unreviewed.")


def _parse_ids(text, session):
    """Parse whitespace/comma-separated IDs, rejecting anything that is not a
    real object. Excluding a cell that does not exist would put a phantom entry
    in the QC log, which is meant to be an exact record of what was done."""
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


def on_exclude(object_ids, reason, show_ids, show_clusters, batch):
    session = _active(batch)
    if session is None or not session.has_segmentation:
        return _review_views(batch, show_ids, show_clusters)
    ids, unknown = _parse_ids(object_ids, session)
    checkpoint(session)
    changed = sum(exclude_object(session.qc, i, reason or "manual exclusion") for i in ids)
    message = "Excluded {} object(s).".format(changed)
    if unknown:
        message += "  No such cell: {}.".format(", ".join(str(u) for u in unknown))
    return _review_views(batch, show_ids, show_clusters, message)


def on_restore(object_ids, show_ids, show_clusters, batch):
    session = _active(batch)
    if session is None or not session.has_segmentation:
        return _review_views(batch, show_ids, show_clusters)
    ids, unknown = _parse_ids(object_ids, session)
    checkpoint(session)
    changed = sum(restore_object(session.qc, i) for i in ids)
    message = "Restored {} object(s).".format(changed)
    if unknown:
        message += "  No such cell: {}.".format(", ".join(str(u) for u in unknown))
    return _review_views(batch, show_ids, show_clusters, message)


def on_exclude_cluster(cluster_id, show_ids, show_clusters, batch):
    session = _active(batch)
    if session is None or not session.has_segmentation:
        return _review_views(batch, show_ids, show_clusters)
    results = compute_results(session)
    target = next((c for c in results.clusters if c.cluster_id == int(cluster_id)), None)
    if target is None:
        return _review_views(
            batch, show_ids, show_clusters, "No cluster {}.".format(int(cluster_id))
        )
    checkpoint(session)
    removed = exclude_cluster(session.qc, target)
    return _review_views(
        batch, show_ids, show_clusters,
        "Excluded cluster {} ({} object(s)).".format(int(cluster_id), removed),
    )


def on_exclude_border(show_ids, show_clusters, batch):
    session = _active(batch)
    if session is None or not session.has_segmentation:
        return _review_views(batch, show_ids, show_clusters)
    checkpoint(session)
    removed = exclude_border_touching(session.qc, session.objects)
    return _review_views(
        batch, show_ids, show_clusters, "Excluded {} border object(s).".format(removed)
    )


def on_filter_area(min_area, max_area, show_ids, show_clusters, batch):
    session = _active(batch)
    if session is None or not session.has_segmentation:
        return _review_views(batch, show_ids, show_clusters)
    checkpoint(session)
    removed = exclude_by_area(
        session.qc,
        session.object_labels,
        min_area=float(min_area) if min_area and float(min_area) > 0 else None,
        max_area=float(max_area) if max_area and float(max_area) > 0 else None,
    )
    return _review_views(
        batch, show_ids, show_clusters, "Excluded {} object(s) by area.".format(removed)
    )


def on_reset_qc(show_ids, show_clusters, batch):
    session = _active(batch)
    if session is not None:
        checkpoint(session)
        for oid in list(session.qc.excluded_ids):
            session.qc.restore(oid, "reset all QC")
    return _review_views(batch, show_ids, show_clusters, "QC reset; all objects restored.")


# --------------------------------------------------------------------------- #
# Tab 4 -- Results
# --------------------------------------------------------------------------- #

PSEUDOREPLICATION_NOTE = (
    "<strong>On analytical levels.</strong> These figures pool every accepted "
    "cell across every image in this batch, weighted by cell count, to describe "
    "the batch. Cells are <em>not</em> independent biological replicates: a "
    "comparison between conditions must use the well or the sample as its "
    "replicate unit. CellScope is a measurement and QC tool and deliberately "
    "computes no p-values."
)

HEADLINE = (
    ("n_cells", "Cells pooled"),
    ("n_images_segmented", "Images"),
    ("isolated_cells", "Isolated"),
    ("multicell_clusters", "Clusters"),
    ("unresolved_clusters", "Unresolved"),
)


def headline_html(row, calibrated) -> str:
    cells = []
    for key, label in HEADLINE:
        value = row.get(key, DASH)
        cells.append(
            '<div class="cs-stat"><span class="cs-k">{}</span>'
            '<span class="cs-v">{}</span></div>'.format(label, value)
        )
    area_key = "mean_area_um2" if calibrated else "mean_area_px2"
    unit = "µm²" if calibrated else "px²"
    area = row.get(area_key)
    text = "{:.0f} {}".format(area, unit) if area is not None and not pd.isna(area) else DASH
    cells.append(
        '<div class="cs-stat"><span class="cs-k">Mean cell area</span>'
        '<span class="cs-v">{}</span></div>'.format(text)
    )
    return '<div class="cs-status">{}</div>'.format("".join(cells))


def _round(frame, digits=3):
    if frame is None or not len(frame):
        return frame
    out = frame.copy()
    for column in out.select_dtypes("number").columns:
        out[column] = out[column].round(digits)
    return out


def on_refresh_results(batch):
    blank = _empty("cell_id")
    empty_figures = build_all_figures(blank, blank, uncalibrated())
    figure_keys = (
        "area_distribution", "major_axis_distribution", "area_vs_major",
        "length_vs_width", "cluster_size_distribution",
    )

    if batch is None or batch.is_empty:
        return (
            batch, "",
            empty_state("No results yet", "Add images on the Batch tab, then segment them."),
            blank, blank, blank,
            *(empty_figures[k] for k in figure_keys), status_html(batch),
        )

    if not batch.segmented:
        return (
            batch, "",
            empty_state(
                "Nothing segmented yet",
                "{} image(s) are loaded. Run segmentation to see results.".format(
                    len(batch.images)
                ),
            ),
            blank, blank, blank,
            *(empty_figures[k] for k in figure_keys), status_html(batch),
        )

    verdict = calibration_consistency(batch)
    if not verdict["consistent"]:
        return (
            batch, "", callout(verdict["reason"], tone="warn"), blank, blank, blank,
            *(empty_figures[k] for k in figure_keys), status_html(batch),
        )

    try:
        summary = batch_summary(batch)
    except CalibrationMismatch as exc:
        return (
            batch, "", callout(str(exc), tone="warn"), blank, blank, blank,
            *(empty_figures[k] for k in figure_keys), status_html(batch),
        )

    cells = pooled_cells(batch)
    clusters = pooled_clusters(batch)
    figures = build_all_figures(cells, clusters, batch.calibration)

    banner = PSEUDOREPLICATION_NOTE
    if batch.n_unreviewed:
        banner = (
            "<strong>{} of {} image(s) not yet reviewed:</strong> {}. "
            "The figures below already include them.<br><br>".format(
                batch.n_unreviewed, len(batch.images),
                ", ".join(batch.unreviewed_names[:5])
                + ("..." if batch.n_unreviewed > 5 else ""),
            )
            + PSEUDOREPLICATION_NOTE
        )
        note = callout(banner, tone="warn")
    else:
        note = callout(banner)

    return (
        batch,
        headline_html(summary.iloc[0], verdict["calibrated"]),
        note,
        _round(summary.T.reset_index().rename(columns={"index": "metric", 0: "value"})),
        _round(per_image_summary(batch)),
        _round(cells),
        *(figures[k] for k in figure_keys),
        status_html(batch),
    )


# --------------------------------------------------------------------------- #
# Tab 5 -- Export
# --------------------------------------------------------------------------- #


def on_export(batch):
    if batch is None or batch.is_empty:
        return None, "Load images first."
    if not batch.segmented:
        return None, "Segment at least one image before exporting."

    overlays = {}
    for session in batch.images:
        if session.has_segmentation:
            results = compute_results(session)
            overlays[session.analysis_id] = make_overlay(
                _display_base(session), results.qc_labels, results.objects, results.clusters
            )

    try:
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        path = export_batch(batch, OUTPUT_DIR, overlays=overlays)
    except CalibrationMismatch as exc:
        return None, _error(exc)
    except Exception as exc:
        traceback.print_exc()
        return None, _error(exc)

    size_kb = os.path.getsize(path) / 1024
    warning = ""
    if batch.n_unreviewed:
        warning = "  \n**{} image(s) were not reviewed**; the count is recorded in " \
                  "`metadata.json`.".format(batch.n_unreviewed)
    return path, (
        "Wrote **{}** ({:.0f} KB) covering {} image(s) and {} pooled cell(s).{}".format(
            os.path.basename(path), size_kb, len(batch.segmented),
            len(pooled_cells(batch)), warning,
        )
    )


# --------------------------------------------------------------------------- #
# Interface (§21)
# --------------------------------------------------------------------------- #


def build_interface():
    engine_choices = ["auto", "threshold_watershed"]
    if cellpose_available():
        engine_choices.insert(0, "cellpose")

    legend = legend_html()
    scope_choices = ["Whole batch", "This image only"]

    with gr.Blocks(title="CellScope", theme=build_theme(), css=CSS) as demo:
        batch = gr.State(BatchSession())

        with gr.Row(elem_id="cs-header"):
            home_button = gr.Button("CellScope", elem_id="cs-home", scale=0)
            gr.HTML(
                '<span class="cs-sub">Fluorescence cell morphometry &middot; '
                'v{} &middot; click the title to start a new batch</span>'.format(__version__)
            )
            theme_mode = gr.Radio(["System", "Light", "Dark"], value="System",
                                  label="Appearance", scale=0, min_width=230)
        demo.load(None, outputs=[theme_mode], js=THEME_JS)
        theme_mode.input(None, inputs=[theme_mode], outputs=[],
                         js="(mode) => { window.cellscopeTheme(mode); }")
        status = gr.HTML(status_html(None))

        with gr.Tabs() as tabs:
            # ------------------------------------------------------------- #
            with gr.Tab("Batch", id="tab_batch") as batch_tab:
                with gr.Row():
                    with gr.Column(scale=1, min_width=320):
                        gr.HTML(section("The batch"))
                        batch_label = gr.Textbox(
                            label="Experiment name", placeholder="shRNA FST, plate 2",
                            info="Recorded in the export.",
                        )
                        upload = gr.File(
                            label="Add images",
                            file_types=["image", ".tif", ".tiff"],
                            file_count="multiple",
                            type="filepath",
                        )
                        gr.HTML(section("Channels (apply to the whole batch)"))
                        channel = gr.Radio(
                            ["grayscale"], value="grayscale",
                            label="Cell channel",
                            info=(
                                "What gets measured. Applies to every image in the "
                                "batch, including ones added later. Recorded in "
                                "metadata; never picked for you."
                            ),
                        )
                        with gr.Group():
                            nuclear_channel = gr.Radio(
                                [_NO_NUCLEAR], value=_NO_NUCLEAR,
                                label="Nuclear channel (optional)",
                                info=(
                                    "Greatly improves separation of touching cells. "
                                    "Applies to the whole batch. Guides segmentation "
                                    "only, never measured."
                                ),
                            )
                            nuclear_file = gr.File(
                                label="...or a separate nuclear file",
                                file_types=["image", ".tif", ".tiff"], type="filepath",
                            )
                        remove_button = gr.Button("Remove image", size="sm", variant="stop")
                        channel_note = gr.Markdown(elem_classes="cs-note")
                    with gr.Column(scale=2, min_width=420):
                        preview = gr.Image(
                            label="Active image", type="numpy", show_download_button=False
                        )
                        batch_note = gr.Markdown(elem_classes="cs-note")

                gr.HTML(section("Images in this batch"))
                gr.Markdown(
                    "Click a row to make it active. **Well**, **Channel** and "
                    "**Nuclear** are editable here — use them to override a single "
                    "image; the radios above set the whole batch.",
                    elem_classes="cs-note",
                )
                image_grid = gr.Dataframe(
                    value=_empty(*IMAGE_TABLE_COLUMNS), interactive=True, wrap=True,
                )

                with gr.Accordion("Calibration", open=True):
                    calibration_scope = gr.Radio(
                        scope_choices, value="Whole batch", label="Apply to",
                        info="Images in one batch normally share a magnification.",
                    )
                    calibration_method = gr.Radio(
                        CALIBRATION_METHODS, value=CALIBRATION_METHODS[0],
                        label="Method", show_label=False,
                    )
                    calibration_note = gr.Markdown(
                        "Uncalibrated -- results will be in px and px2.",
                        elem_classes="cs-note",
                    )

                    with gr.Group(visible=True) as group_bar:
                        gr.Markdown(
                            "Click both ends of the bar on the active image above.",
                            elem_classes="cs-note",
                        )
                        click_note = gr.Markdown(elem_classes="cs-note")
                        with gr.Row():
                            bar_um = gr.Number(value=100.0, label="Bar length (um)", precision=4)
                            apply_bar = gr.Button("Apply", variant="primary")
                            clear_points = gr.Button("Clear clicks")

                    with gr.Group(visible=False) as group_reference:
                        gr.Markdown(
                            "Enter the ruler's known length in µm and confirm its measured pixel length. "
                            "Different image dimensions or crops are allowed. Use a reference with "
                            "the same magnification and pixel scaling as the sample.",
                            elem_classes="cs-note",
                        )
                        with gr.Row():
                            reference_upload = gr.File(
                                label="Reference frame",
                                file_types=["image", ".tif", ".tiff"], type="filepath",
                            )
                            reference_preview = gr.Image(
                                label="Detected bar", type="numpy", height=150,
                                show_download_button=False,
                            )
                        reference_note = gr.Markdown(elem_classes="cs-note")
                        with gr.Row():
                            detected_px = gr.Number(label="Bar length (px)", precision=2)
                            reference_um = gr.Number(value=50.0, label="Represents (um)", precision=4)
                            apply_reference = gr.Button("Apply", variant="primary")

                    with gr.Group(visible=False) as group_resolution:
                        gr.Markdown("Supports anisotropic pixels.", elem_classes="cs-note")
                        with gr.Row():
                            res_x = gr.Number(value=1.0, label="um per pixel (X)", precision=6)
                            res_y = gr.Number(value=1.0, label="um per pixel (Y)", precision=6)
                            apply_resolution = gr.Button("Apply", variant="primary")

                    with gr.Group(visible=False) as group_uncalibrated:
                        gr.Markdown(
                            "Analysis proceeds in pixels. No physical columns are produced "
                            "-- CellScope will not invent a micron value.",
                            elem_classes="cs-note",
                        )
                        set_uncalibrated = gr.Button("Confirm pixel units", variant="primary")

                # ------------------------------------------------------------- #
            with gr.Tab("Segment") as segment_tab:
                segment_nav = gr.HTML()
                with gr.Row():
                    with gr.Column(scale=1, min_width=320):
                        run_all = gr.Button("Run on all images", variant="primary", size="lg")
                        run_one = gr.Button("Run on this image only")
                        skip_done = gr.Checkbox(
                            value=True, label="Skip images already segmented",
                            info=(
                                "Cellpose costs several seconds an image; adding one "
                                "field should not re-run the rest. Untick to redo "
                                "everything after changing a setting."
                            ),
                        )
                        gr.Markdown(
                            "Runs only when pressed. Moving a slider never re-segments.",
                            elem_classes="cs-note",
                        )
                        engine = gr.Radio(
                            engine_choices, value="auto", label="Engine",
                            info="The fallback is always labelled as such in the export.",
                        )
                        with gr.Row():
                            diameter = gr.Number(value=0, label="Cell diameter (px)",
                                                 info="0 = automatic")
                            min_area = gr.Number(value=200, label="Min object area (px)")
                        cellprob = gr.Slider(-6, 6, value=0.0, step=0.1,
                                             label="Cell probability threshold")
                        flow = gr.Slider(0.0, 3.0, value=0.4, step=0.05,
                                         label="Flow threshold")
                        analysis_scale = gr.Slider(
                            0.25, 1.0, value=1.0, step=0.05,
                            label="Analysis resolution",
                            info=(
                                "Segments a downscaled copy; masks return at full "
                                "size, so measurements are unaffected. On the "
                                "reference field 0.75x was 1.36x faster at F1 "
                                "0.971, 0.50x was 3.1x faster at F1 0.909. Below "
                                "1.00 is recorded in the export."
                            ),
                        )

                        with gr.Accordion("Model and hardware", open=False):
                            model = gr.Textbox(value="cpsam", label="Cellpose model")
                            use_gpu = gr.Checkbox(value=True, label="Use GPU if available")

                        with gr.Accordion("Preprocessing", open=False):
                            gr.Markdown(
                                "Affects segmentation only. Measurements always read "
                                "the original image.",
                                elem_classes="cs-note",
                            )
                            normalize = gr.Checkbox(value=True, label="Normalise contrast")
                            gaussian_sigma = gr.Number(value=1.0, label="Gaussian sigma")
                            background_subtract = gr.Checkbox(
                                value=False, label="Subtract background"
                            )
                            background_radius = gr.Number(
                                value=50.0, label="Background radius (px)"
                            )

                        with gr.Accordion("Touching cells and clustering", open=False):
                            gr.Markdown(
                                "A split is accepted only if every gate passes. Anything "
                                "ambiguous stays whole as an unresolved cluster.",
                                elem_classes="cs-note",
                            )
                            split_enabled = gr.Checkbox(value=True, label="Attempt splitting")
                            saddle_depth = gr.Slider(
                                0.05, 0.8, value=0.25, step=0.05,
                                label="Minimum neck depth",
                                info="Higher refuses more splits.",
                            )
                            solidity_max = gr.Slider(
                                0.5, 1.0, value=0.95, step=0.01,
                                label="Flag below solidity",
                            )
                            min_fragment_frac = gr.Slider(
                                0.02, 0.5, value=0.15, step=0.01,
                                label="Min fragment size (fraction of parent)",
                            )
                            contact_distance = gr.Slider(
                                0, 10, value=2, step=1, label="Contact distance (px)",
                            )
                    with gr.Column(scale=2, min_width=420):
                        segmentation_preview = gr.Image(
                            label="Segmentation", type="numpy", show_download_button=True
                        )
                        gr.HTML(legend)
                        segmentation_note = gr.Markdown(elem_classes="cs-note")

            # ------------------------------------------------------------- #
            with gr.Tab("Review") as review_tab:
                review_nav = gr.HTML()
                review_strip = gr.HTML()
                with gr.Row():
                    with gr.Column(scale=1, min_width=320):
                        with gr.Row():
                            review_prev = gr.Button("Prev", size="sm")
                            review_next = gr.Button("Next", size="sm")
                        review_note_box = gr.Textbox(
                            label="Review note (optional)",
                            placeholder="out of focus at the edge",
                        )
                        with gr.Row():
                            mark_reviewed = gr.Button("Mark reviewed", variant="primary")
                            unmark_reviewed = gr.Button("Unmark")
                        gr.Markdown(
                            "Edge-touching cells are automatically excluded after segmentation. "
                            "Click a cell to exclude it; click its gray outline or interior to restore it. "
                            "Exclusions apply to a working copy. The raw segmentation "
                            "is never modified and ships alongside it.",
                            elem_classes="cs-note",
                        )
                        with gr.Row():
                            show_ids = gr.Checkbox(value=True, label="Cell IDs")
                            show_clusters = gr.Checkbox(value=False, label="Cluster IDs")

                        with gr.Group():
                            exclude_ids = gr.Textbox(
                                label="Exclude cells", placeholder="12 15 23",
                                info="IDs, space or comma separated",
                            )
                            exclude_reason = gr.Textbox(
                                label="Reason", value="segmentation error",
                                info="Recorded in qc_log.csv",
                            )
                            exclude_button = gr.Button("Exclude", variant="stop")
                        with gr.Group():
                            restore_ids = gr.Textbox(label="Restore cells", placeholder="12")
                            restore_button = gr.Button("Restore")

                        with gr.Accordion("Bulk filters", open=False):
                            with gr.Row():
                                cluster_id_input = gr.Number(value=1, label="Cluster ID")
                                exclude_cluster_button = gr.Button("Exclude cluster")
                            exclude_border_button = gr.Button("Exclude border-touching cells")
                            with gr.Row():
                                filter_min = gr.Number(value=0, label="Min area (px)")
                                filter_max = gr.Number(value=0, label="Max area (px)")
                            filter_button = gr.Button("Apply area filter")
                        reset_button = gr.Button("Reset all QC")
                        from professional_ui import build_review_controls
                        review_tools = build_review_controls()
                    with gr.Column(scale=2, min_width=420):
                        qc_preview = gr.Image(
                            label="Cell selection — click to exclude / restore", type="numpy",
                            interactive=False, elem_id="cs-review-image",
                            show_download_button=True,
                        )
                        gr.HTML(legend)
                        qc_note = gr.Markdown(elem_classes="cs-note")
                        review_tools.detail = gr.Markdown()
                        with gr.Accordion("Linked cell measurements", open=False):
                            review_tools.table = gr.Dataframe(interactive=False, wrap=False)
                        with gr.Accordion("QC log for this image", open=False):
                            qc_log = gr.Dataframe(interactive=False, wrap=True)

            # ------------------------------------------------------------- #
            with gr.Tab("Results", elem_id="cs-results") as results_tab:
                headline = gr.HTML()
                with gr.Row():
                    refresh_button = gr.Button("Refresh", size="sm", scale=0)
                    results_note = gr.HTML()

                with gr.Tabs():
                    with gr.Tab("Batch summary"):
                        gr.Markdown(
                            "Pooled across every accepted cell in the batch.",
                            elem_classes="cs-note",
                        )
                        summary_table = gr.Dataframe(interactive=False, wrap=True)
                    with gr.Tab("Per image"):
                        gr.Markdown(
                            "A QC view: one row per image, for spotting a field unlike "
                            "its neighbours before trusting the pooled number.",
                            elem_classes="cs-note",
                        )
                        per_image_table = gr.Dataframe(interactive=False, wrap=True)
                    with gr.Tab("All cells"):
                        gr.Markdown(
                            "Every resolved cell in the batch, tagged with its image "
                            "and well. Unresolved groups are absent by design.",
                            elem_classes="cs-note",
                        )
                        cells_table = gr.Dataframe(interactive=False, wrap=True)
                    with gr.Tab("Distributions"):
                        with gr.Row():
                            figure_area = gr.Plot(label="Cell area")
                            figure_major = gr.Plot(label="Major axis")
                        with gr.Row():
                            figure_area_major = gr.Plot(label="Area vs major axis")
                            figure_length_width = gr.Plot(label="Length vs width")
                        figure_cluster_size = gr.Plot(label="Cluster size")

            # ------------------------------------------------------------- #
            with gr.Tab("Export"):
                gr.Markdown(
                    "Everything needed to reproduce and audit the batch.",
                    elem_classes="cs-note",
                )
                export_button = gr.Button(
                    "Build batch_analysis.zip", variant="primary", size="lg"
                )
                export_file = gr.File(label="Download")
                export_note = gr.Markdown(elem_classes="cs-note")
                with gr.Accordion("What is in the archive", open=False):
                    gr.Markdown(
                        "| File | Contents |\n|---|---|\n"
                        "| `batch_summary.csv` | the pooled result |\n"
                        "| `per_image_summary.csv` | one row per image |\n"
                        "| `all_cells.csv` | every cell, with `image_id` and `well` |\n"
                        "| `all_clusters.csv` | every cluster, tagged the same way |\n"
                        "| `qc_log.csv` | every exclusion across the batch |\n"
                        "| `metadata.json` | batch and per-image provenance |\n"
                        "| `images/<name>/...` | a full single-image analysis each |"
                    )

            from professional_ui import build_tools
            extensions = build_tools(batch, review_tools)
            from nuclei_ui import build_nuclei_tab
            build_nuclei_tab(batch)

        # ----------------------------------------------------------------- #
        # Events
        # ----------------------------------------------------------------- #

        #: Widgets that follow whichever image is active.
        active_outputs = [channel, nuclear_channel, preview, channel_note]
        #: The batch tab's own views.
        batch_views = [batch, image_grid, segment_nav, review_strip, batch_note, status]
        nav_outputs = batch_views + active_outputs

        segment_outputs = [
            batch, segmentation_preview, segmentation_note, image_grid,
            segment_nav, review_strip, status,
        ]
        review_outputs = [
            batch, qc_preview, qc_log, qc_note, review_nav, review_strip,
            image_grid, status,
        ]
        results_outputs = [
            batch, headline, results_note, summary_table, per_image_table, cells_table,
            figure_area, figure_major, figure_area_major, figure_length_width,
            figure_cluster_size, status,
        ]

        segment_inputs = [
            model, diameter, cellprob, flow, min_area, use_gpu,
            background_subtract, background_radius, gaussian_sigma, normalize,
            split_enabled, saddle_depth, solidity_max, min_fragment_frac,
            contact_distance, nuclear_channel, analysis_scale, engine, batch,
        ]

        def on_calibration_method(choice):
            """Show one calibration panel at a time."""
            return [gr.update(visible=choice == name) for name in CALIBRATION_METHODS]

        calibration_method.change(
            on_calibration_method, calibration_method,
            [group_bar, group_reference, group_resolution, group_uncalibrated],
        )

        home_button.click(
            on_home, [batch],
            [batch, batch_note, tabs, image_grid, segment_nav, review_strip, status]
            + active_outputs,
            show_progress="hidden",
        )
        batch_label.change(on_batch_label, [batch_label, batch], [batch, status])
        upload.change(on_upload, [upload, batch], nav_outputs,
                      show_progress_on=[image_grid])
        image_grid.select(on_select_image, [batch], nav_outputs, show_progress="hidden")
        image_grid.input(on_table_edited, [image_grid, batch], [batch, image_grid, status],
                         show_progress="hidden")
        remove_button.click(on_remove_image, [batch], nav_outputs,
                            show_progress_on=[image_grid])

        channel.change(on_channel_change, [channel, batch],
                       [batch, image_grid, segment_nav], show_progress="hidden")
        nuclear_channel.change(on_nuclear_change, [nuclear_channel, batch],
                               [batch, image_grid, segment_nav], show_progress="hidden")
        nuclear_file.change(on_nuclear_file, [nuclear_file, batch],
                            [batch, channel_note, image_grid],
                            show_progress_on=[channel_note])

        preview.select(on_image_click, [batch], [batch, click_note],
                       show_progress="hidden")
        clear_points.click(on_clear_points, [batch], [batch, click_note],
                           show_progress="hidden")

        calibration_outputs = [batch, calibration_note, status, image_grid]
        apply_bar.click(on_apply_scale_bar, [bar_um, calibration_scope, batch],
                        calibration_outputs, show_progress_on=[calibration_note])
        reference_upload.change(
            on_reference_upload, [reference_upload, batch],
            [batch, detected_px, reference_note, reference_preview],
            show_progress_on=[reference_preview],
        )
        apply_reference.click(
            on_apply_reference, [detected_px, reference_um, calibration_scope, batch],
            calibration_outputs, show_progress_on=[calibration_note],
        )
        apply_resolution.click(on_apply_resolution, [res_x, res_y, calibration_scope, batch],
                               calibration_outputs, show_progress_on=[calibration_note])
        set_uncalibrated.click(on_set_uncalibrated, [calibration_scope, batch],
                               calibration_outputs, show_progress_on=[calibration_note])

        # Segmentation, then the dependent tabs. The follow-ups are silent
        # because they update tabs that are not on screen.
        run_events = []
        for button, handler, extra in (
            (run_one, on_segment_one, []),
            (run_all, on_segment_all, [skip_done]),
        ):
            run_event = button.click(
                handler, segment_inputs[:-1] + extra + [batch], segment_outputs,
                show_progress_on=[segmentation_preview],
            )
            run_events.append(run_event)
            run_event.then(
                on_toggle_labels, [show_ids, show_clusters, batch], review_outputs,
                show_progress="hidden",
            ).then(
                on_refresh_results, [batch], results_outputs, show_progress="hidden",
            )

        # Both runs, not just whichever the loop above happened to wire last.
        extensions.cancel.click(None, cancels=run_events, queue=False)

        for control in (show_ids, show_clusters):
            control.change(on_toggle_labels, [show_ids, show_clusters, batch],
                           review_outputs, show_progress_on=[qc_preview])

        review_prev.click(on_prev, [batch], nav_outputs, show_progress="hidden").then(
            on_toggle_labels, [show_ids, show_clusters, batch], review_outputs,
            show_progress_on=[qc_preview],
        )
        review_next.click(on_next, [batch], nav_outputs, show_progress="hidden").then(
            on_toggle_labels, [show_ids, show_clusters, batch], review_outputs,
            show_progress_on=[qc_preview],
        )
        qc_preview.select(on_review_click, [show_ids, show_clusters, batch],
                          review_outputs, show_progress="hidden", trigger_mode="always_last")
        mark_reviewed.click(
            on_mark_reviewed, [review_note_box, show_ids, show_clusters, batch],
            review_outputs, show_progress="hidden",
        )
        unmark_reviewed.click(
            on_unmark_reviewed, [show_ids, show_clusters, batch], review_outputs,
            show_progress="hidden",
        )

        exclude_button.click(
            on_exclude, [exclude_ids, exclude_reason, show_ids, show_clusters, batch],
            review_outputs, show_progress_on=[qc_preview],
        )
        restore_button.click(
            on_restore, [restore_ids, show_ids, show_clusters, batch], review_outputs,
            show_progress_on=[qc_preview],
        )
        exclude_cluster_button.click(
            on_exclude_cluster, [cluster_id_input, show_ids, show_clusters, batch],
            review_outputs, show_progress_on=[qc_preview],
        )
        exclude_border_button.click(
            on_exclude_border, [show_ids, show_clusters, batch], review_outputs,
            show_progress_on=[qc_preview],
        )
        filter_button.click(
            on_filter_area, [filter_min, filter_max, show_ids, show_clusters, batch],
            review_outputs, show_progress_on=[qc_preview],
        )
        reset_button.click(
            on_reset_qc, [show_ids, show_clusters, batch], review_outputs,
            show_progress_on=[qc_preview],
        )

        refresh_button.click(on_refresh_results, [batch], results_outputs,
                             show_progress_on=[summary_table])
        export_button.click(on_export, [batch], [export_file, export_note],
                            show_progress_on=[export_file])

        review_tab.select(on_toggle_labels, [show_ids, show_clusters, batch], review_outputs,
                          show_progress="hidden")
        results_tab.select(on_refresh_results, [batch], results_outputs, show_progress="hidden")
        def refresh_batch_view(b):
            return (b, image_table(b), nav_html(b), strip_html(b), "", status_html(b)) + tuple(_active_controls(b))
        batch_tab.select(refresh_batch_view, [batch], nav_outputs, show_progress="hidden")
        segment_tab.select(lambda b: _segment_views(b, ""), [batch], segment_outputs, show_progress="hidden")
        extensions.wire(demo, batch, nav_outputs, review_outputs, results_outputs,
                        show_ids, show_clusters, qc_preview, review_note_box, segment_inputs[:-2])

    return demo


def preload_model_in_background():
    """Load the Cellpose weights while the user is still uploading.

    The weights take about 8 s to reach the GPU and are cached module-level for
    the process. Doing it here means the first segmentation pays only for the
    segmentation. Failures are ignored on purpose: this is an optimisation, and
    the real run reports any genuine problem.
    """
    import threading
    import time

    def warm():
        try:
            from src.segmentation import _get_cellpose_model, cellpose_available

            if cellpose_available():
                started = time.perf_counter()
                _get_cellpose_model("cpsam", True)
                print("Cellpose weights ready ({:.1f}s)".format(
                    time.perf_counter() - started
                ))
        except Exception as error:
            # Not fatal: the real run will load the model itself and report any
            # genuine problem. Printed rather than swallowed so it is visible.
            print("Model preload skipped: {}: {}".format(type(error).__name__, error))

    threading.Thread(target=warm, daemon=True).start()


if __name__ == "__main__":
    print("CellScope {} | Cellpose {}".format(
        __version__, cellpose_version() or "not installed"
    ))
    preload_model_in_background()
    build_interface().launch()
