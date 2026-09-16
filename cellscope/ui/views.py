"""Views shared by every tab: status, stepper, navigator, image strip and table.

Each is a pure function of the batch and cheap to call: they read cached
results (see :func:`src.pipeline.compute_results`) and never measure anything
themselves.
"""

from __future__ import annotations

import functools
from html import escape

import pandas as pd

from src.batch import headline_counts
from src.image_io import make_display
from src.pipeline import compute_results
from src.types import AnalysisSession, BatchSession
from src.workflow import image_state, is_stale, workflow_state

from .theme import badge, stepper_html

DASH = "—"
NO_NUCLEAR = "none"
STATUS_FIELDS = ("Batch", "Images", "Reviewed", "Calibration", "Segmented", "Cells")
IMAGE_TABLE_COLUMNS = ["#", "File", "Well", "Channel", "Nuclear", "Status", "Cells", "Reviewed"]


def empty_frame(*columns) -> pd.DataFrame:
    return pd.DataFrame(columns=list(columns))


def error_text(exc: Exception) -> str:
    return "**{}**: {}".format(type(exc).__name__, exc)


def active(batch: BatchSession | None) -> AnalysisSession | None:
    return batch.active if batch is not None else None


def locked(handler):
    """Run a callback under its batch's lock.

    A batch run commits results from a background generator, and autosave
    snapshots the batch from another thread; interface edits take the same lock
    so none of them sees a half-applied change.
    """
    @functools.wraps(handler)
    def wrapper(*args, **kwargs):
        batch = next((a for a in args if isinstance(a, BatchSession)), None)
        if batch is None:
            return handler(*args, **kwargs)
        with batch.lock():
            return handler(*args, **kwargs)

    return wrapper


def display_base(session: AnalysisSession | None):
    """The browser-sized RGB the overlay is drawn on, rebuilt if missing."""
    if session is None or session.image is None:
        return None
    if session.display_rgb is None:
        session.display_rgb, session.display_scale = make_display(session.image)
    return session.display_rgb


# --------------------------------------------------------------------------- #
# Status strip and stepper
# --------------------------------------------------------------------------- #


def status_fields(batch: BatchSession | None) -> dict:
    """The persistent batch-level header."""
    if batch is None or batch.is_empty:
        return dict.fromkeys(STATUS_FIELDS, DASH)
    counts = headline_counts(batch)
    calibration = batch.calibration.summary()
    if not batch.calibration.is_calibrated and batch.calibration_confirmed:
        calibration = "Pixel units (confirmed)"
    return {
        "Batch": batch.label or "(unnamed)",
        "Images": str(len(batch.images)),
        "Reviewed": "{} / {}".format(counts["n_reviewed"], counts["n_images"]),
        "Calibration": calibration,
        "Segmented": "{} / {}".format(counts["n_segmented"], counts["n_images"]),
        "Cells": str(counts["n_cells"]) if counts["consistent"] else "mixed units",
    }


def status_html(batch: BatchSession | None) -> str:
    """Stepper, next action, and the labelled status strip."""
    fields = status_fields(batch)
    stale = fields["Cells"] == "mixed units"
    cells = "".join(
        '<div class="cs-stat{}"><span class="cs-k">{}</span>'
        '<span class="cs-v" title="{}">{}</span></div>'.format(
            " cs-stale" if (key == "Cells" and stale) else "", key,
            escape(fields[key]), escape(fields[key]),
        )
        for key in STATUS_FIELDS
    )
    steps, action = workflow_state(batch)
    return stepper_html(steps, action) + '<div class="cs-status">{}</div>'.format(cells)


# --------------------------------------------------------------------------- #
# Navigator and strip
# --------------------------------------------------------------------------- #


def nav_html(batch: BatchSession | None) -> str:
    """Which image the current tab is acting on, and its state."""
    session = active(batch)
    if session is None:
        return '<div class="cs-nav"><span class="cs-pos">No images loaded</span></div>'

    badges = []
    state = image_state(session)
    if state == "failed":
        badges.append(badge("failed", "err"))
    elif session.has_segmentation:
        counts = compute_results(session).counts
        badges.append(badge("{} cells".format(counts["cells_accepted"])))
        if counts["objects_excluded"]:
            badges.append(badge("{} excluded".format(counts["objects_excluded"])))
        if counts["unresolved_accepted"]:
            badges.append(badge("{} unresolved".format(counts["unresolved_accepted"]), "warn"))
    else:
        badges.append(badge("not segmented"))
    if state == "stale":
        badges.append(badge("settings changed", "warn"))
    badges.append(badge("reviewed", "ok") if session.reviewed else badge("unreviewed"))
    if not session.calibration.is_calibrated:
        badges.append(badge("px units"))
    if session.well:
        badges.append(badge("well {}".format(session.well)))
    if session.annotation is not None and session.annotation.found:
        badges.append(badge("scale bar in image"))
    scale = session.engine_info.get("analysis_scale")
    if scale is not None and scale < 1.0:
        badges.append(badge("segmented at {:.2f}x".format(scale), "warn"))
    if session.engine_info.get("oom_recovery"):
        badges.append(badge("memory fallback used", "warn"))

    return (
        '<div class="cs-nav"><span class="cs-pos">{} / {}</span>'
        '<span class="cs-name" title="{}">{}</span><span class="cs-sep">|</span>{}</div>'
    ).format(
        batch.active_index + 1, len(batch.images), escape(session.display_name),
        escape(session.display_name), "".join(badges),
    )


_TILE_SYMBOL = {"reviewed": "✓", "failed": "✕", "stale": "!", "segmented": "•", "pending": "○"}


def strip_html(batch: BatchSession | None, running: dict[int, str] | None = None) -> str:
    """Every image's state at a glance. ``running`` marks images in a batch run."""
    if batch is None or batch.is_empty:
        return ""
    tiles = []
    for index, session in enumerate(batch.images):
        state = image_state(session)
        if session.reviewed and state not in ("failed", "stale"):
            state = "reviewed"
        classes = ["cs-tile", {"reviewed": "done", "failed": "err", "stale": "stale"}.get(state, "")]
        label = state
        if running and index in running:
            classes.append("running")
            label = running[index]
        if index == batch.active_index:
            classes.append("active")
        tiles.append(
            '<span class="{}" title="{}: {}"><span class="n">{}</span>'
            '<span class="s" aria-hidden="true">{}</span>{}</span>'.format(
                " ".join(c for c in classes if c), escape(session.display_name), label,
                index + 1, _TILE_SYMBOL.get(state, ""), escape(session.display_name),
            )
        )
    return '<div class="cs-strip" aria-label="Images in this batch">{}</div>'.format("".join(tiles))


def image_table(batch: BatchSession | None) -> pd.DataFrame:
    """One row per image. Well, Channel and Nuclear are editable."""
    if batch is None or batch.is_empty:
        return empty_frame(*IMAGE_TABLE_COLUMNS)
    rows = []
    for index, session in enumerate(batch.images):
        if session.error:
            status, cells = "failed", DASH
        elif session.has_segmentation:
            status = "stale" if is_stale(session) else "segmented"
            cells = str(compute_results(session).counts["cells_accepted"])
        else:
            status, cells = "pending", DASH
        nuclear = (
            session.nuclear_image.filename
            if session.nuclear_image
            else (session.segmentation_params.nuclear_channel or DASH)
        )
        rows.append([
            index + 1, session.display_name, session.well, session.channel, nuclear,
            status, cells, "yes" if session.reviewed else "no",
        ])
    return pd.DataFrame(rows, columns=IMAGE_TABLE_COLUMNS)


def batch_views(batch, message=""):
    """The views most callbacks refresh: table, nav, strip, note, status."""
    return (batch, image_table(batch), nav_html(batch), strip_html(batch), message,
            status_html(batch))
