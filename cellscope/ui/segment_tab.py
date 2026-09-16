"""Segment tab: tune settings on one image, then run the batch.

Changing a setting never re-segments anything. The tab says which segmented
images the current settings would change, and runs only when asked.
"""

from __future__ import annotations

import copy
import threading
import time
import traceback
from dataclasses import replace
from types import SimpleNamespace

import gradio as gr
import pandas as pd

from src.pipeline import (
    compute_results,
    iter_batch_segmentation,
    reapply_automatic_qc,
    segment_session,
)
from src.qc import exclusion_counts
from src.segmentation import cellpose_available
from src.types import (
    AnalysisSession,
    ClusterParams,
    PreprocessParams,
    QCParams,
    SegmentationParams,
    SplitParams,
)
from src.visualize import make_overlay
from src.workflow import image_state, is_stale, segmentation_fingerprint

from .theme import legend_html, section
from .views import (
    NO_NUCLEAR,
    active,
    display_base,
    error_text,
    image_table,
    locked,
    nav_html,
    status_html,
    strip_html,
)

RUN_SCOPES = ("Images that need it (new, failed or stale)", "All images (clears their review)")
PROGRESS_COLUMNS = ["#", "Image", "State", "Objects", "Accepted cells", "Seconds"]


# --------------------------------------------------------------------------- #
# Parameters
# --------------------------------------------------------------------------- #


def collect_params(
    model, diameter, cellprob, flow, min_area, use_gpu,
    background_subtract, background_radius, gaussian_sigma, normalize,
    split_enabled, saddle_depth, solidity_max, min_fragment_frac, contact_distance,
    nuclear_channel, analysis_scale, exclude_border=True, exclude_ruler=True,
):
    """Build every parameter object from the widget values, once."""
    nuclear = None if nuclear_channel in (None, "", NO_NUCLEAR) else nuclear_channel
    return (
        SegmentationParams(
            model=str(model or "cpsam").strip(),
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
        QCParams(exclude_border=bool(exclude_border), exclude_annotations=bool(exclude_ruler)),
    )


def params_for_widgets(target) -> tuple:
    """Widget values for a session's or batch's settings (the inverse of collect_params)."""
    seg, pre, split, cluster, qc = (target.segmentation_params, target.preprocess_params,
                                    target.split_params, target.cluster_params, target.qc_params)
    return (seg.model, seg.diameter or 0, seg.cellprob_threshold, seg.flow_threshold,
            seg.min_object_area, seg.use_gpu, pre.background_subtract, pre.background_radius_px,
            pre.gaussian_sigma, pre.normalize_contrast, split.enabled, split.min_saddle_depth_frac,
            split.solidity_max, split.min_fragment_area_frac, cluster.contact_distance_px,
            seg.nuclear_channel or NO_NUCLEAR, seg.analysis_scale, qc.exclude_border,
            qc.exclude_annotations)


def _with_params(session, params, keep_nuclear=False):
    """Assign parameters to a session. A batch-wide push keeps each image's own nuclear guide."""
    segmentation, preprocess, split, cluster, qc = params
    if keep_nuclear:
        segmentation = replace(segmentation, nuclear_channel=session.segmentation_params.nuclear_channel)
    changed = (session.segmentation_params, session.preprocess_params, session.split_params,
               session.cluster_params, session.qc_params) != (segmentation, preprocess, split, cluster, qc)
    session.segmentation_params = segmentation
    session.preprocess_params = preprocess
    session.split_params = split
    session.cluster_params = cluster
    session.qc_params = qc
    return changed


def push_params(batch, params, to_all: bool) -> None:
    """Apply parameters to the batch defaults and every image, or to the active image."""
    if to_all:
        (batch.segmentation_params, batch.preprocess_params, batch.split_params,
         batch.cluster_params, batch.qc_params) = params
        for session in batch.images:
            _with_params(session, params, keep_nuclear=True)
    elif batch.active is not None:
        _with_params(batch.active, params)


def _would_change(session, params) -> bool:
    """Would these settings produce different masks for this image?"""
    if not session.has_segmentation:
        return False
    probe = copy.copy(session)
    _with_params(probe, params, keep_nuclear=True)
    # Projects from 0.1 did not record what was used; assume the image's
    # current settings are the ones its masks came from.
    used = session.segmented_with or segmentation_fingerprint(session)
    return segmentation_fingerprint(probe) != used


def on_settings_changed(*args):
    """Say which segmented images these settings would change. Never runs anything."""
    *widgets, batch = args
    if batch is None or batch.is_empty:
        return ""
    params = collect_params(*widgets)
    segmented = [s for s in batch.images if s.has_segmentation]
    affected = [s for s in segmented if _would_change(s, params)]
    qc_changed = [s for s in segmented if s.qc_params != params[4]]
    notes = []
    if affected:
        notes.append(
            "**{} of {} segmented image(s)** were segmented with different settings. "
            "Preview on one image first; running the batch re-segments them and clears "
            "their review.".format(len(affected), len(segmented)))
    elif segmented:
        notes.append("These settings match the ones used for all {} segmented image(s).".format(
            len(segmented)))
    if qc_changed:
        notes.append("Automatic exclusion rules differ for {} image(s): use **Apply rules to "
                     "all images** to update them without re-segmenting.".format(len(qc_changed)))
    return "  \n".join(notes)


# --------------------------------------------------------------------------- #
# Views
# --------------------------------------------------------------------------- #


def _overlay(session):
    """The segmentation overlay, cached per results key."""
    if session is None or not session.has_segmentation:
        return display_base(session)
    from src.pipeline import results_key

    key = ("segment_overlay", results_key(session))
    cache = session.view_cache
    if cache.get("segment_overlay_key") != key:
        results = compute_results(session)
        cache["segment_overlay"] = make_overlay(
            display_base(session), results.qc_labels, results.objects, results.clusters,
            excluded_ids=session.qc.excluded_ids, annotation=session.annotation,
            show_cell_ids=False,
        )
        cache["segment_overlay_key"] = key
    return cache["segment_overlay"]


def metrics(session, seconds=None) -> dict:
    """What a segmentation produced, for before/after comparison."""
    if session is None or not session.has_segmentation:
        return {}
    results = compute_results(session)
    counts = results.counts
    excluded = exclusion_counts(session.objects, session.qc)
    calibrated = session.calibration.is_calibrated
    column = "area_um2" if calibrated else "area_px2"
    unit = "µm²" if calibrated else "px²"
    median = results.cells[column].median() if len(results.cells) else float("nan")
    multi = sum(1 for c in results.clusters if c.cell_count is not None and c.cell_count > 1)
    timing = session.engine_info.get("timings_seconds", {})
    return {
        "Objects detected": counts["objects_detected"],
        "Accepted cells": counts["cells_accepted"],
        "Excluded at border": excluded["excluded_border"],
        "Excluded at scale bar": excluded["excluded_annotation"],
        "Excluded otherwise": excluded["excluded_other"],
        "Unresolved groups": counts["unresolved_accepted"],
        "Multi-cell clusters": multi,
        "Median cell area ({})".format(unit): None if pd.isna(median) else round(float(median), 1),
        "Runtime (s)": seconds if seconds is not None else timing.get("total"),
        "Device": session.engine_info.get("device", session.engine_info.get("engine", "")),
    }


def comparison_table(before: dict, after: dict) -> pd.DataFrame:
    keys = list(dict.fromkeys(list(before) + list(after)))
    rows = []
    for key in keys:
        old, new = before.get(key, "—"), after.get(key, "—")
        change = ""
        if isinstance(old, (int, float)) and isinstance(new, (int, float)) and key != "Runtime (s)":
            delta = new - old
            change = "" if delta == 0 else ("+{}".format(delta) if delta > 0 else str(delta))
        rows.append([key, old, new, change])
    return pd.DataFrame(rows, columns=["Measure", "Current", "Preview", "Change"])


def progress_table(batch, report=None) -> pd.DataFrame:
    if batch is None or batch.is_empty:
        return pd.DataFrame(columns=PROGRESS_COLUMNS)
    running = {}
    if report is not None:
        names = [name for name, _ in report["status"]]
        states = [state for _, state in report["status"]]
        running = dict(zip(names, states))
    rows = []
    for index, session in enumerate(batch.images, start=1):
        state = running.get(session.display_name) or image_state(session)
        objects = len(session.objects) if session.has_segmentation else ""
        cells = compute_results(session).counts["cells_accepted"] if session.has_segmentation else ""
        seconds = session.engine_info.get("timings_seconds", {}).get("total", "") \
            if session.has_segmentation else ""
        rows.append([index, session.display_name, state, objects, cells, seconds])
    return pd.DataFrame(rows, columns=PROGRESS_COLUMNS)


def segment_views(batch, message, report=None, comparison=None):
    session = active(batch)
    return (
        batch, _overlay(session), message, image_table(batch), nav_html(batch),
        strip_html(batch), status_html(batch), progress_table(batch, report),
        comparison if comparison is not None else gr.update(),
    )


def describe_run(session) -> str:
    results = compute_results(session)
    info = session.engine_info
    timing = info.get("timings_seconds", {})
    note = ""
    if timing:
        note += "\n\nTook **{:.1f} s** (segmentation {:.1f} s, splitting {:.1f} s).".format(
            timing.get("total", 0), timing.get("segmentation", 0), timing.get("splitting", 0))
    if info.get("device"):
        note += " Device **{}**, tile batch {}.".format(info["device"], info.get("tile_batch_size", "—"))
    for step in info.get("oom_recovery", []):
        note += "\n\n**GPU memory:** " + step
    if info.get("cellpose_error"):
        note += "\n\n**Cellpose could not run; the threshold fallback was used:** `{}`".format(
            info["cellpose_error"])
    unresolved = results.counts["unresolved_accepted"]
    if unresolved:
        note += ("\n\n**{} object(s) could not be separated into cells.** They are kept as "
                 "unresolved groups with group-level measurements only; their cell counts are NA."
                 .format(unresolved))
    if session.annotation is not None and session.annotation.found:
        note += "\n\nA burned-in scale bar was found; objects touching it are excluded."
    return "Segmented **{}** with **{}**: {} object(s), {} accepted cell(s), {} cluster(s).{}".format(
        session.display_name, info.get("engine", "?"), len(session.objects),
        results.counts["cells_accepted"], len(results.clusters), note)


def channel_divergence_warning(batch) -> str:
    """Report when the segmented images did not all use the same channels."""
    segmented = [s for s in batch.images if s.has_segmentation]
    if len(segmented) < 2:
        return ""

    def tally(attribute):
        counts: dict[str, int] = {}
        for session in segmented:
            counts[attribute(session)] = counts.get(attribute(session), 0) + 1
        return ", ".join("{} x{}".format(name, n)
                         for name, n in sorted(counts.items(), key=lambda kv: (-kv[1], kv[0])))

    def nuclear_of(session):
        if session.nuclear_image is not None:
            return session.nuclear_image.filename
        return session.segmentation_params.nuclear_channel or "none"

    lines = []
    if len({s.segmented_channel for s in segmented}) > 1:
        lines.append("cell channel — {}".format(tally(lambda s: str(s.segmented_channel))))
    if len({nuclear_of(s) for s in segmented}) > 1:
        lines.append("nuclear guide — {}".format(tally(nuclear_of)))
    if not lines:
        return ""
    return (
        "\n\n**These {} images were not all segmented the same way:**\n{}\n\nThat is "
        "legitimate if the fields genuinely differ. If they should match, set the channels "
        "on the Import tab and run again."
    ).format(len(segmented), "\n".join("- " + line for line in lines))


# --------------------------------------------------------------------------- #
# Actions
# --------------------------------------------------------------------------- #


def on_segment_one(*args, progress=gr.Progress()):
    """Segment the active image with the current settings, and keep the result."""
    *widgets, engine, batch = args
    session = active(batch)
    if session is None or not session.has_image:
        return segment_views(batch, "Add images on the Import tab first.")
    before = metrics(session)
    with batch.lock():
        push_params(batch, collect_params(*widgets), to_all=False)
    started = time.perf_counter()
    try:
        progress(0.2, desc="Segmenting {}".format(session.display_name))
        segment_session(session, engine=engine)
    except Exception as exc:
        traceback.print_exc()
        with batch.lock():
            session.error = "{}: {}".format(type(exc).__name__, exc)
        return segment_views(batch, error_text(exc))
    after = metrics(session, round(time.perf_counter() - started, 2))
    table = comparison_table(before, after) if before else comparison_table({}, after)
    return segment_views(batch, describe_run(session), comparison=table)


def on_preview(*args, progress=gr.Progress()):
    """Segment a copy of the active image with these settings, without keeping it."""
    *widgets, engine, batch = args
    session = active(batch)
    if session is None or not session.has_image:
        return segment_views(batch, "Add images on the Import tab first.") + (gr.update(),)
    candidate = AnalysisSession()
    for name in ("image", "display_rgb", "display_scale", "channel", "calibration",
                 "nuclear_image", "nuclear_image_channel", "well", "condition", "replicate"):
        setattr(candidate, name, getattr(session, name))
    _with_params(candidate, collect_params(*widgets))
    candidate.segmentation_params = replace(
        candidate.segmentation_params, nuclear_channel=session.segmentation_params.nuclear_channel)
    started = time.perf_counter()
    try:
        progress(0.2, desc="Previewing {}".format(session.display_name))
        segment_session(candidate, engine=engine)
    except Exception as exc:
        traceback.print_exc()
        return segment_views(batch, "Preview failed: " + error_text(exc)) + (gr.update(),)
    seconds = round(time.perf_counter() - started, 2)
    session.view_cache["candidate"] = candidate
    table = comparison_table(metrics(session), metrics(candidate, seconds))
    message = ("**Preview of {}** with these settings. Nothing has changed yet: keep it to "
               "replace this image's segmentation, or discard it.".format(session.display_name))
    views = list(segment_views(batch, message, comparison=table))
    views[1] = _overlay(candidate)
    return (*views, gr.update(visible=True))


@locked
def on_keep_preview(batch):
    session = active(batch)
    candidate = session.view_cache.pop("candidate", None) if session else None
    if candidate is None:
        return segment_views(batch, "There is no preview to keep.") + (gr.update(visible=False),)
    for name in ("raw_labels", "object_labels", "objects", "annotation", "engine_info",
                 "segmented_channel", "segmented_with", "segmentation_params",
                 "preprocess_params", "split_params", "cluster_params", "qc_params", "qc"):
        setattr(session, name, getattr(candidate, name))
    session.segmentation_version += candidate.segmentation_version + 1
    session.results_cache = None
    session.geometry_cache = None
    session.reviewed = False
    session.undo_stack.clear()
    session.redo_stack.clear()
    session.correction_points.clear()
    session.selected_object = 0
    session.error = ""
    return segment_views(batch, "Kept the preview. " + describe_run(session)) + (
        gr.update(visible=False),)


@locked
def on_discard_preview(batch):
    session = active(batch)
    if session is not None:
        session.view_cache.pop("candidate", None)
    return segment_views(batch, "Preview discarded; nothing changed.") + (gr.update(visible=False),)


@locked
def on_apply_to_batch(*args):
    """Make these the batch settings, without running anything."""
    *widgets, batch = args
    if batch is None or batch.is_empty:
        return segment_views(batch, "Add images first.")
    params = collect_params(*widgets)
    push_params(batch, params, to_all=True)
    stale = [s for s in batch.images if is_stale(s)]
    message = "These settings are now the batch defaults."
    if stale:
        message += (" **{} segmented image(s) are now stale**: run the batch to re-segment "
                    "them, which clears their review.".format(len(stale)))
    return segment_views(batch, message)


@locked
def on_apply_rules(border, ruler, batch):
    """Update automatic exclusions everywhere, without re-segmenting."""
    if batch is None or batch.is_empty:
        return segment_views(batch, "Add images first.")
    rules = QCParams(exclude_border=bool(border), exclude_annotations=bool(ruler))
    batch.qc_params = rules
    excluded = restored = 0
    for session in batch.images:
        session.qc_params = rules
        counts = reapply_automatic_qc(session)
        excluded += counts["excluded"]
        restored += counts["restored"]
    return segment_views(
        batch,
        "Automatic rules applied: {} object(s) newly excluded, {} restored. Manual "
        "decisions were kept.".format(excluded, restored),
    )


def _run(batch, engine, targets_filter, widgets=None, all_images=False, progress=None):
    """Shared generator for batch runs."""
    if batch is None or batch.is_empty:
        yield segment_views(batch, "Add images on the Import tab first.")
        return
    if widgets is not None:
        with batch.lock():
            push_params(batch, collect_params(*widgets), to_all=True)
    view = copy.copy(batch)
    view.__dict__["_lock"] = batch.lock()        # commits must share the interface's lock
    view.images = [s for s in batch.images if targets_filter(s)]
    if all_images:
        for session in view.images:
            session.error = ""
    if not view.images:
        yield segment_views(batch, "Nothing to run: every image is segmented with the current "
                                   "settings. Choose *All images* to re-run them anyway.")
        return
    cancel = threading.Event()
    report = None
    try:
        for position, total, session, report in iter_batch_segmentation(
            view, engine=engine, only_missing=False, cancel=cancel,
        ):
            if progress is not None:
                progress((position + 1) / max(total, 1),
                         desc="{} ({}/{})".format(session.display_name, position + 1, total))
            with batch.lock():
                batch.active_index = batch.images.index(session)
            from .projects_tab import AUTOSAVER

            AUTOSAVER.request(batch)
            eta = report.get("eta_seconds") or 0
            message = "Segmented **{} of {}**. {} per image; about {:.0f} s left.".format(
                report["succeeded"] + len(report["failed"]), total,
                "{:.1f} s".format(report["seconds_per_image"] or 0), eta)
            if report["failed"]:
                message += "\n\n**Failed:** " + "; ".join(
                    "{} ({})".format(name, error) for name, error in report["failed"])
            yield segment_views(batch, message, report)
    finally:
        cancel.set()
    if report is None:
        return
    summary = "Finished: **{} of {}** image(s) segmented.".format(report["succeeded"], report["total"])
    if report["failed"]:
        summary += "\n\n**Failed:** " + "; ".join(
            "{} ({})".format(name, error) for name, error in report["failed"])
    session = active(batch)
    if session is not None and session.has_segmentation:
        summary += "\n\n" + describe_run(session)
    summary += channel_divergence_warning(batch)
    yield segment_views(batch, summary, report)


def on_segment_all(*args, progress=gr.Progress()):
    """Segment the batch, streaming each result as it lands."""
    *widgets, engine, scope, batch = args
    all_images = scope == RUN_SCOPES[1]

    def needed(session):
        return all_images or session.error or not session.has_segmentation or is_stale(session)

    yield from _run(batch, engine, needed, widgets, all_images, progress)


def on_retry_failed(engine, batch, progress=gr.Progress()):
    yield from _run(batch, engine, lambda s: bool(s.error), progress=progress)


def refresh_segment_view(batch):
    return segment_views(batch, "")


# --------------------------------------------------------------------------- #
# Layout
# --------------------------------------------------------------------------- #


def build(batch, shared) -> SimpleNamespace:
    engine_choices = ["auto", "threshold_watershed"]
    if cellpose_available():
        engine_choices.insert(0, "cellpose")
    c = SimpleNamespace()
    with gr.Tab("Segment", id="tab_segment") as c.tab:
        c.nav = gr.HTML()
        with gr.Row(equal_height=False):
            with gr.Column(scale=1, min_width=330):
                gr.HTML(section("Settings"))
                with gr.Row():
                    c.diameter = gr.Number(value=0, label="Cell diameter (px)",
                                           info="0 = estimate automatically", minimum=0)
                    c.min_area = gr.Number(value=200, label="Minimum object area (px)",
                                           minimum=1, precision=0)
                c.cellprob = gr.Slider(-6, 6, value=0.0, step=0.1, label="Cell probability threshold",
                                       info="Higher keeps only confident objects.")
                c.flow = gr.Slider(0.0, 3.0, value=0.4, step=0.05, label="Flow threshold",
                                   info="Higher accepts less regular shapes.")
                c.engine = gr.Radio(engine_choices, value="auto", label="Engine",
                                    info="The threshold engine is always labelled as such in exports.")
                with gr.Accordion("Automatic exclusions", open=False):
                    gr.Markdown("Excluded objects stay in the masks, are logged with their "
                                "reason, and can be restored one by one on Review.",
                                elem_classes="cs-note")
                    c.exclude_border = gr.Checkbox(value=True, label="Exclude cells touching the image edge")
                    c.exclude_ruler = gr.Checkbox(
                        value=True, label="Exclude cells touching a burned-in scale bar",
                        info="Yellow or white bars and their captions drawn on the image.")
                    c.apply_rules = gr.Button("Apply rules to all images", size="sm")
                with gr.Accordion("Speed and hardware", open=False):
                    c.analysis_scale = gr.Slider(
                        0.25, 1.0, value=1.0, step=0.05, label="Analysis resolution",
                        info="Segments a downscaled copy; masks return at full size. On the "
                             "reference field 0.75x was 1.36x faster at F1 0.971. Recorded in exports.")
                    c.model = gr.Textbox(value="cpsam", label="Cellpose model")
                    c.use_gpu = gr.Checkbox(value=True, label="Use the GPU when available")
                with gr.Accordion("Preprocessing", open=False):
                    gr.Markdown("Affects segmentation only. Measurements always use the original "
                                "image.", elem_classes="cs-note")
                    c.normalize = gr.Checkbox(value=True, label="Normalise contrast")
                    c.gaussian_sigma = gr.Number(value=1.0, label="Gaussian sigma (px)", minimum=0)
                    c.background_subtract = gr.Checkbox(value=False, label="Subtract background")
                    c.background_radius = gr.Number(value=50.0, label="Background radius (px)", minimum=1)
                with gr.Accordion("Touching cells and clusters", open=False):
                    gr.Markdown("A split is accepted only if every gate passes; anything "
                                "ambiguous stays whole as an unresolved group.", elem_classes="cs-note")
                    c.split_enabled = gr.Checkbox(value=True, label="Attempt splitting")
                    c.saddle_depth = gr.Slider(0.05, 0.8, value=0.25, step=0.05,
                                               label="Minimum neck depth", info="Higher refuses more splits.")
                    c.solidity_max = gr.Slider(0.5, 1.0, value=0.95, step=0.01, label="Flag below solidity")
                    c.min_fragment_frac = gr.Slider(0.02, 0.5, value=0.15, step=0.01,
                                                    label="Minimum fragment (fraction of parent)")
                    c.contact_distance = gr.Slider(0, 10, value=2, step=1, label="Contact distance (px)",
                                                   info="Objects this close count as touching.")
                c.settings_note = gr.Markdown(elem_classes="cs-note")

            with gr.Column(scale=2, min_width=440):
                gr.HTML(section("Try on this image"))
                with gr.Row():
                    c.preview_button = gr.Button("Preview on this image", variant="primary")
                    c.run_one = gr.Button("Segment this image now")
                    c.apply_batch = gr.Button("Use these settings for the batch")
                with gr.Row(visible=False) as c.preview_actions:
                    c.keep = gr.Button("Keep preview", variant="primary", size="sm")
                    c.discard = gr.Button("Discard preview", size="sm")
                c.preview = gr.Image(label="Segmentation", type="numpy", interactive=False,
                                     show_download_button=True)
                gr.HTML(legend_html())
                c.comparison = gr.Dataframe(
                    headers=["Measure", "Current", "Preview", "Change"], interactive=False,
                    label="Before and after", wrap=True)
                c.note = gr.Markdown(elem_classes="cs-note")

                gr.HTML(section("Run the batch"))
                with gr.Row():
                    c.scope = gr.Radio(list(RUN_SCOPES), value=RUN_SCOPES[0], label="Which images", scale=3)
                with gr.Row():
                    c.run_all = gr.Button("Run batch", variant="primary")
                    c.retry = gr.Button("Retry failed images")
                    c.cancel = gr.Button("Cancel", variant="stop")
                gr.Markdown("Cancelling finishes the image on the GPU, then stops. Finished "
                            "images keep their results.", elem_classes="cs-note")
                c.progress = gr.Dataframe(headers=PROGRESS_COLUMNS, interactive=False,
                                          label="Progress", wrap=True)
    c.param_widgets = [
        c.model, c.diameter, c.cellprob, c.flow, c.min_area, c.use_gpu,
        c.background_subtract, c.background_radius, c.gaussian_sigma, c.normalize,
        c.split_enabled, c.saddle_depth, c.solidity_max, c.min_fragment_frac, c.contact_distance,
        shared.nuclear_channel, c.analysis_scale, c.exclude_border, c.exclude_ruler,
    ]
    return c


def wire(c, batch, shared) -> None:
    outputs = [batch, c.preview, c.note, shared.image_grid, c.nav, shared.review_strip,
               shared.status, c.progress, c.comparison]
    shared.segment_outputs = outputs
    widgets = c.param_widgets
    run_one = c.run_one.click(on_segment_one, widgets + [c.engine, batch], outputs,
                              show_progress_on=[c.preview])
    run_all = c.run_all.click(on_segment_all, widgets + [c.engine, c.scope, batch], outputs,
                              show_progress_on=[c.progress])
    retry = c.retry.click(on_retry_failed, [c.engine, batch], outputs, show_progress_on=[c.progress])
    preview = c.preview_button.click(on_preview, widgets + [c.engine, batch],
                                     outputs + [c.preview_actions], show_progress_on=[c.preview])
    # Every run, not just the last one wired.
    c.cancel.click(None, cancels=[run_one, run_all, retry, preview], queue=False)
    c.keep.click(on_keep_preview, [batch], outputs + [c.preview_actions])
    c.discard.click(on_discard_preview, [batch], outputs + [c.preview_actions])
    c.apply_batch.click(on_apply_to_batch, widgets + [batch], outputs)
    c.apply_rules.click(on_apply_rules, [c.exclude_border, c.exclude_ruler, batch], outputs)
    for widget in widgets:
        if widget is shared.nuclear_channel:
            continue
        widget.change(on_settings_changed, widgets + [batch], [c.settings_note],
                      show_progress="hidden", trigger_mode="always_last")
    c.tab.select(refresh_segment_view, [batch], outputs, show_progress="hidden")
    shared.run_events = [run_one, run_all, retry, preview]
