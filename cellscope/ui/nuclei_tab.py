"""DAPI tab: independent nuclear counts with reversible exclusions.

Nuclear counts are segmentation estimates of nuclei, not cytoplasmic cell counts.
"""

from __future__ import annotations

import tempfile
from types import SimpleNamespace

import gradio as gr
import numpy as np
from PIL import Image

from src.clustering import assign_clusters
from src.image_io import extract_channel
from src.nuclei import (
    batch_nuclear_counts,
    export_nuclei,
    nuclear_counts,
    nuclear_tables,
    segment_nuclei,
)
from src.types import ClusterRecord, ObjectRecord, utc_now
from src.visualize import _resize_labels, make_overlay

from .views import locked


def _display_shape(shape):
    dummy = Image.new("L", (shape[1], shape[0]))
    dummy.thumbnail((1024, 1024))
    return dummy.height, dummy.width


def _nuclear_base(s):
    """Greyscale display of the nuclear signal, cached per source and channel."""
    settings = s.nuclei_settings
    key = ("nuclear_base", settings.get("source_sha256"), settings.get("channel"))
    if s.view_cache.get("nuclear_base_key") != key:
        source = s.nuclear_image if settings["source"] == "Separate nuclear file" else s.image
        signal = extract_channel(source, settings["channel"])
        lo, hi = float(signal.min()), float(signal.max())
        grey = np.clip((signal - lo) / max(hi - lo, 1e-12) * 255, 0, 255).astype(np.uint8)
        image = Image.fromarray(np.repeat(grey[..., None], 3, axis=2))
        image.thumbnail((1024, 1024))
        s.view_cache["nuclear_base"] = np.asarray(image)
        s.view_cache["nuclear_base_key"] = key
    return s.view_cache["nuclear_base"]


def nuclear_view(batch):
    s = batch.active if batch else None
    if s is None or s.nuclei_labels is None:
        return (None, "Run the DAPI analysis for this image.",
                nuclear_tables(s)[0] if s else None,
                batch_nuclear_counts(batch) if batch else None)
    key = (id(s.nuclei_labels), frozenset(s.nuclei_excluded), len(s.nuclei_log))
    if s.view_cache.get("nuclear_view_key") != key:
        table, edges = nuclear_tables(s)
        ids = [int(row.nucleus_id) for row in table.itertuples() if row.included]
        groups = assign_clusters(ids, edges)
        members: dict[int, list[int]] = {}
        for i in ids:
            members.setdefault(groups[i], []).append(i)
        clusters = [ClusterRecord(g, tuple(m), tuple(m), ()) for g, m in sorted(members.items())]
        overlay = make_overlay(_nuclear_base(s), s.nuclei_labels,
                               [ObjectRecord(i, i, "resolved") for i in ids], clusters,
                               excluded_ids=s.nuclei_excluded)
        s.view_cache["nuclear_view"] = (overlay, table)
        s.view_cache["nuclear_view_key"] = key
    overlay, table = s.view_cache["nuclear_view"]
    counts = nuclear_counts(s)
    text = (f"**{s.display_name}**: {counts['nuclei_included']} included nuclei; "
            f"**{counts['touching_nuclei']} touching nuclei in {counts['touching_groups']} groups**; "
            f"{counts['isolated_nuclei']} isolated; {counts['nuclei_excluded']} excluded. "
            "Touching means the masks share an edge or corner. Blue = isolated, orange = "
            "touching, grey = excluded. These are segmentation estimates: check the "
            "boundaries, and click a nucleus to exclude or restore it.")
    summary = batch_nuclear_counts(batch)
    if len(summary) > 1:
        text += (f"  \nAcross {len(summary)} analysed images: {int(summary.nuclei_included.sum())} "
                 f"included nuclei, {int(summary.touching_nuclei.sum())} touching in "
                 f"{int(summary.touching_groups.sum())} groups.")
    return overlay, text, table, summary


def analyze(source, channel, area, seeds, sigma, b):
    try:
        if b.active is None:
            raise ValueError("Load an image first.")
        with b.lock():
            segment_nuclei(b.active, channel, int(area), int(seeds), float(sigma), 0, source)
        from .projects_tab import AUTOSAVER

        AUTOSAVER.request(b)
        return b, *nuclear_view(b)
    except Exception as exc:
        return b, None, "Nuclear analysis failed: " + str(exc), None, batch_nuclear_counts(b)


def analyze_all(source, channel, area, seeds, sigma, b):
    for index in range(len(b.images)):
        b.active_index = index
        yield analyze(source, channel, area, seeds, sigma, b)


@locked
def step(b, delta):
    if b.images:
        b.active_index = (b.active_index + delta) % len(b.images)
    return b, *nuclear_view(b)


@locked
def click(b, evt: gr.SelectData):
    s = b.active
    if s is None or s.nuclei_labels is None:
        return b, *nuclear_view(b)
    shape = _display_shape(s.nuclei_labels.shape)
    key = ("nuclear_small", id(s.nuclei_labels), shape)
    if s.view_cache.get("nuclear_small_key") != key:
        s.view_cache["nuclear_small"] = _resize_labels(s.nuclei_labels, shape)
        s.view_cache["nuclear_small_key"] = key
    labels = s.view_cache["nuclear_small"]
    x, y = map(int, evt.index[:2])
    if 0 <= y < labels.shape[0] and 0 <= x < labels.shape[1]:
        oid = int(labels[y, x])
        if oid:
            if oid in s.nuclei_excluded:
                s.nuclei_excluded.remove(oid)
                action = "restore"
            else:
                s.nuclei_excluded.add(oid)
                action = "exclude"
            s.nuclei_log.append({"action": action, "nucleus_id": oid, "timestamp": utc_now()})
    return b, *nuclear_view(b)


def export(b):
    try:
        return export_nuclei(b, tempfile.mkdtemp(prefix="cellscope_nuclei_export_"))
    except ValueError as exc:
        raise gr.Error(str(exc)) from exc


def build(batch, shared) -> SimpleNamespace:
    c = SimpleNamespace()
    with gr.Tab("DAPI nuclei", id="tab_nuclei") as c.tab:
        gr.Markdown(
            "Count nuclei independently of the cell segmentation, from the blue channel or a "
            "separate nuclear file attached on Import. Otsu threshold plus seeded watershed, "
            "with its own settings and exclusions. These counts are not cell counts.",
            elem_classes="cs-note")
        with gr.Row():
            c.previous = gr.Button("◀ Previous image", size="sm")
            c.following = gr.Button("Next image ▶", size="sm")
            c.refresh = gr.Button("Refresh", size="sm")
        with gr.Row(equal_height=False):
            with gr.Column(scale=1, min_width=300):
                c.source = gr.Radio(["Image channel", "Separate nuclear file"], value="Image channel",
                                    label="DAPI source")
                c.channel = gr.Dropdown(["blue", "grayscale", "red", "green"], value="blue",
                                        label="Nuclear channel")
                c.min_area = gr.Number(value=25, minimum=1, precision=0, label="Minimum nuclear area (px)")
                c.seeds = gr.Number(value=5, minimum=1, precision=0, label="Seed separation (px)",
                                    info="Larger values split fewer touching nuclei.")
                c.sigma = gr.Number(value=1, minimum=0, label="Smoothing sigma (px)")
                gr.Markdown("Nuclei touching the image edge are excluded automatically.",
                            elem_classes="cs-note")
                c.run = gr.Button("Count nuclei in this image", variant="primary")
                c.run_all = gr.Button("Count nuclei in all images")
            with gr.Column(scale=2):
                c.image = gr.Image(label="Nuclear masks · click to exclude or restore",
                                   interactive=False, type="numpy")
                c.note = gr.Markdown(elem_classes="cs-note")
                c.table = gr.Dataframe(interactive=False, label="Nuclei", max_height=320)
        c.summary = gr.Dataframe(interactive=False, label="Nuclear counts by image")
        with gr.Row():
            c.download_button = gr.Button("Export nuclear counts and masks")
            c.download = gr.File(label="Nuclear analysis archive")
    return c


def wire(c, batch, shared) -> None:
    outputs = [c.image, c.note, c.table, c.summary]
    inputs = [c.source, c.channel, c.min_area, c.seeds, c.sigma, batch]
    c.download_button.click(export, [batch], [c.download])
    c.run.click(analyze, inputs, [batch] + outputs)
    c.run_all.click(analyze_all, inputs, [batch] + outputs)
    c.refresh.click(nuclear_view, [batch], outputs)
    c.tab.select(nuclear_view, [batch], outputs, show_progress="hidden")
    c.previous.click(lambda b: step(b, -1), [batch], [batch] + outputs, show_progress="hidden")
    c.following.click(lambda b: step(b, 1), [batch], [batch] + outputs, show_progress="hidden")
    c.image.select(click, [batch], [batch] + outputs, show_progress="hidden")
