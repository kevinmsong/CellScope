"""Separate DAPI counting tab with reversible nuclear exclusions."""
import gradio as gr
import numpy as np
from src.nuclei import segment_nuclei, nuclear_counts, nuclear_tables, batch_nuclear_counts
from src.image_io import make_display, extract_channel
from src.visualize import make_overlay, _resize_labels
from src.types import ObjectRecord, ClusterRecord, utc_now
from src.clustering import assign_clusters


def nuclear_view(batch):
    s = batch.active if batch else None
    if s is None or s.nuclei_labels is None:
        return None, "Run DAPI analysis for this image.", nuclear_tables(s)[0] if s else None, batch_nuclear_counts(batch) if batch else None
    table, edges = nuclear_tables(s)
    source = s.nuclear_image if s.nuclei_settings["source"] == "Separate nuclear file" else s.image
    signal = extract_channel(source, s.nuclei_settings["channel"])
    lo, hi = signal.min(), signal.max()
    gray = np.clip((signal-lo) / max(float(hi-lo), 1e-12) * 255, 0, 255).astype(np.uint8)
    rgb = np.repeat(gray[..., None], 3, axis=2)
    from PIL import Image
    image = Image.fromarray(rgb)
    image.thumbnail((1024, 1024))
    display = np.asarray(image)
    ids = [int(row.nucleus_id) for row in table.itertuples() if row.included]
    groups = assign_clusters(ids, edges)
    clusters = [ClusterRecord(g, tuple(i for i in ids if groups[i] == g),
                              tuple(i for i in ids if groups[i] == g), ()) for g in sorted(set(groups.values()))]
    overlay = make_overlay(display, s.nuclei_labels, [ObjectRecord(i, i, "resolved") for i in ids], clusters,
                           excluded_ids=s.nuclei_excluded)
    counts = nuclear_counts(s)
    text = (f"**{s.display_name}** — {counts['nuclei_included']} included nuclei; "
            f"**{counts['touching_nuclei']} touching nuclei in {counts['touching_groups']} groups**; "
            f"{counts['isolated_nuclei']} isolated; {counts['nuclei_excluded']} excluded. "
            "Direct contact includes shared edges/corners. Blue = isolated; orange = touching; gray = excluded. "
            "Counts are segmentation estimates: inspect the boundaries. Click nuclei to exclude or restore.")
    summary = batch_nuclear_counts(batch)
    if len(summary) > 1:
        text += (f" Across {len(summary)} analyzed images: {int(summary.nuclei_included.sum())} included nuclei, "
                 f"{int(summary.touching_nuclei.sum())} touching nuclei in {int(summary.touching_groups.sum())} groups.")
    return overlay, text, table, summary


def build_nuclei_tab(batch):
    with gr.Tab("DAPI nuclei") as tab:
        gr.Markdown("Count nuclei independently of cytoplasmic cells. The blue channel is the default; a separate DAPI file attached on Batch can also be used. This analysis uses Otsu thresholding and seeded watershed, with its own settings and exclusions.")
        with gr.Row():
            previous = gr.Button("Previous image")
            following = gr.Button("Next image")
            refresh = gr.Button("Refresh active image")
        with gr.Row():
            with gr.Column(scale=1):
                source = gr.Radio(["Image channel", "Separate nuclear file"], value="Image channel", label="DAPI source")
                channel = gr.Dropdown(["blue", "grayscale", "red", "green"], value="blue", label="Nuclear channel")
                min_area = gr.Number(value=25, minimum=1, precision=0, label="Minimum nuclear area (px)")
                seeds = gr.Number(value=5, minimum=1, precision=0, label="Seed separation (px)")
                sigma = gr.Number(value=1, minimum=0, label="Smoothing sigma (px)")
                gr.Markdown("Touching = direct 8-neighbour mask contact, with no gap allowance. Edge-touching nuclei are excluded automatically. Seed separation affects the estimated number of nuclei in a connected DAPI region.")
                run = gr.Button("Count nuclei in active image", variant="primary")
                run_all = gr.Button("Count nuclei in all images")
            with gr.Column(scale=2):
                image = gr.Image(label="DAPI nuclear masks — click to exclude / restore", interactive=False, type="numpy")
                note = gr.Markdown()
                table = gr.Dataframe(interactive=False, label="Nucleus measurements")
        summary = gr.Dataframe(interactive=False, label="Separate nuclear counts by image")
        download_button = gr.Button("Export nuclear counts and masks")
        download = gr.File(label="Nuclear analysis archive")
        def export(b):
            import tempfile
            from src.nuclei import export_nuclei
            try:
                return export_nuclei(b, tempfile.mkdtemp(prefix="cellscope_nuclei_export_"))
            except ValueError as exc:
                raise gr.Error(str(exc)) from exc
        download_button.click(export, [batch], [download])
        outputs = [image, note, table, summary]
        def analyze(source, channel, area, seeds, sigma, b):
            try:
                if b.active is None:
                    raise ValueError("Load an image first.")
                segment_nuclei(b.active, channel, int(area), int(seeds), float(sigma), 0, source)
                from professional_ui import autosave
                autosave(b)
                return b, *nuclear_view(b)
            except Exception as exc:
                return b, None, "Nuclear analysis failed: " + str(exc), None, batch_nuclear_counts(b)
        inputs = [source, channel, min_area, seeds, sigma, batch]
        run.click(analyze, inputs, [batch] + outputs)
        def all_images(source, channel, area, seeds, sigma, b):
            for index in range(len(b.images)):
                b.active_index = index
                yield analyze(source, channel, area, seeds, sigma, b)
        run_all.click(all_images, inputs, [batch] + outputs)
        refresh.click(nuclear_view, [batch], outputs)
        tab.select(nuclear_view, [batch], outputs)
        def prev(b):
            if b.images:
                b.active_index = (b.active_index-1) % len(b.images)
            return b, *nuclear_view(b)
        def nxt(b):
            if b.images:
                b.active_index = (b.active_index+1) % len(b.images)
            return b, *nuclear_view(b)
        previous.click(prev, [batch], [batch]+outputs)
        following.click(nxt, [batch], [batch]+outputs)
        def click(b, evt: gr.SelectData):
            s = b.active
            if s is None or s.nuclei_labels is None:
                return b, *nuclear_view(b)
            # Match precisely the nearest-neighbour label grid used by the overlay.
            h, w = s.nuclei_labels.shape
            from PIL import Image
            dummy = Image.new("L", (w, h))
            dummy.thumbnail((1024, 1024))
            labels = _resize_labels(s.nuclei_labels, (dummy.height, dummy.width))
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
        image.select(click, [batch], [batch]+outputs, show_progress="hidden")
