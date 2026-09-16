"""Projects tab: save, open and recover projects; analysis presets; autosave."""

from __future__ import annotations

import json
import os
import re
import tempfile
from pathlib import Path
from types import SimpleNamespace

import gradio as gr

from src.autosave import Autosaver
from src.autosave import recovery_path as _recovery_path
from src.project import apply_preset, load_project, preset_dict, save_project

from .theme import section

#: Where recovery files live. Local to this computer and excluded from Git.
PROJECTS = Path(os.environ.get("CELLSCOPE_PROJECTS", Path(__file__).resolve().parents[2] / "projects"))
AUTOSAVER = Autosaver(PROJECTS)


def recovery_path(batch):
    """Derived from a hash of the batch ID, never from user-supplied text."""
    return _recovery_path(PROJECTS, batch)


def autosave(batch, wait=False):
    """Save in the background if anything changed; returns a status line."""
    return AUTOSAVER.request(batch, wait=wait)


def recovery_choices():
    return [("{} · {} image(s) · {}".format(r["label"], r["n_images"], r["saved"]), r["name"])
            for r in AUTOSAVER.recoveries()]


def on_save(batch):
    try:
        if not batch or batch.is_empty:
            raise ValueError("Add images first.")
        folder = Path(tempfile.mkdtemp(prefix="cellscope_project_"))
        name = re.sub(r"[^\w-]", "_", batch.label or "project")[:80] or "project"
        path = save_project(batch, folder / (name + ".cellscope"))
        save_project(batch, recovery_path(batch), compression_level=1)
        AUTOSAVER.mark_clean(batch, str(recovery_path(batch)))
        return path, "Project saved. Download it for a portable copy.", AUTOSAVER.status(batch)
    except Exception as exc:
        return None, "Save failed: " + str(exc), AUTOSAVER.status(batch)


def open_project(path, batch):
    """Open a project, keeping the current work recoverable first."""
    if not path:
        return batch, "Choose a project first."
    try:
        notes: list[str] = []
        loaded = load_project(path, notes)
    except Exception as exc:
        return batch, "Could not open the project: " + str(exc)
    if batch is not None and not batch.is_empty:
        autosave(batch, wait=True)
    AUTOSAVER.mark_clean(loaded)
    message = "Opened **{}** ({} image(s)).".format(loaded.label or Path(path).name, len(loaded.images))
    return loaded, message + "".join("  \n" + n for n in notes)


def on_open(path, batch):
    return open_project(path, batch)


def on_recover(name, batch):
    if not name or Path(name).name != name:
        return batch, "Select a recovery file."
    return open_project(PROJECTS / name, batch)


def on_tick(enabled, batch):
    if not enabled:
        return "Autosave is off. Save the project manually."
    try:
        return autosave(batch)
    except Exception as exc:
        return "Autosave failed: {}. Save the project manually.".format(exc)


def on_save_preset(name, batch):
    folder = Path(tempfile.mkdtemp(prefix="cellscope_preset_"))
    path = folder / ((re.sub(r"[^\w-]", "_", name or "preset")[:80] or "preset") + ".json")
    data = preset_dict(batch)
    data["name"] = name
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")
    return str(path)


def on_load_preset(path, batch):
    try:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        with batch.lock():
            apply_preset(batch, data)
        return batch, ("Preset applied as the batch settings. Segmented images whose settings "
                       "changed are marked stale; nothing was re-run.")
    except Exception as exc:
        return batch, "Preset failed: " + str(exc)


def build(batch, shared) -> SimpleNamespace:
    c = SimpleNamespace()
    with gr.Tab("Projects", id="tab_projects") as c.tab:
        gr.HTML(section("Save and open"))
        gr.Markdown(
            "A `.cellscope` project holds the images, masks, QC decisions, corrections and "
            "settings. Recovery copies stay on this computer in `projects/` and are never "
            "committed to Git.", elem_classes="cs-note")
        with gr.Row():
            with gr.Column():
                c.save = gr.Button("Save project", variant="primary")
                c.download = gr.File(label="Project download")
            with gr.Column():
                c.upload = gr.File(label="Open a project", file_types=[".cellscope"], type="filepath")
                c.open = gr.Button("Open this project")
        c.note = gr.Markdown(elem_classes="cs-note")
        gr.HTML(section("Recovery"))
        with gr.Row():
            c.recoveries = gr.Dropdown(label="Autosaved projects", choices=recovery_choices(), scale=3)
            c.scan = gr.Button("Refresh list", size="sm", scale=0)
            c.recover = gr.Button("Open autosave", scale=0)
        with gr.Row():
            c.auto = gr.Checkbox(value=True, label="Autosave when something changes")
            c.auto_note = gr.Markdown(elem_classes="cs-note")
        with gr.Accordion("Analysis presets", open=False):
            gr.Markdown(
                "A preset holds the channel, segmentation, splitting, clustering and automatic "
                "exclusion settings, but never the calibration.", elem_classes="cs-note")
            with gr.Row():
                c.preset_name = gr.Textbox(label="Preset name", value="My analysis")
                c.save_preset = gr.Button("Download preset")
            c.preset_download = gr.File(label="Preset")
            c.preset_upload = gr.File(label="Load a preset", file_types=[".json"], type="filepath")
            c.load_preset = gr.Button("Apply preset to the batch")
            c.preset_note = gr.Markdown(elem_classes="cs-note")
    return c


def wire(c, batch, shared, interval: float) -> None:
    c.timer = gr.Timer(interval)
    c.timer.tick(on_tick, [c.auto, batch], [c.auto_note], show_progress="hidden")
    c.save.click(on_save, [batch], [c.download, c.note, c.auto_note])
    c.scan.click(lambda: gr.update(choices=recovery_choices()), outputs=c.recoveries)
    shared.opened = []
    for button, fn, source in ((c.open, on_open, c.upload), (c.recover, on_recover, c.recoveries)):
        shared.opened.append(button.click(fn, [source, batch], [batch, c.note]))
    c.save_preset.click(on_save_preset, [c.preset_name, batch], c.preset_download)
    shared.preset_loaded = c.load_preset.click(on_load_preset, [c.preset_upload, batch],
                                               [batch, c.preset_note])
    c.tab.select(lambda b: (gr.update(choices=recovery_choices()), AUTOSAVER.status(b)),
                 [batch], [c.recoveries, c.auto_note], show_progress="hidden")
