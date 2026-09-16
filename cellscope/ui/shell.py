"""The application shell: header, workflow stepper, tabs, and cross-tab wiring.

Dependencies point one way: this module imports the tabs; tabs import
``views``, ``theme`` and ``src``; nothing imports this module except the
launcher and the ``app.py`` compatibility shim.
"""

from __future__ import annotations

from types import SimpleNamespace

import gradio as gr

from src import __version__
from src.types import BatchSession

from . import (
    batch_tab,
    export_tab,
    nuclei_tab,
    projects_tab,
    quality_tab,
    results_tab,
    review_tab,
    segment_tab,
)
from .theme import CSS, THEME_JS, build_theme
from .viewer import VIEWER_JS
from .views import status_html

DEFAULT_AUTOSAVE_SECONDS = 30.0


def label_of(batch):
    return batch.label if batch is not None else ""


def build_interface(autosave_interval: float = DEFAULT_AUTOSAVE_SECONDS):
    with gr.Blocks(title="CellScope", theme=build_theme(), css=CSS, head=VIEWER_JS,
                   analytics_enabled=False) as demo:
        batch = gr.State(BatchSession())
        shared = SimpleNamespace()

        with gr.Row(elem_id="cs-header"):
            home = gr.Button("CellScope", elem_id="cs-home", scale=0)
            gr.HTML('<span class="cs-sub">Fluorescence cell morphometry · v{} · '
                    "click the title to start a new batch</span>".format(__version__))
            theme_mode = gr.Radio(["System", "Light", "Dark"], value="System", label="Appearance",
                                  show_label=False, scale=0, min_width=300, elem_id="cs-theme")
        demo.load(None, outputs=[theme_mode], js=THEME_JS)
        theme_mode.input(None, inputs=[theme_mode], outputs=[],
                         js="(mode) => { window.cellscopeTheme(mode); }")
        shared.status = gr.HTML(status_html(None))

        with gr.Tabs() as tabs:
            batch_c = batch_tab.build(batch, shared)
            shared.image_grid = batch_c.grid
            shared.nuclear_channel = batch_c.nuclear_channel
            segment_c = segment_tab.build(batch, shared)
            shared.segment_nav = segment_c.nav
            review_c = review_tab.build(batch, shared)
            quality_c = quality_tab.build(batch, shared)
            results_c = results_tab.build(batch, shared)
            export_c = export_tab.build(batch, shared)
            nuclei_c = nuclei_tab.build(batch, shared)
            projects_c = projects_tab.build(batch, shared)

        batch_tab.wire(batch_c, batch, shared)
        segment_tab.wire(segment_c, batch, shared)
        review_tab.wire(review_c, batch, shared)
        quality_tab.wire(quality_c, batch, shared)
        results_tab.wire(results_c, batch, shared)
        export_tab.wire(export_c, batch, shared)
        nuclei_tab.wire(nuclei_c, batch, shared)
        projects_tab.wire(projects_c, batch, shared, autosave_interval)

        active_outputs = [batch_c.channel, batch_c.nuclear_channel, batch_c.preview, batch_c.channel_note]
        home.click(
            batch_tab.on_home, [batch],
            [batch, batch_c.note, tabs, batch_c.grid, shared.segment_nav, shared.review_strip,
             shared.status] + active_outputs,
            show_progress="hidden",
        )

        # Opening a project or applying a preset replaces settings everywhere.
        settings = [w for w in segment_c.param_widgets if w is not shared.nuclear_channel]

        def settings_of(b):
            target = b.active or b
            values = segment_tab.params_for_widgets(target)
            # params_for_widgets follows param_widgets order, which includes the
            # nuclear channel; that widget follows the active image instead.
            index = segment_c.param_widgets.index(shared.nuclear_channel)
            return values[:index] + values[index + 1:]

        for event in shared.opened:
            (event
             .then(batch_tab.refresh_batch_view, [batch], shared.nav_outputs, show_progress="hidden")
             .then(settings_of, [batch], settings, show_progress="hidden")
             .then(label_of, [batch], [batch_c.label], show_progress="hidden"))
        shared.preset_loaded.then(settings_of, [batch], settings, show_progress="hidden").then(
            batch_tab.refresh_batch_view, [batch], shared.nav_outputs, show_progress="hidden")
    return demo
