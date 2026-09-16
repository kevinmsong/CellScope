"""What each user gesture costs on the server, for ``tools/bench.py``.

Zoom, pan and overlay opacity are handled in the browser, so they have no
server cost at all; they are listed so the benchmark table stays comparable
with the baseline.
"""

from __future__ import annotations

import json

from src.review import history

from . import results_tab, review_tab
from .views import status_html


def gestures(click_event):
    def payload(batch):
        x, y = click_event(batch).index[:2]
        return json.dumps({"x": x, "y": y})

    def next_image(batch):
        return review_tab.on_review_next(True, False, batch)

    def client_side(batch):
        return None

    def inspect_click(batch):
        batch.active.review_mode = "Inspect"
        return review_tab.on_canvas_click(payload(batch), True, False, batch)

    def exclude_click(batch):
        batch.active.review_mode = "Toggle inclusion"
        return review_tab.on_canvas_click(payload(batch), True, False, batch)

    def undo(batch):
        history(batch.active)
        return review_tab.on_toggle_labels(True, False, batch)

    return {
        "next_image": next_image,
        "opacity_change": client_side,
        "zoom_change": client_side,
        "inspect_click": inspect_click,
        "exclude_click": exclude_click,
        "undo": undo,
        "results_tab": results_tab.on_refresh_results,
        "status_strip": status_html,
    }
