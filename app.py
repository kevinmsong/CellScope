"""CellScope -- research-grade fluorescence cell morphometry.

Launch with ``python app.py``, ``python -m cellscope`` or ``cellscope``.

The interface lives in :mod:`cellscope.ui`; this module keeps the historical
entry point and re-exports the callbacks that scripts and tests have used by
name. The scientific code is in ``src/`` and never imports Gradio.
"""

from __future__ import annotations

from cellscope.cli import main, preload_model
from cellscope.ui.batch_tab import (
    CALIBRATION_METHODS,
    RESET_ARMED,
    active_controls,
    on_apply_reference,
    on_apply_resolution,
    on_apply_scale_bar,
    on_batch_label,
    on_channel_change,
    on_clear_points,
    on_detect_bar,
    on_home,
    on_image_click,
    on_next,
    on_nuclear_change,
    on_nuclear_file,
    on_prev,
    on_reference_upload,
    on_remove_image,
    on_select_image,
    on_set_uncalibrated,
    on_table_edited,
    on_upload,
)
from cellscope.ui.export_tab import OUTPUT_DIR, on_export
from cellscope.ui.results_tab import PSEUDOREPLICATION_NOTE, headline_html, on_refresh_results
from cellscope.ui.review_tab import (
    on_canvas_click,
    on_exclude,
    on_exclude_border,
    on_exclude_cluster,
    on_filter_area,
    on_mark_reviewed,
    on_reset_qc,
    on_restore,
    on_review_click,
    on_toggle_labels,
    on_unmark_reviewed,
    review_views,
)
from cellscope.ui.review_tab import parse_ids as _parse_ids
from cellscope.ui.segment_tab import channel_divergence_warning as _channel_divergence_warning
from cellscope.ui.segment_tab import collect_params as _collect_params
from cellscope.ui.segment_tab import describe_run as _describe_run
from cellscope.ui.segment_tab import on_segment_all, on_segment_one
from cellscope.ui.shell import build_interface
from cellscope.ui.views import (
    DASH,
    IMAGE_TABLE_COLUMNS,
    STATUS_FIELDS,
    image_table,
    nav_html,
    status_fields,
    status_html,
    strip_html,
)
from cellscope.ui.views import NO_NUCLEAR as _NO_NUCLEAR
from cellscope.ui.views import active as _active
from cellscope.ui.views import display_base as _display_base

__all__ = [
    "CALIBRATION_METHODS", "DASH", "IMAGE_TABLE_COLUMNS", "OUTPUT_DIR", "PSEUDOREPLICATION_NOTE",
    "RESET_ARMED", "STATUS_FIELDS", "_NO_NUCLEAR", "_active", "_channel_divergence_warning",
    "_collect_params", "_describe_run", "_display_base", "_parse_ids", "active_controls",
    "build_interface", "headline_html", "image_table", "main", "nav_html", "on_apply_reference",
    "on_apply_resolution", "on_apply_scale_bar", "on_batch_label", "on_canvas_click",
    "on_channel_change", "on_clear_points", "on_detect_bar", "on_exclude", "on_exclude_border",
    "on_exclude_cluster", "on_export", "on_filter_area", "on_home", "on_image_click",
    "on_mark_reviewed", "on_next", "on_nuclear_change", "on_nuclear_file", "on_prev",
    "on_reference_upload", "on_refresh_results", "on_remove_image", "on_reset_qc", "on_restore",
    "on_review_click", "on_segment_all", "on_segment_one", "on_select_image",
    "on_set_uncalibrated", "on_table_edited", "on_toggle_labels", "on_unmark_reviewed",
    "on_upload", "preload_model", "review_views", "status_fields", "status_html", "strip_html",
]


if __name__ == "__main__":
    raise SystemExit(main())
