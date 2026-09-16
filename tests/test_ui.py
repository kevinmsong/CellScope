"""The restructured interface: module boundaries, handler contracts, and the review viewer."""

from __future__ import annotations

import ast
import json
from pathlib import Path

import numpy as np
import pytest

gradio = pytest.importorskip("gradio")

from cellscope.ui import (
    batch_tab,
    export_tab,
    projects_tab,
    quality_tab,
    results_tab,
    review_tab,
    segment_tab,
    shell,
    viewer,
    views,
)
from src.perf import image_record, synthetic_field
from src.pipeline import compute_results, run_segmentation
from src.types import AnalysisSession, BatchSession

ROOT = Path(__file__).resolve().parents[1]


def _imports(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    names = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
            names.add(node.module)
    return names


def test_science_package_never_imports_gradio():
    for path in (ROOT / "src").glob("*.py"):
        assert not any(n.split(".")[0] == "gradio" for n in _imports(path)), path.name


def test_interface_modules_never_import_the_app_shim():
    for path in (ROOT / "cellscope").rglob("*.py"):
        imported = _imports(path)
        assert "app" not in imported and "professional_ui" not in imported, path.name


def test_tabs_do_not_import_the_shell():
    for path in (ROOT / "cellscope" / "ui").glob("*_tab.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        relative = {n.module for n in ast.walk(tree) if isinstance(n, ast.ImportFrom) and n.level}
        assert "shell" not in relative, path.name


# --------------------------------------------------------------------------- #


@pytest.fixture
def batch():
    b = BatchSession(label="ui")
    b.channel = "red"
    for index in range(3):
        rgb = synthetic_field(300, 300, 8, seed=index, ruler=False)
        s = AnalysisSession()
        s.image = image_record(rgb, "f{}.png".format(index))
        from src.image_io import make_display

        s.display_rgb, s.display_scale = make_display(s.image)
        b.add(s)
        b.apply_channels(s)
    for s in b.images[:2]:
        run_segmentation(s, s.image.pixels[..., 0].astype(np.float32), engine="threshold_watershed")
    return b


@pytest.fixture(scope="module")
def demo():
    return shell.build_interface()


def _dependencies(demo):
    fns = demo.fns
    return list(fns.values() if isinstance(fns, dict) else fns)


def _wired(demo, fn):
    name = fn.__name__
    found = [d for d in _dependencies(demo) if getattr(d.fn, "__name__", "") == name]
    assert found, "no event wired to " + name
    return found


def _widget_defaults():
    return segment_tab.params_for_widgets(BatchSession())


REVIEW_CALLS = [
    (review_tab.on_toggle_labels, lambda b: (True, False, b)),
    (review_tab.on_canvas_click, lambda b: ('{"x": 5, "y": 5}', True, False, b)),
    (review_tab.on_mode, lambda b: ("Inspect", True, False, b)),
    (review_tab.on_review_prev, lambda b: (True, False, b)),
    (review_tab.on_review_next, lambda b: (True, False, b)),
    (review_tab.on_next_unreviewed, lambda b: (True, False, b)),
    (review_tab.on_toggle_selected, lambda b: (True, False, b)),
    (review_tab.on_select_cell, lambda b: (1, True, False, b)),
    (review_tab.on_mark_reviewed, lambda b: ("", True, False, b)),
    (review_tab.on_unmark_reviewed, lambda b: (True, False, b)),
    (review_tab.on_undo, lambda b: (True, False, b)),
    (review_tab.on_redo, lambda b: (True, False, b)),
    (review_tab.on_reset_qc, lambda b: (True, False, b)),
    (review_tab.on_exclude, lambda b: ("1", "test", True, False, b)),
    (review_tab.on_restore, lambda b: ("1", True, False, b)),
    (review_tab.on_exclude_cluster, lambda b: (1, True, False, b)),
    (review_tab.on_exclude_border, lambda b: (True, False, b)),
    (review_tab.on_filter_area, lambda b: (10, 0, True, False, b)),
    (review_tab.on_correct, lambda b: ("Merge", "1 2", "[]", True, False, b)),
    (review_tab.on_clear_correction_points, lambda b: (True, False, b)),
]


@pytest.mark.parametrize("handler,args", REVIEW_CALLS, ids=lambda v: getattr(v, "__name__", ""))
def test_review_handlers_match_their_outputs(demo, batch, handler, args):
    for active in (0, 2):                     # a segmented and an unsegmented image
        batch.active_index = active
        returned = handler(*args(batch))
        for dependency in _wired(demo, handler):
            assert len(returned) == len(dependency.outputs), handler.__name__


def test_segment_handlers_match_their_outputs(demo, batch):
    widgets = _widget_defaults()
    calls = [
        (segment_tab.on_segment_one, (*widgets, "threshold_watershed", batch)),
        (segment_tab.on_apply_to_batch, (*widgets, batch)),
        (segment_tab.on_apply_rules, (True, True, batch)),
        (segment_tab.refresh_segment_view, (batch,)),
    ]
    for handler, args in calls:
        returned = handler(*args)
        for dependency in _wired(demo, handler):
            assert len(returned) == len(dependency.outputs), handler.__name__
    for handler, args in [
        (segment_tab.on_preview, (*widgets, "threshold_watershed", batch)),
        (segment_tab.on_discard_preview, (batch,)),
    ]:
        returned = handler(*args)
        for dependency in _wired(demo, handler):
            assert len(returned) == len(dependency.outputs), handler.__name__
    runs = list(segment_tab.on_segment_all(*widgets, "threshold_watershed",
                                           segment_tab.RUN_SCOPES[0], batch))
    for dependency in _wired(demo, segment_tab.on_segment_all):
        assert all(len(r) == len(dependency.outputs) for r in runs)


def test_other_tab_handlers_match_their_outputs(demo, batch):
    for handler, args in [
        (quality_tab.on_quality_view, ("area", batch)),
        (quality_tab.on_save_design, (quality_tab.design_table(batch), "area", batch)),
        (results_tab.on_refresh_results, (batch,)),
        (export_tab.on_export, (batch,)),
        (projects_tab.on_save, (batch,)),
        (batch_tab.refresh_batch_view, (batch,)),
    ]:
        returned = handler(*args)
        for dependency in _wired(demo, handler):
            assert len(returned) == len(dependency.outputs), handler.__name__


def test_keyboard_targets_and_bridge_exist(demo):
    config = demo.get_config_file()
    ids = {c.get("props", {}).get("elem_id") for c in config["components"]}
    for element in ("cs-prev", "cs-next", "cs-undo", "cs-redo", "cs-toggle-selected",
                    "cs-mark-reviewed", "cs-mode", "cs-bridge", "cs-review-viewer"):
        assert element in ids, element
    bridge = next(c for c in config["components"] if c.get("props", {}).get("elem_id") == "cs-bridge")
    assert bridge["props"].get("visible") == "hidden"


def test_cancel_stops_every_segmentation_run(demo):
    config = demo.get_config_file()
    names = {d["id"]: d.get("api_name") for d in config["dependencies"]}
    groups = [{names[i] for i in d["cancels"]} for d in config["dependencies"] if d.get("cancels")]
    assert any({"on_segment_one", "on_segment_all", "on_retry_failed", "on_preview"} <= g
               for g in groups)


# --------------------------------------------------------------------------- #
# Viewer
# --------------------------------------------------------------------------- #


def test_viewer_carries_layers_state_and_hover_data(batch):
    session = batch.images[0]
    html = viewer.viewer_html(session)
    assert html.count("data:image/png;base64,") == 3          # fill, lines, labels
    assert "data:image/jpeg;base64," in html                 # the display image
    state = json.loads(__import__("html").unescape(html.split('data-state="')[1].split('"')[0]))
    assert state["w"] == session.display_rgb.shape[1]
    assert state["key"] == session.analysis_id
    objects = json.loads(__import__("html").unescape(html.split('data-objects="')[1].split('"')[0]))
    assert set(objects) == {str(o.object_id) for o in session.objects}


def test_viewer_rendering_is_cached_until_results_change(batch, monkeypatch):
    session = batch.images[0]
    calls = []
    real = viewer.overlay_layers
    monkeypatch.setattr(viewer, "overlay_layers", lambda *a, **k: calls.append(1) or real(*a, **k))
    viewer.viewer_html(session)
    viewer.viewer_html(session)
    assert len(calls) == 1
    target = next(o.object_id for o in session.objects if session.qc.is_included(o.object_id))
    session.qc.exclude(target, "test")
    viewer.viewer_html(session)
    assert len(calls) == 2


def test_label_map_round_trips_ids():
    import base64
    import io

    from PIL import Image

    labels = np.array([[0, 1, 255], [256, 300, 65535]], np.int32)
    uri = viewer._label_png(labels)
    image = np.asarray(Image.open(io.BytesIO(base64.b64decode(uri.split(",", 1)[1]))))
    decoded = image[..., 0].astype(int) + 256 * image[..., 1].astype(int)
    np.testing.assert_array_equal(decoded, labels)


def test_unsegmented_image_still_shows(batch):
    html = viewer.viewer_html(batch.images[2], message="Not segmented yet.")
    assert "data:image/jpeg" in html and "Not segmented yet" in html
    assert "cs-labels" not in html


# --------------------------------------------------------------------------- #
# Review behaviour
# --------------------------------------------------------------------------- #


def _point_of(session, object_id):
    rows, cols = np.nonzero(session.object_labels == object_id)
    scale = session.display_scale
    return json.dumps({"x": int(cols[len(cols) // 2] / scale), "y": int(rows[len(rows) // 2] / scale)})


def test_inspect_selects_without_changing_qc(batch):
    session = batch.images[0]
    target = next(o.object_id for o in session.objects if session.qc.is_included(o.object_id))
    session.review_mode = "Inspect"
    before = session.qc.version
    out = review_tab.on_canvas_click(_point_of(session, target), True, False, batch)
    assert session.selected_object == target
    assert session.qc.version == before
    assert "Cell {}".format(target) in out[9]


def test_x_toggles_the_selected_cell_and_undo_restores_it(batch):
    session = batch.images[0]
    target = next(o.object_id for o in session.objects if session.qc.is_included(o.object_id))
    review_tab.on_select_cell(target, True, False, batch)
    review_tab.on_toggle_selected(True, False, batch)
    assert target in session.qc.excluded_ids
    review_tab.on_undo(True, False, batch)
    assert target not in session.qc.excluded_ids


def test_table_selection_uses_the_cell_id_not_the_row(batch):
    session = batch.images[0]
    target = session.objects[-1].object_id
    event = gradio.SelectData(None, {"index": [0, 0], "value": target,
                                     "row_value": [target, True, "cell", ""]})
    review_tab.on_table_select(True, False, batch, event)
    assert session.selected_object == target


def test_selected_row_is_highlighted(batch):
    session = batch.images[0]
    session.selected_object = session.objects[0].object_id
    table = review_tab.review_views(batch)[8]
    assert hasattr(table, "to_html") and "background-color" in table.to_html()


def test_correction_defaults_to_the_selected_cell(batch):
    session = batch.images[0]
    target = session.objects[0].object_id
    rows, cols = np.nonzero(session.object_labels == target)
    points = json.dumps([[int(cols.min()), int(rows.min())], [int(cols.max()), int(rows.min())],
                         [int(cols.max()), int(rows.max())]])
    session.selected_object = target
    out = review_tab.on_correct("Replace boundary", "", points, True, False, batch)
    assert "applied" in out[3]


def test_next_unreviewed_skips_reviewed_images(batch):
    batch.images[1].reviewed = True
    batch.active_index = 0
    review_tab.on_next_unreviewed(True, False, batch)
    assert batch.active_index == 0, "image 1 is reviewed and image 2 is not segmented"


def test_marking_reviewed_points_to_the_next_image(batch):
    batch.active_index = 0
    out = review_tab.on_mark_reviewed("fine", True, False, batch)
    assert "Next unreviewed: f1.png" in out[3]
    assert batch.images[0].review_note == "fine"


# --------------------------------------------------------------------------- #
# Segment tab: preview and staleness
# --------------------------------------------------------------------------- #


def _widgets_with(**changes):
    values = list(_widget_defaults())
    names = ["model", "diameter", "cellprob", "flow", "min_area", "use_gpu",
             "background_subtract", "background_radius", "gaussian_sigma", "normalize",
             "split_enabled", "saddle_depth", "solidity_max", "min_fragment_frac",
             "contact_distance", "nuclear_channel", "analysis_scale", "exclude_border",
             "exclude_ruler"]
    for key, value in changes.items():
        values[names.index(key)] = value
    return values


def test_preview_changes_nothing_until_kept(batch):
    session = batch.images[0]
    before = session.object_labels
    version = session.segmentation_version
    out = segment_tab.on_preview(*_widgets_with(min_area=400), "threshold_watershed", batch)
    assert session.object_labels is before and session.segmentation_version == version
    comparison = out[8]
    assert list(comparison.columns) == ["Measure", "Current", "Preview", "Change"]
    assert session.view_cache.get("candidate") is not None

    segment_tab.on_keep_preview(batch)
    assert session.object_labels is not before
    assert session.segmentation_version > version
    assert session.segmentation_params.min_object_area == 400
    assert not session.reviewed
    assert compute_results(session).counts["objects_detected"] == len(session.objects)


def test_discarding_a_preview_leaves_the_image_alone(batch):
    session = batch.images[0]
    before = session.object_labels
    segment_tab.on_preview(*_widgets_with(min_area=400), "threshold_watershed", batch)
    segment_tab.on_discard_preview(batch)
    assert session.object_labels is before
    assert "candidate" not in session.view_cache


def test_settings_note_counts_images_that_would_change(batch):
    note = segment_tab.on_settings_changed(*_widgets_with(flow=1.2), batch)
    assert "2 of 2 segmented image(s)" in note
    assert "match" in segment_tab.on_settings_changed(*_widgets_with(), batch)


def test_using_settings_for_the_batch_marks_images_stale_without_running(batch):
    versions = [s.segmentation_version for s in batch.images]
    out = segment_tab.on_apply_to_batch(*_widgets_with(cellprob=1.0), batch)
    assert [s.segmentation_version for s in batch.images] == versions
    assert "2 segmented image(s) are now stale" in out[2]


def test_batch_run_updates_only_images_that_need_it(batch):
    segment_tab.on_apply_to_batch(*_widgets_with(cellprob=1.0), batch)
    batch.images[0].segmented_with = batch.images[0].segmented_with   # still stale
    from src.workflow import segmentation_fingerprint

    batch.images[1].segmented_with = segmentation_fingerprint(batch.images[1])   # pretend current
    before = batch.images[1].segmentation_version
    list(segment_tab.on_segment_all(*_widgets_with(cellprob=1.0), "threshold_watershed",
                                    segment_tab.RUN_SCOPES[0], batch))
    assert batch.images[1].segmentation_version == before
    assert batch.images[2].has_segmentation
    assert not any(views.image_table(batch)["Status"] == "stale")


def test_apply_rules_changes_exclusions_not_masks(batch):
    session = batch.images[0]
    labels = session.object_labels
    segment_tab.on_apply_rules(False, False, batch)
    assert session.object_labels is labels
    assert session.qc_params.exclude_border is False
    assert not any(o.object_id in session.qc.excluded_ids for o in session.objects if o.touches_border)


# --------------------------------------------------------------------------- #
# Other tabs
# --------------------------------------------------------------------------- #


def test_export_marks_the_workflow_step_done(batch):
    batch.calibration_confirmed = True
    path, note, status = export_tab.on_export(batch)
    assert path and "Wrote" in note
    assert batch.exported_token is not None
    assert 'cs-step done" aria-label="Export' in status


def test_design_is_saved_and_summarised(batch):
    table = quality_tab.design_table(batch)
    table.loc[0, ["Well", "Condition", "Biological replicate"]] = ["A1", "ctrl", "R1"]
    table.loc[1, ["Well", "Condition", "Biological replicate"]] = ["B1", "ctrl", "R2"]
    out = quality_tab.on_save_design(table, "area", batch)
    assert out[-1] == "Experimental design saved."
    conditions = out[2]
    assert conditions.loc[0, "biological_replicates"] == 2


def test_opening_a_project_keeps_the_current_one_recoverable(batch, tmp_path, monkeypatch):
    from src.autosave import Autosaver

    saver = Autosaver(tmp_path)
    monkeypatch.setattr(projects_tab, "AUTOSAVER", saver)
    path, _, _ = projects_tab.on_save(batch)
    other = BatchSession(label="other")
    other.add(AnalysisSession())
    other.images[0].image = batch.images[0].image
    loaded, note = projects_tab.on_open(path, other)
    assert loaded.label == "ui" and "Opened" in note
    assert list(tmp_path.glob("*.cellscope")), "the replaced batch was autosaved first"
    saver.shutdown()


def test_status_contains_the_stepper(batch):
    html = views.status_html(batch)
    assert 'class="cs-steps"' in html and "cs-next" in html
    assert html.count('<li class="cs-step ') == 6
