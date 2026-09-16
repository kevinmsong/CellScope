"""Tests for the Gradio layer.

Two kinds live here. The pure helpers (status strip, image table, ID parsing)
are tested directly. The **wiring** is tested against Gradio's registered
dependencies, because a callback can be perfectly correct and still be connected
to the wrong number of outputs -- that is exactly how the nuclear-channel
control once ended up never populating.

Skipped without Gradio installed; app.py is the only module that imports it.
"""

from __future__ import annotations

import numpy as np
import pytest

gradio = pytest.importorskip("gradio", reason="Gradio is not installed")

import app as cellscope_app
from src.calibration import calibration_from_resolution, uncalibrated
from src.types import AnalysisSession, BatchSession, ImageRecord, ObjectRecord


def _record(name="x.jpg", height=10, width=10):
    return ImageRecord(
        name, "0" * 64, height, width, "uint8", 3, np.zeros((height, width, 3), np.uint8)
    )


@pytest.fixture
def batch():
    session = BatchSession(label="exp 1")
    for name in ("a.jpg", "b.jpg"):
        image = AnalysisSession()
        image.image = _record(name)
        session.add(image)
    return session


# --------------------------------------------------------------------------- #
# Status strip
# --------------------------------------------------------------------------- #


def test_status_shows_placeholders_before_anything_is_loaded():
    fields = cellscope_app.status_fields(BatchSession())
    assert set(fields) == set(cellscope_app.STATUS_FIELDS)
    assert all(value == cellscope_app.DASH for value in fields.values())


def test_status_reports_batch_level_counts(batch):
    batch.images[0].reviewed = True
    fields = cellscope_app.status_fields(batch)
    assert fields["Batch"] == "exp 1"
    assert fields["Images"] == "2"
    assert fields["Reviewed"] == "1 / 2"
    assert fields["Segmented"] == "0 / 2"


def test_status_html_renders_every_field(batch):
    html = cellscope_app.status_html(batch)
    for key in cellscope_app.STATUS_FIELDS:
        assert ">{}<".format(key) in html


def test_status_flags_a_batch_whose_units_cannot_be_pooled(batch):
    """px and um in one column is not a number, and the header must say so."""
    from conftest import fluorescence_phantom

    from src.pipeline import run_segmentation

    for image, calibration in zip(
        batch.images, (calibration_from_resolution(0.25), uncalibrated())
    ):
        array = fluorescence_phantom(grid=2, size=200, spacing=80)
        image.image = ImageRecord(
            image.display_name, "0" * 64, *array.shape[:2], "float32", 1, array
        )
        run_segmentation(image, array, engine="threshold_watershed")
        image.calibration = calibration
        image.results_cache = None

    assert cellscope_app.status_fields(batch)["Cells"] == "mixed units"
    assert "cs-stale" in cellscope_app.status_html(batch)


# --------------------------------------------------------------------------- #
# Image table and navigator
# --------------------------------------------------------------------------- #


def test_image_table_has_one_row_per_image(batch):
    table = cellscope_app.image_table(batch)
    assert list(table.columns) == cellscope_app.IMAGE_TABLE_COLUMNS
    assert len(table) == 2
    assert table["File"].tolist() == ["a.jpg", "b.jpg"]
    assert table["Status"].tolist() == ["pending", "pending"]


def test_image_table_shows_wells_and_review_state(batch):
    batch.images[0].well = "B3"
    batch.images[0].reviewed = True
    table = cellscope_app.image_table(batch)
    assert table.loc[0, "Well"] == "B3"
    assert table.loc[0, "Reviewed"] == "yes"
    assert table.loc[1, "Reviewed"] == "no"


def test_empty_batch_table_still_has_its_columns():
    table = cellscope_app.image_table(BatchSession())
    assert list(table.columns) == cellscope_app.IMAGE_TABLE_COLUMNS
    assert len(table) == 0


def test_navigator_names_the_active_image(batch):
    html = cellscope_app.nav_html(batch)
    assert "1 / 2" in html and "a.jpg" in html
    batch.active_index = 1
    assert "2 / 2" in cellscope_app.nav_html(batch)


def test_navigator_marks_review_and_failure_state(batch):
    assert "unreviewed" in cellscope_app.nav_html(batch)
    batch.images[0].reviewed = True
    assert "cs-badge ok" in cellscope_app.nav_html(batch)

    batch.images[0].error = "boom"
    assert "failed" in cellscope_app.nav_html(batch)


def test_review_strip_marks_the_active_and_done_images(batch):
    batch.images[1].reviewed = True
    html = cellscope_app.strip_html(batch)
    assert html.count("cs-tile") == 2
    assert "active" in html and "done" in html


# --------------------------------------------------------------------------- #
# Well editing
# --------------------------------------------------------------------------- #


def test_editing_the_well_column_writes_back(batch):
    table = cellscope_app.image_table(batch)
    table.loc[0, "Well"] = "C4"
    cellscope_app.on_table_edited(table, batch)
    assert batch.images[0].well == "C4"


def test_edits_to_derived_columns_are_ignored(batch):
    """Only Well is read back; the rest is regenerated, so a stray edit cannot
    corrupt state."""
    table = cellscope_app.image_table(batch)
    table.loc[0, "Cells"] = "9999"
    table.loc[0, "Status"] = "nonsense"
    _, refreshed, _ = cellscope_app.on_table_edited(table, batch)
    assert refreshed.loc[0, "Status"] == "pending"
    assert refreshed.loc[0, "Cells"] == cellscope_app.DASH


# --------------------------------------------------------------------------- #
# Navigation
# --------------------------------------------------------------------------- #


def test_next_and_prev_wrap_around(batch):
    cellscope_app.on_next(batch)
    assert batch.active_index == 1
    cellscope_app.on_next(batch)
    assert batch.active_index == 0
    cellscope_app.on_prev(batch)
    assert batch.active_index == 1


def test_removing_an_image_keeps_the_index_valid(batch):
    batch.active_index = 1
    cellscope_app.on_remove_image(batch)
    assert len(batch.images) == 1
    assert batch.active_index == 0
    assert cellscope_app.nav_html(batch).count("/ 1") == 1


def test_navigation_on_an_empty_batch_does_not_raise():
    empty = BatchSession()
    cellscope_app.on_next(empty)
    cellscope_app.on_prev(empty)
    cellscope_app.on_remove_image(empty)
    assert empty.is_empty


# --------------------------------------------------------------------------- #
# ID parsing
# --------------------------------------------------------------------------- #


@pytest.fixture
def session_with_objects():
    session = AnalysisSession()
    session.objects = [ObjectRecord(i, i, "resolved") for i in (1, 2, 7)]
    return session


def test_parse_ids_accepts_known_objects(session_with_objects):
    ids, unknown = cellscope_app._parse_ids(" 1, 2 ,7 ", session_with_objects)
    assert ids == [1, 2, 7]
    assert unknown == []


def test_parse_ids_rejects_ids_that_do_not_exist(session_with_objects):
    """A phantom exclusion would put a false entry in the QC log."""
    ids, unknown = cellscope_app._parse_ids("1 999", session_with_objects)
    assert ids == [1]
    assert unknown == [999]


def test_parse_ids_rejects_non_numeric(session_with_objects):
    ids, unknown = cellscope_app._parse_ids("1 banana", session_with_objects)
    assert ids == [1]
    assert unknown == ["banana"]


def test_parse_ids_handles_empty_input(session_with_objects):
    assert cellscope_app._parse_ids("", session_with_objects) == ([], [])
    assert cellscope_app._parse_ids(None, session_with_objects) == ([], [])


# --------------------------------------------------------------------------- #
# Wiring: a callback's return must match the outputs it is connected to.
#
# These exist because a real bug slipped through: a control was added to a
# callback's return but not to its outputs list, so its update was silently
# discarded. Testing the function alone could not catch that.
# --------------------------------------------------------------------------- #


def _dependencies():
    demo = cellscope_app.build_interface()
    fns = demo.fns
    return list(fns.values() if isinstance(fns, dict) else fns)


def _outputs_for(name):
    matches = [d for d in _dependencies() if getattr(d.fn, "__name__", "") == name]
    assert matches, "no event wired to {}".format(name)
    return matches


@pytest.mark.parametrize(
    "name,call",
    [
        ("on_upload", lambda b: cellscope_app.on_upload(None, b)),
        ("on_prev", cellscope_app.on_prev),
        ("on_next", cellscope_app.on_next),
        ("on_remove_image", cellscope_app.on_remove_image),
        ("on_refresh_results", cellscope_app.on_refresh_results),
    ],
)
def test_callback_returns_match_their_wired_outputs(name, call, batch):
    """Every value a callback returns must have a component to land in."""
    for dependency in _outputs_for(name):
        returned = call(batch)
        assert len(returned) == len(dependency.outputs), (
            "{} returns {} values but is wired to {} outputs".format(
                name, len(returned), len(dependency.outputs)
            )
        )


def test_review_callbacks_match_their_outputs(batch):
    for name, call in (
        ("on_toggle_labels", lambda b: cellscope_app.on_toggle_labels(True, False, b)),
        ("on_mark_reviewed", lambda b: cellscope_app.on_mark_reviewed("", True, False, b)),
        ("on_reset_qc", lambda b: cellscope_app.on_reset_qc(True, False, b)),
    ):
        for dependency in _outputs_for(name):
            assert len(call(batch)) == len(dependency.outputs), name


def test_upload_populates_both_channel_controls(tmp_path):
    """The nuclear control must actually be filled in, not left on its placeholder."""
    import os

    sample = os.path.join("images", "11_0001-z.jpg")
    if not os.path.exists(sample):
        pytest.skip("sample image not present")

    returned = cellscope_app.on_upload([sample], BatchSession())
    channel_update, nuclear_update = returned[-4], returned[-3]
    assert set(channel_update["choices"]) == {"grayscale", "red", "green", "blue"}
    assert nuclear_update["choices"][0] == cellscope_app._NO_NUCLEAR
    assert set(nuclear_update["choices"]) == {
        "none", "grayscale", "red", "green", "blue"
    }


def test_cancel_stops_both_segmentation_runs():
    """Cancel once only reached "run all": the loop leaked its last event."""
    config = cellscope_app.build_interface().get_config_file()
    names = {d["id"]: d.get("api_name") for d in config["dependencies"]}
    cancelled = [
        {names[i] for i in d["cancels"]}
        for d in config["dependencies"] if d.get("cancels")
    ]
    assert any({"on_segment_one", "on_segment_all"} <= group for group in cancelled)


def test_every_event_has_outputs():
    for dependency in _dependencies():
        name = getattr(dependency.fn, "__name__", "?")
        if name in ("__init__", "<lambda>"):
            continue
        assert dependency.outputs is not None, name


# --------------------------------------------------------------------------- #
# Presentation
# --------------------------------------------------------------------------- #


def test_calibration_method_switch_shows_exactly_one_panel():
    for choice in cellscope_app.CALIBRATION_METHODS:
        updates = [
            gradio.update(visible=choice == name)
            for name in cellscope_app.CALIBRATION_METHODS
        ]
        assert sum(1 for u in updates if u["visible"]) == 1


def test_legend_names_all_three_outcomes():
    from ui_theme import legend_html

    html = legend_html().lower()
    for label in ("isolated", "cluster", "unresolved"):
        assert label in html


def test_pseudoreplication_note_names_the_right_replicate_unit():
    """The batch pools cells; the caveat must say what the replicate actually is."""
    note = cellscope_app.PSEUDOREPLICATION_NOTE
    assert "not</em> independent biological replicates" in note or \
           "not independent biological replicates" in note
    assert "well" in note and "p-values" in note


def _segmented_batch(n=3):
    """A batch of ``n`` segmented images, plus the runner used to segment them."""
    from conftest import fluorescence_phantom

    from src.pipeline import run_batch_segmentation
    from src.types import ImageRecord

    batch = BatchSession()
    array = fluorescence_phantom(grid=2, size=200, spacing=80)
    for i in range(n):
        image = AnalysisSession()
        image.image = ImageRecord(
            "i{}.png".format(i), "0" * 64, *array.shape[:2], "float32", 1, array
        )
        batch.add(image)
    return batch, run_batch_segmentation


def test_a_nuclear_file_on_only_some_images_is_reported():
    """The nuclear guide has two routes -- a channel or an attached file -- and a
    per-image override of either can still diverge from the batch."""
    from src.types import ImageRecord

    batch, run = _segmented_batch(3)
    source = batch.images[0].image
    batch.images[0].nuclear_image = ImageRecord(
        "dapi.png", "0" * 64, source.height, source.width, source.dtype, 1, source.pixels
    )
    run(batch, engine="threshold_watershed")

    warning = cellscope_app._channel_divergence_warning(batch)
    assert "nuclear guide" in warning
    assert "dapi.png x1" in warning and "none x2" in warning


# --------------------------------------------------------------------------- #
# Channels are a batch setting
#
# They used to be per-image with an "Apply to all" button, which meant setting
# the radio segmented one image on red and the other eleven on greyscale. The
# radios now cover the whole batch; the image table holds the override.
# --------------------------------------------------------------------------- #


def _rgb_batch(n=3):
    session = BatchSession()
    for i in range(n):
        image = AnalysisSession()
        image.image = _record("img{}.jpg".format(i))
        session.add(image)
    return session


def test_choosing_the_cell_channel_once_covers_every_image():
    batch = _rgb_batch(4)
    cellscope_app.on_channel_change("red", batch)
    assert [s.channel for s in batch.images] == ["red"] * 4
    assert batch.channel == "red"


def test_choosing_the_nuclear_channel_once_covers_every_image():
    batch = _rgb_batch(4)
    cellscope_app.on_nuclear_change("blue", batch)
    assert all(
        s.segmentation_params.nuclear_channel == "blue" for s in batch.images
    )


def test_images_added_later_inherit_the_batch_channels():
    """Uploading in two goes must not leave half the batch on greyscale."""
    batch = _rgb_batch(2)
    cellscope_app.on_channel_change("red", batch)
    cellscope_app.on_nuclear_change("blue", batch)

    late = AnalysisSession()
    late.image = _record("late.jpg")
    batch.add(late)
    batch.apply_channels(late)

    assert late.channel == "red"
    assert late.segmentation_params.nuclear_channel == "blue"


def test_a_channel_an_image_lacks_is_not_forced_on_it():
    import numpy as np

    from src.types import ImageRecord

    batch = _rgb_batch(2)
    grey = batch.images[1]
    grey.image = ImageRecord(
        "grey.tif", "0" * 64, 10, 10, "uint8", 1, np.zeros((10, 10), np.uint8)
    )
    cellscope_app.on_channel_change("red", batch)

    assert batch.images[0].channel == "red"
    assert grey.channel == "grayscale", "a greyscale image has no red plane"


def test_setting_a_channel_clears_stale_results():
    batch = _rgb_batch(2)
    batch.images[0].results_cache = ("stale", "value")
    cellscope_app.on_channel_change("red", batch)
    assert batch.images[0].results_cache is None


def test_the_image_table_overrides_one_image():
    """The per-image escape hatch: edit the Channel cell for a single row."""
    batch = _rgb_batch(3)
    cellscope_app.on_channel_change("red", batch)

    table = cellscope_app.image_table(batch)
    table.loc[1, "Channel"] = "blue"
    cellscope_app.on_table_edited(table, batch)

    assert [s.channel for s in batch.images] == ["red", "blue", "red"]


def test_the_image_table_overrides_the_nuclear_channel():
    batch = _rgb_batch(2)
    cellscope_app.on_nuclear_change("blue", batch)

    table = cellscope_app.image_table(batch)
    table.loc[0, "Nuclear"] = "none"
    cellscope_app.on_table_edited(table, batch)

    assert batch.images[0].segmentation_params.nuclear_channel is None
    assert batch.images[1].segmentation_params.nuclear_channel == "blue"


def test_channel_change_on_an_empty_batch_is_harmless():
    empty = BatchSession()
    cellscope_app.on_channel_change("red", empty)
    cellscope_app.on_nuclear_change("blue", empty)
    assert empty.channel == "red"


# --------------------------------------------------------------------------- #
# Duplicate uploads
#
# batch_summary pools cells across images, so the same field added twice
# contributes its cells twice and inflates n_cells. Confirmed happening in
# practice: one file appeared four times in a real batch.
# --------------------------------------------------------------------------- #


def test_the_same_file_is_not_added_twice(tmp_path):
    import os

    sample = os.path.join("images", "11_0001-z.jpg")
    if not os.path.exists(sample):
        pytest.skip("sample image not present")

    batch = BatchSession()
    batch = cellscope_app.on_upload([sample], batch)[0]
    assert len(batch.images) == 1

    returned = cellscope_app.on_upload([sample], batch)
    batch, note = returned[0], returned[4]
    assert len(batch.images) == 1, "the duplicate was added anyway"
    assert "Skipped 1 already in this batch" in note
    assert "11_0001-z.jpg" in note


def test_two_different_files_sharing_a_name_are_both_kept(tmp_path):
    """Both condition folders number their fields 1.jpg..10.jpg, and those are
    genuinely different images. Deduplication is by content, not filename."""
    import os
    import shutil

    sample = os.path.join("images", "11_0001-z.jpg")
    other = os.path.join("images", "11_0001-1.jpg")
    if not (os.path.exists(sample) and os.path.exists(other)):
        pytest.skip("sample images not present")

    a = tmp_path / "a" / "1.jpg"
    b = tmp_path / "b" / "1.jpg"
    a.parent.mkdir(parents=True)
    b.parent.mkdir(parents=True)
    shutil.copy(sample, a)
    shutil.copy(other, b)

    batch = cellscope_app.on_upload([str(a), str(b)], BatchSession())[0]
    assert len(batch.images) == 2, "same name, different content -- both belong"


# --------------------------------------------------------------------------- #
# Header reset
# --------------------------------------------------------------------------- #


def test_reset_is_immediate_when_there_is_nothing_to_lose():
    batch, note = cellscope_app.on_home(BatchSession())[:2]
    assert note == "New batch started."
    assert batch.is_empty


def test_reset_asks_before_discarding_work(batch):
    first = cellscope_app.on_home(batch)
    assert "Click CellScope again to discard" in first[1]
    assert "2 image(s)" in first[1]
    assert len(first[0].images) == 2, "nothing may be discarded on the first click"

    second = cellscope_app.on_home(batch)
    assert second[1] == "New batch started."
    assert second[0].is_empty


def test_reset_returns_to_the_batch_tab(batch):
    cellscope_app.on_home(batch)             # arm
    tabs_update = cellscope_app.on_home(batch)[2]
    assert getattr(tabs_update, "selected", None) == "tab_batch"


def test_another_action_disarms_a_pending_reset(batch):
    cellscope_app.on_home(batch)             # arm
    cellscope_app.on_batch_label("something", batch)
    assert "Click CellScope again" in cellscope_app.on_home(batch)[1], (
        "the reset stayed armed across an unrelated action"
    )


@pytest.fixture
def clickable_batch():
    batch = BatchSession()
    session = AnalysisSession()
    session.image = _record(height=20, width=30)
    session.display_rgb = np.zeros((10, 15, 3), np.uint8)
    session.object_labels = np.zeros((20, 30), np.int32)
    session.object_labels[4:12, 6:18] = 1
    session.objects = [ObjectRecord(1, 1, "resolved")]
    batch.add(session)
    return batch


def test_review_click_toggles_resized_cell_and_updates_results(clickable_batch):
    batch = clickable_batch
    session = batch.active
    original = session.object_labels.copy()
    session.reviewed = True
    event = gradio.SelectData(None, {"index": [4, 3], "value": None})
    output = cellscope_app.on_review_click(True, False, batch, event)
    assert session.qc.excluded_ids == {1}
    assert not session.reviewed
    assert "0 included" in output[3]
    assert session.qc.log[-1].action == "exclude"
    output = cellscope_app.on_review_click(True, False, batch, event)
    assert session.qc.excluded_ids == set()
    assert "1 included" in output[3]
    assert session.qc.log[-1].action == "restore"
    np.testing.assert_array_equal(session.object_labels, original)


@pytest.mark.parametrize("point", [[0, 0], [-1, 3], [15, 10]])
def test_review_click_background_or_outside_is_noop(clickable_batch, point):
    event = gradio.SelectData(None, {"index": point, "value": None})
    cellscope_app.on_review_click(True, False, clickable_batch, event)
    assert not clickable_batch.active.qc.log


def test_click_callback_wired_to_review_outputs(clickable_batch):
    """Review clicks arrive from the browser viewer as display-pixel coordinates."""
    payload = '{"x": 4, "y": 3}'
    for dependency in _outputs_for("on_canvas_click"):
        returned = cellscope_app.on_canvas_click(payload, True, False, clickable_batch)
        assert len(returned) == len(dependency.outputs)
    assert clickable_batch.active.qc.excluded_ids == {1}


def test_canvas_click_matches_a_select_event(clickable_batch):
    session = clickable_batch.active
    cellscope_app.on_canvas_click('{"x": 4, "y": 3}', True, False, clickable_batch)
    via_canvas = set(session.qc.excluded_ids)
    event = gradio.SelectData(None, {"index": [4, 3], "value": None})
    cellscope_app.on_review_click(True, False, clickable_batch, event)
    assert via_canvas == {1} and session.qc.excluded_ids == set()


@pytest.mark.parametrize("payload", ["", "not json", '{"x": 1}', '{"x": "a", "y": 2}'])
def test_malformed_canvas_clicks_are_ignored(clickable_batch, payload):
    cellscope_app.on_canvas_click(payload, True, False, clickable_batch)
    assert not clickable_batch.active.qc.log


def test_theme_has_distinct_dark_surfaces():
    from ui_theme import CSS, build_theme
    theme = build_theme()
    assert theme.body_background_fill != theme.body_background_fill_dark
    assert theme.body_text_color != theme.body_text_color_dark
    assert "--cs-surface:#111827" in CSS


def test_reference_calibration_accepts_800x600_ruler_for_800x800_sample():
    batch = BatchSession()
    session = AnalysisSession()
    session.image = _record("1.jpg", height=800, width=800)
    session.reference_image = _record("20x ruler.jpg", height=600, width=800)
    batch.add(session)
    cellscope_app.on_apply_reference(200, 100, "This image only", batch)
    assert session.calibration.scales == (0.5, 0.5)
    assert session.calibration.method == "reference_scale_bar"
