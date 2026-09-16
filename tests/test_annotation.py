"""Cells touching a scale bar burned into the data image are excluded automatically.

The synthetic ruler reproduces the geometry of the lab's exports: an 8 px
yellow bar, 126 px long, about 80 px above the bottom-right corner of an
800x800 field, with a caption of blocky glyphs beneath it.
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest
from skimage import draw

from src.perf import RULER_RGB, draw_ruler, image_record
from src.pipeline import compute_results, reapply_automatic_qc, run_segmentation
from src.project import apply_preset, load_project, preset_dict, save_project
from src.qc import REASON_ANNOTATION, REASON_BORDER, exclusion_counts
from src.review import checkpoint, correct_mask, history, signature, validate_review
from src.scalebar import annotation_touching_ids, detect_annotation, detect_scale_bar
from src.types import AnalysisSession, BatchSession, QCParams

BAR = (713, 646, 721, 772)      # rows 713-720, columns 646-771


def _field(cells, size=800, ruler=True):
    """Red disks at ``(row, col, radius)`` on black, plus the ruler."""
    rgb = np.zeros((size, size, 3), np.uint8)
    for row, col, radius in cells:
        rr, cc = draw.disk((row, col), radius, shape=rgb.shape[:2])
        rgb[rr, cc, 0] = 180
    if ruler:
        assert draw_ruler(rgb) == BAR
    return rgb


def _segmented(cells, ruler=True):
    rgb = _field(cells, ruler=ruler)
    session = AnalysisSession()
    session.image = image_record(rgb, "ruler.png")
    session.channel = "red"
    run_segmentation(session, rgb[..., 0].astype(np.float32), engine="threshold_watershed")
    return session


def _object_at(session, row, col):
    return int(session.object_labels[row, col])


# --------------------------------------------------------------------------- #
# Detection
# --------------------------------------------------------------------------- #


def test_detects_the_bar_and_its_caption():
    found = detect_annotation(_field([]))
    assert found.found and found.colour == "yellow"
    assert found.bar_bbox == BAR
    assert found.bar_length_px == 126
    assert found.caption_bbox is not None
    assert found.caption_bbox[0] > BAR[2], "the caption sits under the bar"
    r0, c0, r1, c1 = found.mask_bbox
    assert r0 <= BAR[0] - 2 and r1 >= found.caption_bbox[2] + 2


def test_region_covers_every_annotation_pixel_with_a_margin():
    rgb = _field([])
    found = detect_annotation(rgb)
    painted = np.all(rgb == RULER_RGB, axis=2)
    full = np.zeros(painted.shape, bool)
    r0, c0, r1, c1 = found.mask_bbox
    full[r0:r1, c0:c1] = found.mask
    assert full[painted].all()
    assert r0 <= BAR[0] - found.margin_px and c1 >= BAR[3] + found.margin_px

    # Contact extends two pixels beyond the annotation, not three.
    labels = np.zeros(painted.shape, np.int32)
    labels[716, BAR[3] + 1] = 1          # distance 2 from the bar's last column
    labels[716, BAR[3] + 2] = 2          # distance 3
    assert annotation_touching_ids(labels, found) == {1}


def test_greyscale_images_never_report_an_annotation():
    grey = _field([]).max(axis=2)
    assert not detect_annotation(grey).found
    stacked = np.repeat(grey[..., None], 3, axis=2)
    stacked[stacked > 0] = 255                 # a white bar in a grey RGB file
    assert not detect_annotation(stacked).found


def test_a_bright_elongated_cell_is_not_an_annotation():
    rgb = np.zeros((800, 800, 3), np.uint8)
    rr, cc = draw.rectangle((740, 600), extent=(8, 150))
    rgb[rr, cc] = (255, 120, 40)               # orange-red, not ruler yellow
    rgb[400:408, 300:450] = (255, 255, 0)      # yellow, but mid-frame
    assert not detect_annotation(rgb).found


def test_reference_frame_detection_is_unchanged():
    """The looser calibration detector still accepts a plain bright bar."""
    frame = np.zeros((600, 800, 3), np.uint8)
    frame[500:506, 600:711] = (240, 90, 240)      # bright magenta: not ruler colours
    assert detect_scale_bar(frame).found
    assert not detect_annotation(frame).found


# --------------------------------------------------------------------------- #
# Exclusion
# --------------------------------------------------------------------------- #


def test_contact_is_decided_by_distance_to_the_ruler():
    """Within the 2 px margin counts as touching; 3 px away does not.

    Radius-12 disks span their centre +/- 11 px, so centres at rows 710, 700
    and 699 put a cell's lowest row at 721 (on the bar), 711 (distance 2 from
    the bar's top row, 713) and 710 (distance 3).
    """
    session = _segmented([(710, 665, 12), (700, 710, 12), (699, 755, 12), (200, 200, 20)])
    on_bar = _object_at(session, 705, 665)
    near = _object_at(session, 700, 710)
    clear = _object_at(session, 699, 755)
    far = _object_at(session, 200, 200)
    assert np.nonzero(session.object_labels == near)[0].max() == 711
    assert np.nonzero(session.object_labels == clear)[0].max() == 710

    by_id = {o.object_id: o for o in session.objects}
    assert by_id[on_bar].touches_annotation and on_bar in session.qc.excluded_ids
    assert by_id[near].touches_annotation and near in session.qc.excluded_ids
    assert not by_id[clear].touches_annotation and clear not in session.qc.excluded_ids
    assert not by_id[far].touches_annotation and far not in session.qc.excluded_ids
    assert annotation_touching_ids(session.object_labels, session.annotation) >= {on_bar, near}


def test_a_cell_touching_only_the_caption_is_excluded():
    # The first glyph starts at column 640, rows 740-773.
    session = _segmented([(756, 620, 12), (200, 200, 20)])
    beside = _object_at(session, 756, 620)
    assert np.nonzero(session.object_labels == beside)[1].max() == 631
    assert beside not in session.qc.excluded_ids, "9 px from the glyph"

    # 1 px from the glyph. Threshold segmentation merges a strip of this cell
    # into the glyph's object; what is left of the cell must still count.
    session = _segmented([(756, 628, 12), (200, 200, 20)])
    touching = _object_at(session, 756, 624)
    glyph = _object_at(session, 756, 642)
    assert touching != glyph
    assert {touching, glyph} <= session.qc.excluded_ids


def test_neighbours_of_a_real_cell_on_the_ruler_are_not_excluded():
    """Only objects that are mostly annotation pass contact on."""
    # A large cell straddling the bar, and a cell touching it from above.
    session = _segmented([(705, 700, 30), (660, 700, 16), (200, 200, 20)])
    straddling = _object_at(session, 700, 700)
    above = _object_at(session, 655, 700)
    assert straddling != above
    assert straddling in session.qc.excluded_ids
    assert above not in session.qc.excluded_ids


def test_ruler_glyphs_segmented_as_objects_are_excluded():
    session = _segmented([(200, 200, 20)])
    glyph_objects = [o for o in session.objects if o.touches_annotation]
    assert glyph_objects, "the red channel sees the yellow ruler"
    for obj in glyph_objects:
        assert obj.object_id in session.qc.excluded_ids
    reasons = {e.object_id: e.reason for e in session.qc.log}
    assert all(reasons[o.object_id] == REASON_ANNOTATION for o in glyph_objects)
    results = compute_results(session)
    assert results.counts["objects_accepted"] == 1


def test_exclusion_counts_break_down_by_reason():
    session = _segmented([(700, 700, 20), (10, 400, 20), (200, 200, 20)])
    counts = exclusion_counts(session.objects, session.qc)
    assert counts["excluded_border"] == 1
    assert counts["excluded_annotation"] >= 1
    summary = compute_results(session).image_summary.iloc[0]
    assert summary["excluded_annotation"] == counts["excluded_annotation"]


def test_an_image_without_a_ruler_is_unaffected():
    session = _segmented([(700, 700, 20), (200, 200, 20)], ruler=False)
    assert not session.annotation.found
    assert not session.qc.excluded_ids
    assert not any(o.touches_annotation for o in session.objects)


def test_a_manual_restore_survives_reapplying_the_rules():
    session = _segmented([(700, 700, 20), (200, 200, 20)])
    target = _object_at(session, 700, 700)
    checkpoint(session)
    session.qc.restore(target, "cell is fine")
    reapply_automatic_qc(session)
    assert target not in session.qc.excluded_ids


def test_switching_the_rule_off_restores_only_automatic_exclusions():
    session = _segmented([(700, 700, 20), (200, 200, 20)])
    ruler_ids = {o.object_id for o in session.objects if o.touches_annotation}
    far = _object_at(session, 200, 200)
    session.qc.exclude(far, "out of focus")

    session.qc_params = replace(session.qc_params, exclude_annotations=False)
    counts = reapply_automatic_qc(session)
    assert counts["restored"] == len(ruler_ids)
    assert not (ruler_ids & session.qc.excluded_ids)
    assert far in session.qc.excluded_ids, "a manual exclusion is never undone"

    session.qc_params = QCParams()
    reapply_automatic_qc(session)
    assert ruler_ids <= session.qc.excluded_ids


def test_the_rule_can_be_disabled_before_segmentation():
    rgb = _field([(700, 700, 20)])
    session = AnalysisSession()
    session.image = image_record(rgb, "ruler.png")
    session.qc_params = QCParams(exclude_annotations=False)
    run_segmentation(session, rgb[..., 0].astype(np.float32), engine="threshold_watershed")
    assert any(o.touches_annotation for o in session.objects), "still flagged"
    assert not session.qc.excluded_ids


def test_undo_restores_an_automatic_exclusion_state():
    session = _segmented([(700, 700, 20), (200, 200, 20)])
    target = _object_at(session, 700, 700)
    checkpoint(session)
    session.qc.restore(target, "manual")
    assert history(session)
    assert target in session.qc.excluded_ids


def test_a_correction_that_moves_a_cell_onto_the_ruler_excludes_it():
    session = _segmented([(600, 700, 20), (200, 200, 20)])
    target = _object_at(session, 600, 700)
    assert target not in session.qc.excluded_ids
    polygon = [(690, 590), (710, 590), (710, 716), (690, 716)]
    correct_mask(session, "Replace boundary", [target], polygon)
    assert next(o for o in session.objects if o.object_id == target).touches_annotation
    assert target in session.qc.excluded_ids


def test_a_split_child_touching_the_border_is_excluded():
    """Consistent with segmentation: new objects face the automatic rules too."""
    labels = np.zeros((100, 200), np.int32)
    labels[40:60, 0:80] = 1                    # reaches the left edge
    labels[40:60, 120:160] = 2
    session = AnalysisSession()
    session.image = image_record(np.zeros((100, 200, 3), np.uint8), "edge.png")
    session.object_labels = labels
    from src.types import ObjectRecord

    session.objects = [ObjectRecord(1, 1, "resolved"), ObjectRecord(2, 2, "resolved")]
    session.segmentation_version = 1
    correct_mask(session, "Split", [1], [(2, 50), (70, 50)])
    edge_objects = [o for o in session.objects if o.touches_border]
    assert len(edge_objects) == 1
    assert edge_objects[0].object_id in session.qc.excluded_ids
    reasons = {e.object_id: e.reason for e in session.qc.log}
    assert reasons[edge_objects[0].object_id] == REASON_BORDER
    assert 2 not in session.qc.excluded_ids


# --------------------------------------------------------------------------- #
# Persistence, presets and review compatibility
# --------------------------------------------------------------------------- #


def test_annotation_survives_a_project_round_trip(tmp_path):
    session = _segmented([(700, 700, 20)])
    batch = BatchSession()
    batch.add(session)
    session.qc_params = QCParams(exclude_border=False)
    loaded = load_project(save_project(batch, tmp_path / "p.cellscope"))
    restored = loaded.images[0]
    assert restored.annotation.found
    np.testing.assert_array_equal(restored.annotation.mask, session.annotation.mask)
    assert restored.annotation.bar_bbox == session.annotation.bar_bbox
    assert restored.qc_params == QCParams(exclude_border=False)
    assert {o.object_id for o in restored.objects if o.touches_annotation} == \
        {o.object_id for o in session.objects if o.touches_annotation}


def test_version_1_presets_still_apply():
    batch = BatchSession()
    data = preset_dict(batch)
    assert data["schema"] == 2 and "qc_params" in data["parameters"]
    data["schema"] = 1
    del data["parameters"]["qc_params"]
    apply_preset(batch, data)
    assert batch.qc_params == QCParams()


def test_presets_carry_the_qc_rules():
    source = BatchSession()
    source.qc_params = QCParams(exclude_annotations=False)
    target = BatchSession()
    apply_preset(target, preset_dict(source))
    assert target.qc_params == QCParams(exclude_annotations=False)


def test_default_qc_rules_do_not_invalidate_an_existing_review():
    session = _segmented([(200, 200, 20)])
    session.reviewed = True
    session.reviewed_signature = signature(session)
    validate_review(session)
    assert session.reviewed
    session.qc_params = QCParams(exclude_annotations=False)
    validate_review(session)
    assert not session.reviewed


@pytest.mark.parametrize("name", ["9_0001-1.jpg", "9_0001-4.jpg"])
def test_real_ruler_images_are_detected(name):
    """Local-only check against the lab's exports, skipped when absent."""
    import glob

    from src.image_io import load_image

    matches = glob.glob("images/**/" + name, recursive=True)
    if not matches:
        pytest.skip("local images not present")
    found = detect_annotation(load_image(matches[0]).pixels)
    assert found.found and found.bar_length_px == 126
    assert found.caption_bbox is not None
