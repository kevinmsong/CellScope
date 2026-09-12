"""Calibration tests, including the case §19 names explicitly."""

from __future__ import annotations

import numpy as np
import pytest

from src.calibration import (
    calculate_scale,
    calibration_from_resolution,
    calibration_from_scale_bar,
    display_to_original,
    pixel_distance,
    uncalibrated,
)
from src.morphometry import measure_cells
from src.types import ClusterRecord, ObjectRecord

# --------------------------------------------------------------------------- #
# The required case: 100 µm = 400 px  =>  0.25 µm/px  =>  100×100 px = 625 µm²
# --------------------------------------------------------------------------- #


def test_required_scale_bar_case():
    """§19: a 400 px bar labelled 100 µm gives exactly 0.25 µm/px."""
    scale = calculate_scale((100, 200), (500, 200), known_length_um=100.0)
    assert scale == pytest.approx(0.25, abs=1e-12)


def test_required_area_case():
    """§19: at 0.25 µm/px a 100×100 px square measures exactly 625 µm²."""
    calibration = calibration_from_resolution(0.25, 0.25)

    labels = np.zeros((200, 200), dtype=np.int32)
    labels[50:150, 50:150] = 1  # exactly 100 × 100 px

    objects = [ObjectRecord(object_id=1, raw_label=1, status="resolved")]
    clusters = [ClusterRecord(1, (1,), (1,), ())]
    frame = measure_cells(labels, objects, clusters, calibration)

    assert frame.loc[0, "area_px2"] == pytest.approx(10_000.0)
    assert frame.loc[0, "area_um2"] == pytest.approx(625.0, abs=1e-9)


def test_required_case_end_to_end_from_clicks():
    """The same 625 µm² result, arriving through the scale-bar click path."""
    calibration = calibration_from_scale_bar((0, 0), (400, 0), 100.0)
    assert calibration.um_per_px_x == pytest.approx(0.25)
    assert calibration.um_per_px_y == pytest.approx(0.25)
    assert calibration.is_isotropic

    labels = np.zeros((200, 200), dtype=np.int32)
    labels[50:150, 50:150] = 1
    frame = measure_cells(
        labels,
        [ObjectRecord(1, 1, "resolved")],
        [ClusterRecord(1, (1,), (1,), ())],
        calibration,
    )
    assert frame.loc[0, "area_um2"] == pytest.approx(625.0, abs=1e-9)


# --------------------------------------------------------------------------- #
# Scale-bar geometry
# --------------------------------------------------------------------------- #


def test_diagonal_scale_bar_uses_euclidean_length():
    """A 3-4-5 triangle: the bar is 5 px long, not 3 or 4."""
    assert pixel_distance((0, 0), (3, 4)) == pytest.approx(5.0)
    assert calculate_scale((0, 0), (3, 4), 10.0) == pytest.approx(2.0)


def test_scale_bar_rejects_zero_length():
    with pytest.raises(ValueError, match="same pixel"):
        calculate_scale((10, 10), (10, 10), 100.0)


@pytest.mark.parametrize("bad", [0.0, -5.0, None])
def test_scale_bar_rejects_non_positive_length(bad):
    with pytest.raises(ValueError, match="positive"):
        calculate_scale((0, 0), (400, 0), bad)


def test_scale_bar_records_its_inputs():
    """Provenance (§18): the endpoints and entered length survive into params."""
    calibration = calibration_from_scale_bar((10, 20), (410, 20), 100.0)
    params = calibration.params
    assert params["point1_px"] == [10.0, 20.0]
    assert params["point2_px"] == [410.0, 20.0]
    assert params["measured_length_px"] == pytest.approx(400.0)
    assert params["known_length_um"] == pytest.approx(100.0)


# --------------------------------------------------------------------------- #
# Display-coordinate mapping — a silent error here would scale every export
# --------------------------------------------------------------------------- #


def test_display_to_original_round_trip():
    """A 3264 px image shown at 1024 px: clicks must map back by the same factor."""
    scale = 3264 / 1024
    assert display_to_original((0, 0), scale) == (0.0, 0.0)
    x, y = display_to_original((512, 256), scale)
    assert x == pytest.approx(512 * scale)
    assert y == pytest.approx(256 * scale)


def test_display_scale_affects_calibration_as_expected():
    """A bar spanning 400 display px on a 2x-downscaled image is 800 original px."""
    scale = 2.0
    p1 = display_to_original((100, 50), scale)
    p2 = display_to_original((500, 50), scale)
    assert pixel_distance(p1, p2) == pytest.approx(800.0)
    assert calculate_scale(p1, p2, 100.0) == pytest.approx(0.125)


def test_display_to_original_rejects_bad_scale():
    with pytest.raises(ValueError, match="positive"):
        display_to_original((10, 10), 0.0)


# --------------------------------------------------------------------------- #
# Anisotropic and uncalibrated modes
# --------------------------------------------------------------------------- #


def test_manual_resolution_supports_anisotropic_pixels():
    calibration = calibration_from_resolution(0.5, 0.25)
    assert not calibration.is_isotropic
    assert calibration.scales == (0.5, 0.25)


def test_manual_resolution_defaults_to_square_pixels():
    calibration = calibration_from_resolution(0.4)
    assert calibration.is_isotropic
    assert calibration.scales == (0.4, 0.4)


def test_anisotropic_area_uses_both_scales():
    """area_um2 = pixel_count × sx × sy, not pixel_count × mean(sx, sy)²."""
    calibration = calibration_from_resolution(0.5, 0.25)
    labels = np.zeros((200, 200), dtype=np.int32)
    labels[50:150, 50:150] = 1  # 10 000 px

    frame = measure_cells(
        labels,
        [ObjectRecord(1, 1, "resolved")],
        [ClusterRecord(1, (1,), (1,), ())],
        calibration,
    )
    assert frame.loc[0, "area_um2"] == pytest.approx(10_000 * 0.5 * 0.25)
    # The mean-scale shortcut would give 10 000 × 0.375² = 1406.25.
    assert frame.loc[0, "area_um2"] != pytest.approx(1406.25)


@pytest.mark.parametrize("bad_x,bad_y", [(0, 1), (-1, 1), (1, 0), (1, -2), (None, 1)])
def test_manual_resolution_rejects_non_positive(bad_x, bad_y):
    with pytest.raises(ValueError, match="positive"):
        calibration_from_resolution(bad_x, bad_y)


def test_uncalibrated_emits_no_physical_columns():
    """§4C: uncalibrated results are px only — never a zero or NaN µm value that
    could be mistaken for a measurement."""
    calibration = uncalibrated()
    assert not calibration.is_calibrated

    labels = np.zeros((200, 200), dtype=np.int32)
    labels[50:150, 50:150] = 1
    frame = measure_cells(
        labels,
        [ObjectRecord(1, 1, "resolved")],
        [ClusterRecord(1, (1,), (1,), ())],
        calibration,
    )

    physical = [c for c in frame.columns if c.endswith(("_um", "_um2"))]
    assert physical == [], f"uncalibrated output leaked physical columns: {physical}"
    assert frame.loc[0, "area_px2"] == pytest.approx(10_000.0)


def test_uncalibrated_scales_are_unit_but_flag_is_false():
    """`scales` returns (1, 1) so pixel-space maths works, but nothing claims µm."""
    calibration = uncalibrated()
    assert calibration.scales == (1.0, 1.0)
    assert calibration.is_calibrated is False
    assert calibration.um_per_px_x is None
    assert "Not calibrated" in calibration.summary()
