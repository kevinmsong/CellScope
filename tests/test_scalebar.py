"""Scale-bar detection and reference-frame calibration (§4A)."""

from __future__ import annotations

import numpy as np
import pytest

from src.calibration import (
    calibration_from_reference_bar,
    check_reference_compatible,
)
from src.scalebar import detect_scale_bar
from src.types import ImageRecord


def bar_image(width=800, height=600, bar_px=111, colour=(255, 255, 0)):
    """A dark frame with a solid horizontal bar, as microscope software exports.

    The bar is right-aligned with a margin, so it always fits the frame however
    long it is -- a clipped bar would measure short and silently weaken the test.
    """
    image = np.zeros((height, width, 3), dtype=np.uint8)
    start = width - 50 - bar_px
    assert start > 0, "bar does not fit the test frame"
    image[520:528, start : start + bar_px] = colour
    return image


def record(width=800, height=600, name="ref.jpg"):
    return ImageRecord(name, "0" * 64, height, width, "uint8", 3,
                       np.zeros((height, width, 3), np.uint8))


def test_detects_a_yellow_bar():
    detection = detect_scale_bar(bar_image(bar_px=111))
    assert detection.found
    assert detection.length_px == pytest.approx(111)
    assert detection.colour == "yellow"


def test_detects_a_white_bar():
    detection = detect_scale_bar(bar_image(colour=(255, 255, 255)))
    assert detection.found
    assert detection.colour == "white"


def test_reports_endpoints_for_the_manual_path():
    detection = detect_scale_bar(bar_image(bar_px=111))
    (x1, _), (x2, _) = detection.endpoints
    assert x2 - x1 == pytest.approx(110)  # inclusive span of 111 px


def test_ignores_text_and_picks_the_bar():
    """The caption is also bright; the bar is what is long, thin and solid."""
    image = bar_image(bar_px=111)
    image[540:574, 600:640] = (255, 255, 0)  # chunky glyphs, like a caption
    image[540:574, 650:690] = (255, 255, 0)
    detection = detect_scale_bar(image)
    assert detection.length_px == pytest.approx(111)


def test_no_bar_is_reported_honestly():
    detection = detect_scale_bar(np.zeros((600, 800, 3), np.uint8))
    assert detection.found is False
    assert "no scale bar" in detection.note


def test_detection_is_stable_across_bar_lengths():
    for length in (40, 80, 111, 250):
        assert detect_scale_bar(bar_image(bar_px=length)).length_px == pytest.approx(length)


def test_reference_calibration_arithmetic():
    """The shipped reference frames: 111 px is 50 µm at 40x, 100 µm at 20x."""
    assert calibration_from_reference_bar(111.0, 50.0).um_per_px_x == pytest.approx(50 / 111)
    assert calibration_from_reference_bar(111.0, 100.0).um_per_px_x == pytest.approx(100 / 111)


def test_forty_x_is_exactly_twice_twenty_x():
    """A physical sanity check: the same 111 px bar at 2x magnification."""
    twenty = calibration_from_reference_bar(111.0, 100.0).um_per_px_x
    forty = calibration_from_reference_bar(111.0, 50.0).um_per_px_x
    assert twenty / forty == pytest.approx(2.0)


@pytest.mark.parametrize("reference_size,sample_size", [((800, 600), (800, 800)), ((1024, 1024), (800, 600))])
def test_different_crop_dimensions_preserve_reference_pixel_scale(reference_size, sample_size):
    reference = record(*reference_size)
    sample = record(*sample_size, "sample.jpg")
    check_reference_compatible(reference, sample)
    calibration = calibration_from_reference_bar(111.0, 100.0, reference, sample)
    assert calibration.scales == pytest.approx((100 / 111, 100 / 111))
    assert calibration.params["reference_dimensions"] == list(reference_size)
    assert calibration.params["sample_dimensions"] == list(sample_size)


def test_matching_dimensions_are_accepted():
    calibration = calibration_from_reference_bar(
        111.0, 50.0, record(800, 600), record(800, 600, "sample.jpg")
    )
    assert calibration.is_calibrated
    assert calibration.method == "reference_scale_bar"


def test_reference_identity_is_recorded_for_provenance():
    """§18: the calibration must be traceable to the exact frame it came from."""
    reference = record(800, 600, "40x_ruler.jpg")
    calibration = calibration_from_reference_bar(
        111.0, 50.0, reference, record(800, 600, "sample.jpg")
    )
    assert calibration.params["reference_filename"] == "40x_ruler.jpg"
    assert calibration.params["reference_sha256"] == "0" * 64
    assert calibration.params["reference_dimensions"] == [800, 600]
    assert calibration.params["bar_length_px"] == pytest.approx(111.0)


@pytest.mark.parametrize("bad", [0, -1, None, float("nan"), float("inf")])
def test_non_positive_inputs_are_rejected(bad):
    with pytest.raises(ValueError):
        calibration_from_reference_bar(bad, 50.0)
    with pytest.raises(ValueError):
        calibration_from_reference_bar(111.0, bad)
