"""Synthetic phantom factories for measurement validation (§19).

These validate *measurement correctness* independently of segmentation
accuracy. They say nothing about how well Cellpose finds real cells, and no
claim about segmentation performance may rest on them (§20).

Tolerances used across the suite are empirically grounded rather than guessed.
Rasterising a shape onto a pixel grid introduces real, systematic bias, and
asserting exact equality against analytic values would fail on correct code:

    disk r=25   area +0.64%   major +0.31%   circularity 1.0023   Feret +3.32%   solidity 0.980
    disk r=50   area -0.37%   major -0.18%   circularity 1.0094   Feret +1.59%   solidity 0.989
    disk r=100  area -0.06%   major -0.03%   circularity 1.0039   Feret +0.78%   solidity 0.994

Feret's bias is one-sided: a digital diameter spans both end pixels, so it
overestimates. Phantoms therefore use radii >= 25 where discretisation error is
small, and Feret gets an asymmetric tolerance.

Discretisation error also depends on where the shape's centre falls within a
pixel. The same r=25 disk is 0.64% off when centred on a pixel centre but
1.15% off when centred on a pixel corner. Phantoms therefore centre on
``_centre()`` -- the midpoint of the index range, a pixel centre on the
even-sized canvases used here -- which is the symmetric case the table above
was measured on.
"""

from __future__ import annotations

import numpy as np
import pytest
from skimage import draw

# --- tolerances, derived from the table above ------------------------------ #

REL_TOL_AREA = 0.01
REL_TOL_AXIS = 0.01
REL_TOL_RATIO = 0.01
ABS_TOL_CIRCULARITY = 0.05
FERET_TOL_HIGH = 0.05   # digital diameters overestimate
FERET_TOL_LOW = 0.01
MIN_SOLIDITY_CONVEX = 0.97


def assert_close(actual: float, expected: float, rel: float, what: str = "") -> None:
    """Relative-tolerance assertion with a message that names the numbers."""
    assert expected != 0, f"{what}: expected value must be non-zero"
    error = abs(actual - expected) / abs(expected)
    assert error <= rel, (
        f"{what}: got {actual:.6g}, expected {expected:.6g} "
        f"({error * 100:.3f}% off, tolerance {rel * 100:.3f}%)"
    )


def assert_feret(actual: float, expected: float, what: str = "feret") -> None:
    """Feret tolerance is asymmetric because the digital bias is one-sided."""
    ratio = actual / expected
    assert 1.0 - FERET_TOL_LOW <= ratio <= 1.0 + FERET_TOL_HIGH, (
        f"{what}: got {actual:.6g}, expected {expected:.6g} "
        f"(ratio {ratio:.4f}, allowed [{1 - FERET_TOL_LOW:.3f}, {1 + FERET_TOL_HIGH:.3f}])"
    )


# --- phantom builders ------------------------------------------------------ #


def _centre(size: int) -> float:
    """Midpoint of the index range [0, size-1].

    On the even-sized canvases these phantoms use this lands on a pixel centre,
    giving the symmetric rasterisation the tolerance table was measured on.
    """
    return (size - 1) / 2.0


def disk_labels(radius: int = 50, pad: int = 20, label: int = 1) -> np.ndarray:
    """A single filled disk, centred, with clearance from the border."""
    size = 2 * radius + 2 * pad
    labels = np.zeros((size, size), dtype=np.int32)
    centre = _centre(size)
    rr, cc = draw.disk((centre, centre), radius, shape=labels.shape)
    labels[rr, cc] = label
    return labels


def ellipse_labels(
    semi_major: int = 100, semi_minor: int = 50, pad: int = 20, rotation: float = 0.0
) -> np.ndarray:
    """A single ellipse with ``semi_major`` along x (columns) when unrotated."""
    size = 2 * semi_major + 2 * pad
    labels = np.zeros((size, size), dtype=np.int32)
    centre = _centre(size)
    rr, cc = draw.ellipse(
        centre, centre, semi_minor, semi_major, shape=labels.shape, rotation=rotation
    )
    labels[rr, cc] = 1
    return labels


def rectangle_labels(height: int = 80, width: int = 160, pad: int = 20) -> np.ndarray:
    """An axis-aligned filled rectangle."""
    labels = np.zeros((height + 2 * pad, width + 2 * pad), dtype=np.int32)
    labels[pad : pad + height, pad : pad + width] = 1
    return labels


def separated_disks(
    n: int = 3, radius: int = 30, gap: int = 40, pad: int = 20
) -> np.ndarray:
    """``n`` disks in a row, each ``gap`` pixels clear of its neighbour."""
    step = 2 * radius + gap
    width = n * step + 2 * pad
    height = 2 * radius + 2 * pad
    labels = np.zeros((height, width), dtype=np.int32)
    for i in range(n):
        cx = pad + radius + i * step
        rr, cc = draw.disk((height / 2, cx), radius, shape=labels.shape)
        labels[rr, cc] = i + 1
    return labels


def disks_with_gap(radius: int = 30, gap: int = 2, pad: int = 20) -> np.ndarray:
    """Two disks whose masks are exactly ``gap`` pixels apart.

    ``gap`` is the minimum Euclidean distance between a pixel of one mask and a
    pixel of the other -- the same quantity the contact rule thresholds on. Its
    smallest meaningful value is 1 (side-by-side pixels).

    Note the centre separation is *not* ``2 * radius + gap``: ``draw.disk``
    rasterises a radius-r disk spanning centre +/- (r - 1), so the centres must
    be ``2 * (radius - 1) + gap`` apart. Deriving this from the centre distance
    alone gives a mask separation 2 px larger than intended.
    """
    centre_distance = 2 * (radius - 1) + gap
    height = 2 * radius + 2 * pad
    width = centre_distance + 2 * radius + 2 * pad
    labels = np.zeros((height, width), dtype=np.int32)
    cy = _centre(height)
    for i, cx in enumerate((pad + radius, pad + radius + centre_distance)):
        rr, cc = draw.disk((cy, cx), radius, shape=labels.shape)
        labels[rr, cc] = i + 1
    return labels


def touching_disks_binary(
    radius: int = 40, overlap: int = 20, pad: int = 20
) -> np.ndarray:
    """A dumbbell: two disks merged into ONE binary object.

    ``overlap`` is how far the disks interpenetrate; larger values give a
    fatter neck that conservative splitting should refuse to cut.
    """
    centre_distance = 2 * radius - overlap
    height = 2 * radius + 2 * pad
    width = centre_distance + 2 * radius + 2 * pad
    mask = np.zeros((height, width), dtype=bool)
    cy = height / 2
    for cx in (pad + radius, pad + radius + centre_distance):
        rr, cc = draw.disk((cy, cx), radius, shape=mask.shape)
        mask[rr, cc] = True
    return mask.astype(np.int32)


def fluorescence_phantom(
    radius: int = 26, spacing: int = 90, grid: int = 3, size: int = 320
) -> np.ndarray:
    """A greyscale image resembling a sparse fluorescence field.

    Bright blobs on a dark background with a soft edge, suitable for driving
    the threshold+watershed fallback engine end to end.
    """
    image = np.zeros((size, size), dtype=np.float32)
    yy, xx = np.mgrid[0:size, 0:size]
    start = (size - (grid - 1) * spacing) / 2
    for i in range(grid):
        for j in range(grid):
            cy, cx = start + i * spacing, start + j * spacing
            d = np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2)
            image += 200.0 * np.exp(-(d**2) / (2 * (radius / 1.6) ** 2))
    return np.clip(image, 0, 255)


# --- fixtures -------------------------------------------------------------- #


@pytest.fixture
def disk50() -> np.ndarray:
    return disk_labels(radius=50)


@pytest.fixture
def ellipse_2to1() -> np.ndarray:
    return ellipse_labels(semi_major=100, semi_minor=50)


@pytest.fixture
def three_separated() -> np.ndarray:
    return separated_disks(n=3, radius=30, gap=40)


@pytest.fixture
def phantom_image() -> np.ndarray:
    return fluorescence_phantom()
