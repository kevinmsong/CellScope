"""Optimised implementations must reproduce the originals exactly.

Each test runs the current code and the frozen pre-optimisation copy in
``reference_impl`` on the same inputs and demands identical output. Floating
point columns are compared with a relative tolerance of 1e-12, which admits
nothing but rounding in the last bits.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import reference_impl as ref
from skimage import draw

from src.calibration import calibration_from_resolution
from src.clustering import build_cell_contact_graph, build_clusters
from src.morphometry import measure_cells, measure_clusters
from src.segmentation import split_touching_cells
from src.types import Calibration, ObjectRecord


def crowded_labels(seed: int, size=(180, 220), n=40, holes=True) -> np.ndarray:
    """Random ellipses, many touching, some clipped by the image edge."""
    rng = np.random.default_rng(seed)
    labels = np.zeros(size, np.int32)
    for index in range(1, n + 1):
        rr, cc = draw.ellipse(
            rng.uniform(-5, size[0] + 5), rng.uniform(-5, size[1] + 5),
            rng.uniform(3, 16), rng.uniform(3, 22), shape=size,
            rotation=rng.uniform(-1.5, 1.5),
        )
        labels[rr, cc] = index
    if holes:
        labels[40:44, 40:44] = 0
    # Single-pixel and two-pixel objects.
    labels[2, size[1] // 2] = n + 1
    labels[size[0] - 3, 5:7] = n + 2
    return labels


def _objects(labels, unresolved_every=5):
    ids = sorted(int(v) for v in np.unique(labels) if v)
    return [
        ObjectRecord(i, i, "unresolved_cluster" if i % unresolved_every == 0 else "resolved")
        for i in ids
    ]


CALIBRATIONS = [
    Calibration(),
    calibration_from_resolution(0.45),
    calibration_from_resolution(0.4, 0.6),
]


def _assert_frames_equal(new: pd.DataFrame, old: pd.DataFrame):
    assert list(new.columns) == list(old.columns)
    pd.testing.assert_frame_equal(new, old, check_exact=False, rtol=1e-12, atol=0)


@pytest.mark.parametrize("seed", range(6))
@pytest.mark.parametrize("calibration", CALIBRATIONS, ids=["px", "iso", "aniso"])
@pytest.mark.parametrize("distance", [0, 2, 5])
def test_measure_clusters_matches_reference(seed, calibration, distance):
    labels = crowded_labels(seed)
    objects = _objects(labels)
    clusters, _ = build_clusters(labels, objects, distance)
    cells = measure_cells(labels, objects, clusters, calibration)
    new = measure_clusters(labels, objects, clusters, calibration, cells, distance)
    old = ref.measure_clusters(labels, objects, clusters, calibration, cells, distance)
    _assert_frames_equal(new, old)


def test_measure_clusters_on_an_empty_image():
    labels = np.zeros((20, 20), np.int32)
    new = measure_clusters(labels, [], [], Calibration(), pd.DataFrame(), 2)
    old = ref.measure_clusters(labels, [], [], Calibration(), pd.DataFrame(), 2)
    assert new.empty and old.empty


@pytest.mark.parametrize("seed", range(6))
@pytest.mark.parametrize("distance", [0, 1, 2, 4])
def test_contact_graph_matches_reference(seed, distance):
    labels = crowded_labels(seed, holes=False)
    assert build_cell_contact_graph(labels, distance) == ref.build_cell_contact_graph(
        labels, distance
    )


@pytest.mark.parametrize("seed", range(4))
def test_measure_cells_with_a_cache_matches_without(seed):
    """Geometry reused across QC changes must equal geometry measured fresh."""
    labels = crowded_labels(seed)
    objects = _objects(labels)
    calibration = calibration_from_resolution(0.4, 0.6)
    clusters, _ = build_clusters(labels, objects, 2)
    cache: dict = {}
    first = measure_cells(labels, objects, clusters, calibration, cache=cache)
    # Remove a third of the cells, as QC would, and measure again from the cache.
    kept = objects[::3]
    approved = np.where(np.isin(labels, [o.object_id for o in kept]), labels, 0)
    clusters, _ = build_clusters(approved, kept, 2)
    cached = measure_cells(approved, kept, clusters, calibration, cache=cache)
    fresh = measure_cells(approved, kept, clusters, calibration)
    _assert_frames_equal(cached, fresh)
    assert len(first) > len(cached)


def test_drop_small_labels_matches_reference():
    from src.segmentation import _drop_small_labels

    labels = crowded_labels(3)
    for minimum in (1, 5, 50, 400):
        np.testing.assert_array_equal(
            _drop_small_labels(labels, minimum), ref.drop_small_labels(labels, minimum)
        )


@pytest.mark.parametrize("seed", range(4))
def test_nuclear_edges_match_reference(seed):
    from src.nuclei import nuclear_edges

    labels = crowded_labels(seed, holes=False)
    for gap in (0, 1):
        assert nuclear_edges(labels, gap) == ref.nuclear_edges(labels, gap)


@pytest.mark.parametrize("seed", range(3))
def test_nuclear_seeding_matches_reference(seed):
    from scipy import ndimage as ndi
    from skimage.feature import peak_local_max

    from src.nuclei import _seed_and_filter

    mask = crowded_labels(seed) > 0
    distance = ndi.distance_transform_edt(mask)
    # Few seeds, so some components must be seeded by the fallback.
    coordinates = peak_local_max(distance, min_distance=15, labels=mask,
                                 exclude_border=False)[::2]
    np.testing.assert_array_equal(
        _seed_and_filter(mask, distance, coordinates, 30),
        ref.nuclear_seed_and_filter(mask, distance, coordinates, 30),
    )


@pytest.mark.parametrize("seed", range(3))
@pytest.mark.parametrize("show_ids", [True, False])
def test_overlay_matches_reference(seed, show_ids):
    from src.visualize import make_overlay

    labels = crowded_labels(seed)
    objects = _objects(labels)
    excluded = {o.object_id for o in objects[::4]}
    included = [o for o in objects if o.object_id not in excluded]
    clusters, _ = build_clusters(np.where(np.isin(labels, list(excluded)), 0, labels),
                                 included, 2)
    rng = np.random.default_rng(seed)
    display = rng.integers(0, 255, (*labels.shape, 3), dtype=np.uint8)
    for alpha in (0.0, 0.28, 0.8):
        new = make_overlay(display, labels, included, clusters, alpha=alpha,
                           show_cell_ids=show_ids, show_cluster_ids=show_ids,
                           excluded_ids=excluded)
        old = ref.make_overlay(display, labels, included, clusters, alpha=alpha,
                               show_cell_ids=show_ids, show_cluster_ids=show_ids,
                               excluded_ids=excluded)
        np.testing.assert_array_equal(new, old)


def test_overlay_on_a_downscaled_display_matches_reference():
    from src.visualize import make_overlay

    labels = np.kron(crowded_labels(1), np.ones((3, 3), np.int32))
    objects = _objects(labels)
    clusters, _ = build_clusters(labels, objects, 2)
    display = np.full((labels.shape[0] // 3, labels.shape[1] // 3, 3), 90, np.uint8)
    np.testing.assert_array_equal(
        make_overlay(display, labels, objects, clusters),
        ref.make_overlay(display, labels, objects, clusters),
    )


#: sha256 prefixes of the splitter's output recorded before the performance
#: pass, over the fields that existed then.
SPLIT_GOLDEN = {
    (0, True): "ffe67ca860828cd5",
    (0, False): "4f3e2c09ab3f004f",
    (1, True): "7f9af2a85a6cc1be",
    (1, False): "a2b341703567baf6",
    (2, True): "2b95db4592731813",
    (2, False): "3a9b1ce44fca8f4f",
}


@pytest.mark.parametrize("seed,enabled", sorted(SPLIT_GOLDEN))
def test_split_output_is_unchanged(seed, enabled):
    import hashlib

    from src.types import SplitParams

    out, objects = split_touching_cells(crowded_labels(seed), SplitParams(enabled=enabled))
    key = [(o.object_id, o.raw_label, o.status, o.split_from, o.touches_border,
            o.flagged_suspicious, o.unresolved_reason) for o in objects]
    digest = hashlib.sha256(out.tobytes() + repr(key).encode()).hexdigest()[:16]
    assert digest == SPLIT_GOLDEN[(seed, enabled)]
