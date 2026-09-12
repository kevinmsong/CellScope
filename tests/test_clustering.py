"""Contact-graph and cluster-assignment tests (§8)."""

from __future__ import annotations

import numpy as np
import pytest
from conftest import disks_with_gap, separated_disks

from src.clustering import assign_clusters, build_cell_contact_graph, build_clusters
from src.types import ObjectRecord


def resolved_objects(labels):
    return [
        ObjectRecord(int(v), int(v), "resolved")
        for v in np.unique(labels)
        if int(v) != 0
    ]


def cluster_sizes(clusters):
    return sorted(c.cluster_size for c in clusters)


# --------------------------------------------------------------------------- #
# Contact graph
# --------------------------------------------------------------------------- #


def test_separated_cells_form_singleton_clusters():
    labels = separated_disks(n=3, radius=30, gap=40)
    clusters, edges = build_clusters(labels, resolved_objects(labels), contact_distance_px=2)
    assert edges == []
    assert cluster_sizes(clusters) == [1, 1, 1]
    assert all(c.status == "isolated_cell" for c in clusters)
    assert all(c.cell_count == 1 for c in clusters)


def test_gap_smaller_than_contact_distance_joins_cells():
    """A 2 px gap is a contact at threshold 3, and is not at threshold 1."""
    labels = disks_with_gap(radius=30, gap=2)

    joined, _ = build_clusters(labels, resolved_objects(labels), contact_distance_px=3)
    assert cluster_sizes(joined) == [2]
    assert joined[0].status == "resolved_cluster"
    assert joined[0].cell_count == 2

    apart, _ = build_clusters(labels, resolved_objects(labels), contact_distance_px=1)
    assert cluster_sizes(apart) == [1, 1]


def test_directly_touching_cells_are_one_cluster():
    labels = disks_with_gap(radius=30, gap=1)
    clusters, edges = build_clusters(labels, resolved_objects(labels), contact_distance_px=1)
    assert cluster_sizes(clusters) == [2]
    assert len(edges) == 1


def test_contact_is_mask_based_not_centroid_based():
    """Two long bars whose centroids are far apart but whose masks touch.

    A centroid-distance rule would call these separate; §8 requires the masks
    to decide, so they are one cluster.
    """
    labels = np.zeros((60, 400), dtype=np.int32)
    labels[28:32, 10:200] = 1
    labels[28:32, 201:390] = 2

    centroids = [200 / 2 + 10 / 2, (201 + 390) / 2]
    assert abs(centroids[1] - centroids[0]) > 150  # far apart by centroid

    clusters, _ = build_clusters(labels, resolved_objects(labels), contact_distance_px=2)
    assert cluster_sizes(clusters) == [2]


def test_clustering_is_transitive():
    """A-B and B-C touching puts A and C in one cluster even though A and C do not."""
    labels = np.zeros((60, 300), dtype=np.int32)
    labels[20:40, 10:100] = 1
    labels[20:40, 101:190] = 2
    labels[20:40, 191:280] = 3

    clusters, edges = build_clusters(labels, resolved_objects(labels), contact_distance_px=2)
    assert cluster_sizes(clusters) == [3]
    assert sorted(edges) == [(1, 2), (2, 3)]


def test_diagonal_contact_needs_distance_two():
    """disk() is a true Euclidean footprint, so a diagonal neighbour at sqrt(2)
    is not a contact at distance 1 but is at distance 2."""
    labels = np.zeros((20, 20), dtype=np.int32)
    labels[5:10, 5:10] = 1
    labels[10:15, 10:15] = 2  # touches only at a corner

    assert build_cell_contact_graph(labels, 1) == []
    assert build_cell_contact_graph(labels, 2) == [(1, 2)]


def test_contact_distance_zero_finds_no_contacts():
    """Dilating by nothing can never reach a neighbouring label."""
    labels = disks_with_gap(radius=30, gap=1)
    assert build_cell_contact_graph(labels, 0) == []


@pytest.mark.parametrize("separation", [1, 2, 3, 4])
def test_contact_fires_exactly_when_separation_is_within_threshold(separation):
    """The rule is dist(A, B) <= r, verified at the boundary in both directions."""
    labels = disks_with_gap(radius=30, gap=separation)
    assert build_cell_contact_graph(labels, separation) == [(1, 2)]
    assert build_cell_contact_graph(labels, separation - 1) == []


def test_negative_contact_distance_is_rejected():
    with pytest.raises(ValueError, match="must be >= 0"):
        build_cell_contact_graph(np.zeros((10, 10), dtype=np.int32), -1)


# --------------------------------------------------------------------------- #
# Cluster IDs
# --------------------------------------------------------------------------- #


def test_cluster_ids_are_deterministic_and_ordered_by_smallest_member():
    mapping = assign_clusters([5, 1, 9, 3], [(1, 9)])
    assert mapping[1] == mapping[9] == 1  # component containing the smallest ID
    assert mapping[3] == 2
    assert mapping[5] == 3


def test_assign_clusters_ignores_edges_to_removed_objects():
    """QC can delete an endpoint; a dangling edge must not resurrect it."""
    mapping = assign_clusters([1, 2], [(1, 2), (2, 99)])
    assert set(mapping) == {1, 2}
    assert mapping[1] == mapping[2]


def test_empty_input_gives_empty_clusters():
    assert assign_clusters([], []) == {}
    labels = np.zeros((20, 20), dtype=np.int32)
    clusters, edges = build_clusters(labels, [], contact_distance_px=2)
    assert clusters == [] and edges == []
