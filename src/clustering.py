"""Cell contact graph and cluster assignment (§8).

Membership is decided from the **masks**, never from centroid distance: two
objects are neighbours when the actual pixel sets come within
``contact_distance_px`` of each other. Each object is dilated by a Euclidean
disk inside its own padded bounding box and intersected with the label image,
which is exact — ``A ⊕ disk(r) ∩ B ≠ ∅ ⟺ dist(A, B) ≤ r`` — symmetric, and
costs O(Σ bounding-box area) rather than O(N²) full-image passes.

Note that the disk is a true Euclidean structuring element, so at
``contact_distance_px = 1`` a diagonal neighbour (distance √2) is *not* a
contact. The default of 2 includes it.
"""

from __future__ import annotations

from typing import Iterable, Sequence

import numpy as np
from scipy import ndimage as ndi
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from skimage.morphology import binary_dilation, disk

from .types import ClusterRecord, ObjectRecord


def build_cell_contact_graph(
    labels: np.ndarray, contact_distance_px: int = 2
) -> list[tuple[int, int]]:
    """Undirected edges between labels whose masks are within ``contact_distance_px``.

    Returns a sorted list of ``(low_label, high_label)`` pairs, each once.
    """
    if contact_distance_px < 0:
        raise ValueError(
            f"contact_distance_px must be >= 0, got {contact_distance_px!r}."
        )

    labels = np.asarray(labels)
    height, width = labels.shape[:2]
    radius = int(contact_distance_px)
    footprint = disk(radius) if radius > 0 else None
    pad = radius + 1

    edges: set[tuple[int, int]] = set()
    if not labels.size or labels.max() <= 0:
        return []
    for index, window in enumerate(ndi.find_objects(labels)):
        if window is None:
            continue
        label = index + 1
        r0, c0 = max(0, window[0].start - pad), max(0, window[1].start - pad)
        r1, c1 = min(height, window[0].stop + pad), min(width, window[1].stop + pad)

        crop = labels[r0:r1, c0:c1]
        own = crop == label
        reach = binary_dilation(own, footprint) if footprint is not None else own

        touched = crop[reach]
        for neighbour in np.unique(touched[(touched != 0) & (touched != label)]):
            neighbour = int(neighbour)
            edges.add((min(label, neighbour), max(label, neighbour)))

    return sorted(edges)


def assign_clusters(
    object_ids: Sequence[int], edges: Iterable[tuple[int, int]]
) -> dict[int, int]:
    """Map each object ID to a cluster ID via connected components.

    Cluster IDs are 1-based and assigned in order of each component's smallest
    member ID, so the same segmentation always produces the same numbering.
    """
    ids = sorted(int(i) for i in object_ids)
    if not ids:
        return {}

    index_of = {object_id: i for i, object_id in enumerate(ids)}
    present = set(ids)

    rows, cols = [], []
    for a, b in edges:
        a, b = int(a), int(b)
        if a not in present or b not in present:
            continue  # an endpoint was removed by QC
        rows.append(index_of[a])
        cols.append(index_of[b])

    n = len(ids)
    if rows:
        data = np.ones(len(rows), dtype=np.uint8)
        graph = coo_matrix((data, (rows, cols)), shape=(n, n))
    else:
        graph = coo_matrix((n, n), dtype=np.uint8)

    _, component_of = connected_components(graph, directed=False)

    # Renumber components by their smallest member so IDs are deterministic.
    smallest: dict[int, int] = {}
    for object_id, index in index_of.items():
        component = int(component_of[index])
        if component not in smallest or object_id < smallest[component]:
            smallest[component] = object_id
    order = sorted(smallest, key=lambda c: smallest[c])
    cluster_of_component = {component: i + 1 for i, component in enumerate(order)}

    return {
        object_id: cluster_of_component[int(component_of[index])]
        for object_id, index in index_of.items()
    }


def build_clusters(
    labels: np.ndarray,
    objects: Iterable[ObjectRecord],
    contact_distance_px: int = 2,
) -> tuple[list[ClusterRecord], list[tuple[int, int]]]:
    """Build the contact graph and the resulting clusters.

    Only objects actually present in ``labels`` participate — the caller passes
    the QC-approved label image, so excluded cells neither appear in a cluster
    nor hold two other cells together.
    """
    from .morphometry import present_labels

    objects = list(objects)
    present = present_labels(labels)
    live = [o for o in objects if o.object_id in present]

    edges = build_cell_contact_graph(labels, contact_distance_px)
    cluster_of = assign_clusters([o.object_id for o in live], edges)

    status_of = {o.object_id: o.status for o in live}
    members: dict[int, list[int]] = {}
    for object_id, cluster_id in cluster_of.items():
        members.setdefault(cluster_id, []).append(object_id)

    records: list[ClusterRecord] = []
    for cluster_id in sorted(members):
        member_ids = tuple(sorted(members[cluster_id]))
        resolved = tuple(i for i in member_ids if status_of[i] == "resolved")
        unresolved = tuple(i for i in member_ids if status_of[i] != "resolved")
        records.append(
            ClusterRecord(
                cluster_id=cluster_id,
                member_ids=member_ids,
                resolved_ids=resolved,
                unresolved_ids=unresolved,
            )
        )

    # Keep only edges whose endpoints survived, for the exported contact graph.
    live_edges = [(a, b) for a, b in edges if a in present and b in present]
    return records, live_edges
