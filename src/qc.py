"""Segmentation quality control (§13).

Two label images coexist for the life of a session: the raw segmentation and
the QC-approved one. :func:`apply_qc` is a pure function from the first to the
second, so the raw array is never edited in place and every QC verdict is
reversible.

Every action is appended to a log carrying the object ID, what was done, and
why -- the record §13 requires and §18 traces through.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np
import pandas as pd

from .types import ClusterRecord, ObjectRecord, QCState


def apply_qc(labels: np.ndarray, qc_state: QCState) -> np.ndarray:
    """Return a new label image with excluded objects removed.

    The input is never modified. Excluded objects become background; surviving
    objects keep their IDs, so a measurement can always be traced back to the
    same ID in the raw mask.
    """
    approved = np.asarray(labels).copy()
    if not qc_state.excluded_ids:
        return approved
    excluded = np.asarray(sorted(qc_state.excluded_ids), dtype=approved.dtype)
    approved[np.isin(approved, excluded)] = 0
    return approved


def exclude_object(qc_state, object_id, reason="manual exclusion") -> bool:
    return qc_state.exclude(int(object_id), reason, scope="object")


def restore_object(qc_state, object_id, reason="manual restore") -> bool:
    return qc_state.restore(int(object_id), reason, scope="object")


def exclude_cluster(qc_state, cluster: ClusterRecord, reason: str | None = None) -> int:
    """Exclude every member of a cluster. Returns how many were newly excluded."""
    reason = reason or "member of excluded cluster {}".format(cluster.cluster_id)
    return sum(
        qc_state.exclude(object_id, reason, scope="cluster")
        for object_id in cluster.member_ids
    )


def exclude_border_touching(qc_state, objects, reason: str | None = None) -> int:
    """Exclude objects clipped by the image edge.

    Their morphology is truncated by the field of view rather than by biology,
    so including them biases size distributions downwards.
    """
    reason = reason or "touches image border (morphology truncated by the field of view)"
    return sum(
        qc_state.exclude(obj.object_id, reason, scope="filter")
        for obj in objects
        if obj.touches_border
    )


def exclude_by_area(qc_state, labels, min_area=None, max_area=None) -> int:
    """Exclude objects outside an area range, in pixels.

    Areas are counted from the label image directly, so the filter matches the
    mask actually being measured.
    """
    if min_area is None and max_area is None:
        return 0

    counts = np.bincount(np.asarray(labels).ravel())
    excluded = 0
    for object_id, area in enumerate(counts):
        if object_id == 0 or area == 0:
            continue
        if min_area is not None and area < min_area:
            excluded += qc_state.exclude(
                object_id,
                "area {} px below the {:.0f} px minimum".format(area, min_area),
                scope="filter",
            )
        elif max_area is not None and area > max_area:
            excluded += qc_state.exclude(
                object_id,
                "area {} px above the {:.0f} px maximum".format(area, max_area),
                scope="filter",
            )
    return excluded


def qc_log_dataframe(qc_state) -> pd.DataFrame:
    """The QC log as a table, one row per action (§17)."""
    columns = ["timestamp", "object_id", "action", "scope", "reason"]
    if not qc_state.log:
        return pd.DataFrame(columns=columns)
    return pd.DataFrame(
        [
            {
                "timestamp": entry.timestamp,
                "object_id": entry.object_id,
                "action": entry.action,
                "scope": entry.scope,
                "reason": entry.reason,
            }
            for entry in qc_state.log
        ],
        columns=columns,
    )


def qc_counts(objects: Sequence[ObjectRecord], qc_state: QCState) -> dict[str, int]:
    """Headline counts for the status bar and the image summary."""
    included = [o for o in objects if qc_state.is_included(o.object_id)]
    return {
        "objects_detected": len(objects),
        "objects_excluded": len(objects) - len(included),
        "objects_accepted": len(included),
        "cells_accepted": sum(1 for o in included if o.is_resolved),
        "unresolved_accepted": sum(1 for o in included if not o.is_resolved),
    }
