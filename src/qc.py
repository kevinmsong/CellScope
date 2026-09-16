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

from .types import ClusterRecord, ObjectRecord, QCParams, QCState

#: Reasons recorded by the automatic rules. The border text predates the
#: constant and is kept verbatim so existing logs keep their meaning.
REASON_BORDER = "automatically excluded: touches image border"
REASON_ANNOTATION = "automatically excluded: touches burned-in scale bar or caption"
REASON_RULE_OFF = "automatic rule no longer applies"
AUTOMATIC_REASONS = frozenset({REASON_BORDER, REASON_ANNOTATION})


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


def exclude_annotation_touching(qc_state, objects, reason: str = REASON_ANNOTATION) -> int:
    """Exclude objects that touch a scale bar or caption burned into the image.

    The annotation is drawn over the cells, so whatever lies under it is
    hidden; such an object's morphology is truncated just as a border object's
    is. Objects that *are* the bar or its glyphs are caught by the same rule.
    """
    return sum(
        qc_state.exclude(obj.object_id, reason, scope="filter")
        for obj in objects
        if obj.touches_annotation
    )


def _last_actions(qc_state: QCState) -> dict[int, object]:
    last = {}
    for entry in qc_state.log:
        if entry.object_id is not None:
            last[int(entry.object_id)] = entry
    return last


def automatic_reason(obj: ObjectRecord, params: QCParams) -> str | None:
    """The automatic rule that applies to an object, or ``None``."""
    if params.exclude_border and obj.touches_border:
        return REASON_BORDER
    if params.exclude_annotations and obj.touches_annotation:
        return REASON_ANNOTATION
    return None


def apply_automatic_qc(qc_state: QCState, objects, params: QCParams, only=None) -> dict[str, int]:
    """Bring automatic exclusions in line with the current rules.

    * An object a rule applies to is excluded -- unless the researcher restored
      it by hand, which always wins.
    * An object excluded *only* by an automatic rule that no longer applies (the
      rule was switched off, or a correction moved the cell) is restored.
    * Manual exclusions are never touched.

    ``only`` limits the pass to some object IDs, as after a mask correction.
    Returns how many objects were excluded and restored.
    """
    last = _last_actions(qc_state)
    counts = {"excluded": 0, "restored": 0}
    wanted = None if only is None else {int(i) for i in only}
    for obj in objects:
        if wanted is not None and obj.object_id not in wanted:
            continue
        reason = automatic_reason(obj, params)
        previous = last.get(obj.object_id)
        manually_restored = (
            previous is not None and previous.action == "restore" and previous.scope != "filter"
        )
        if reason and not manually_restored:
            counts["excluded"] += qc_state.exclude(obj.object_id, reason, scope="filter")
        elif (
            not reason
            and obj.object_id in qc_state.excluded_ids
            and previous is not None
            and previous.action == "exclude"
            and previous.reason in AUTOMATIC_REASONS
        ):
            counts["restored"] += qc_state.restore(obj.object_id, REASON_RULE_OFF, scope="filter")
    return counts


def exclusion_reasons(qc_state: QCState) -> dict[int, str]:
    """Why each currently excluded object is excluded (its latest exclusion)."""
    reasons = {}
    for entry in qc_state.log:
        if entry.object_id is not None and entry.action == "exclude":
            reasons[int(entry.object_id)] = entry.reason
    return {oid: reasons.get(oid, "") for oid in qc_state.excluded_ids}


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


def exclusion_counts(objects: Sequence[ObjectRecord], qc_state: QCState) -> dict[str, int]:
    """Excluded objects broken down by why they were excluded."""
    ids = {o.object_id for o in objects}
    reasons = [r for oid, r in exclusion_reasons(qc_state).items() if oid in ids]
    return {
        "excluded_border": sum(1 for r in reasons if r == REASON_BORDER),
        "excluded_annotation": sum(1 for r in reasons if r == REASON_ANNOTATION),
        "excluded_other": sum(1 for r in reasons if r not in AUTOMATIC_REASONS),
    }
