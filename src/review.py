"""Reversible review edits and conservative, explicitly manual mask corrections."""
from copy import deepcopy
from dataclasses import replace
import hashlib
import json

import numpy as np
from skimage.draw import polygon
from skimage.segmentation import watershed
from .types import ObjectRecord, utc_now


def signature(session):
    values = [session.segmentation_version, session.qc.version, session.channel,
              session.calibration.describe(), session.segmentation_params.describe(),
              session.preprocess_params.describe(), session.split_params.describe(),
              session.cluster_params.describe()]
    return hashlib.sha256(json.dumps(values, sort_keys=True).encode()).hexdigest()


def validate_review(session):
    if session.reviewed and session.reviewed_signature and session.reviewed_signature != signature(session):
        session.reviewed = False


def snapshot(session):
    return (session.object_labels.copy() if session.object_labels is not None else None,
            list(session.objects), set(session.qc.excluded_ids))


def checkpoint(session):
    session.undo_stack.append(snapshot(session))
    session.undo_stack[:] = session.undo_stack[-12:]
    session.redo_stack.clear()
    session.reviewed = False


def history(session, redo=False):
    source, target = (session.redo_stack, session.undo_stack) if redo else (session.undo_stack, session.redo_stack)
    if not source:
        return False
    target.append(snapshot(session))
    labels, objects, excluded = source.pop()
    for oid in session.qc.excluded_ids - excluded:
        session.qc.restore(oid, "redo" if redo else "undo")
    for oid in excluded - session.qc.excluded_ids:
        session.qc.exclude(oid, "redo" if redo else "undo")
    session.object_labels, session.objects = labels, objects
    session.segmentation_version += 1
    session.results_cache = None
    session.reviewed = False
    session.edit_log.append({"action": "redo" if redo else "undo", "timestamp": utc_now()})
    return True


def correct_mask(session, operation, ids, points):
    if not session.has_segmentation:
        raise ValueError("Segment this image first.")
    labels = session.object_labels.copy()
    ids = list(dict.fromkeys(map(int, ids)))
    known = {o.object_id for o in session.objects}
    if not ids or not set(ids) <= known:
        raise ValueError("Choose existing cell IDs.")
    new_id = max(known, default=0) + 1
    if operation == "Merge":
        if len(ids) < 2:
            raise ValueError("Merge requires at least two IDs.")
        from scipy.ndimage import label as connected
        if connected(np.isin(labels, ids))[1] != 1:
            raise ValueError("Only touching cells can be merged.")
        if len({session.qc.is_included(i) for i in ids}) > 1:
            raise ValueError("Restore or exclude the selected cells consistently before merging.")
        labels[np.isin(labels, ids)] = ids[0]
    elif operation == "Split":
        if len(ids) != 1 or len(points) != 2:
            raise ValueError("Split requires one ID and two seed points inside it.")
        mask = labels == ids[0]
        markers = np.zeros(labels.shape, np.int32)
        for i, (x, y) in enumerate(points, 1):
            if not (0 <= y < labels.shape[0] and 0 <= x < labels.shape[1]) or not mask[y, x]:
                raise ValueError("Both seeds must be inside the selected cell.")
            markers[y, x] = i
        if len(np.unique(markers)) != 3:
            raise ValueError("Use two distinct seeds.")
        from scipy.ndimage import distance_transform_edt
        split = watershed(-distance_transform_edt(mask), markers, mask=mask)
        labels[split == 2] = new_id
    elif operation == "Replace boundary":
        if len(ids) != 1 or len(points) < 3:
            raise ValueError("Boundary replacement requires one ID and at least three polygon vertices.")
        rr, cc = polygon([p[1] for p in points], [p[0] for p in points], labels.shape)
        if not len(rr):
            raise ValueError("The polygon is empty.")
        if np.any((labels[rr, cc] != 0) & (labels[rr, cc] != ids[0])):
            raise ValueError("The polygon overlaps another cell. Adjust its vertices.")
        labels[labels == ids[0]] = 0
        labels[rr, cc] = ids[0]
    else:
        raise ValueError("Unknown correction operation.")
    checkpoint(session)
    previous = {o.object_id: o for o in session.objects}
    objects = []
    border_ids = set(np.concatenate([labels[0], labels[-1], labels[:, 0], labels[:, -1]]))
    for oid in sorted(set(np.unique(labels)) - {0}):
        obj = previous.get(int(oid), ObjectRecord(int(oid), None, "resolved", split_from=ids[0]))
        if oid in ids or oid == new_id:
            obj = replace(obj, status="resolved", touches_border=oid in border_ids, unresolved_reason=None)
        objects.append(obj)
    session.object_labels = labels
    session.objects = objects
    # Split children inherit their parent's inclusion decision.
    if operation == "Split" and ids[0] in session.qc.excluded_ids:
        session.qc.exclude(new_id, "inherits excluded split parent")
    session.segmentation_version += 1
    session.results_cache = None
    session.edit_log.append({"action": operation, "ids": ids, "points": points, "timestamp": utc_now()})


def viewport(session, base_shape):
    h, w = base_shape[:2]
    zoom = max(1.0, min(8.0, float(session.review_zoom)))
    cw, ch = max(1, round(w / zoom)), max(1, round(h / zoom))
    x = round((w - cw) * session.review_pan_x / 100)
    y = round((h - ch) * session.review_pan_y / 100)
    return x, y, cw, ch


def display_view(session, overlay):
    from PIL import Image, ImageDraw
    x, y, w, h = viewport(session, overlay.shape)
    image = Image.fromarray(overlay[y:y+h, x:x+w]).resize(
        (overlay.shape[1], overlay.shape[0]), Image.Resampling.NEAREST)
    return np.asarray(image)


def click_position(session, point, base_shape):
    x, y, w, h = viewport(session, base_shape)
    px, py = map(int, point[:2])
    if not (0 <= px < base_shape[1] and 0 <= py < base_shape[0]):
        return None
    return x + min(w-1, int((px + .5) * w / base_shape[1])), y + min(h-1, int((py + .5) * h / base_shape[0]))
