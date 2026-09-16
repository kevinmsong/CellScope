"""Caches may only ever return what a fresh computation would.

Random sequences of the edits a reviewer makes -- exclude, restore, merge,
split, redraw a boundary, undo, redo, recalibrate, change the contact distance --
are applied to a segmented image. After every step, the cached results and the
pooled batch tables are compared with a computation that bypasses every cache.
"""

from __future__ import annotations

import random
from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from src import morphometry
from src.batch import pooled_cells, pooled_clusters
from src.calibration import calibration_from_resolution, uncalibrated
from src.perf import image_record, synthetic_field
from src.pipeline import compute_results, excluded_cell_measurements, run_segmentation
from src.review import checkpoint, correct_mask, history
from src.types import AnalysisSession, BatchSession


def _batch(n=2, seed=0):
    batch = BatchSession()
    for index in range(n):
        rgb = synthetic_field(360, 360, n_cells=18, seed=seed + index)
        session = AnalysisSession()
        session.image = image_record(rgb, "f{}.png".format(index))
        session.channel = "red"
        batch.add(session)
        run_segmentation(session, rgb[..., 0].astype(np.float32), engine="threshold_watershed")
    return batch


def _fresh(session):
    return compute_results(session, use_cache=False)


def _same(a: pd.DataFrame, b: pd.DataFrame):
    pd.testing.assert_frame_equal(a.reset_index(drop=True), b.reset_index(drop=True))


def _pooled_fresh(batch):
    frames = []
    for s in batch.images:
        cells = _fresh(s).cells
        if len(cells):
            cells = cells.copy()
            cells.insert(0, "image_id", s.display_name)
            cells.insert(1, "well", s.well)
            frames.append(cells)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def _random_edit(rng: random.Random, session: AnalysisSession) -> str:
    ids = sorted(o.object_id for o in session.objects)
    action = rng.choice(["exclude", "restore", "merge", "split", "boundary", "undo", "redo",
                         "calibrate", "contact", "well"])
    if action in ("exclude", "restore") and ids:
        checkpoint(session)
        target = rng.choice(ids)
        (session.qc.exclude if action == "exclude" else session.qc.restore)(target, "test")
    elif action == "merge":
        for a in ids:
            neighbours = set(np.unique(
                session.object_labels[np.roll(session.object_labels == a, 1, axis=1)]
            )) - {0, a}
            if neighbours:
                b = min(neighbours)
                if session.qc.is_included(a) == session.qc.is_included(b):
                    try:
                        correct_mask(session, "Merge", [a, b], [])
                    except ValueError:
                        pass
                    break
    elif action in ("split", "boundary") and ids:
        target = rng.choice(ids)
        rows, cols = np.nonzero(session.object_labels == target)
        if len(rows) > 20:
            try:
                if action == "split":
                    order = np.argsort(cols)
                    points = [(int(cols[order[0]]), int(rows[order[0]])),
                              (int(cols[order[-1]]), int(rows[order[-1]]))]
                    correct_mask(session, "Split", [target], points)
                else:
                    points = [(int(cols.min()), int(rows.min())), (int(cols.max()), int(rows.min())),
                              (int(cols.min()), int(rows.max()))]
                    correct_mask(session, "Replace boundary", [target], points)
            except ValueError:
                pass
    elif action == "undo":
        history(session)
    elif action == "redo":
        history(session, redo=True)
    elif action == "calibrate":
        session.calibration = rng.choice([
            uncalibrated(), calibration_from_resolution(0.45),
            calibration_from_resolution(0.4, 0.55),
        ])
    elif action == "contact":
        session.cluster_params = replace(session.cluster_params,
                                         contact_distance_px=rng.choice([0, 1, 2, 4]))
    elif action == "well":
        session.well = rng.choice(["A1", "B2", ""])
    return action


@pytest.mark.parametrize("seed", range(4))
def test_cached_results_always_equal_fresh_results(seed):
    rng = random.Random(seed)
    batch = _batch(seed=seed)
    for _ in range(30):
        session = rng.choice(batch.images)
        action = _random_edit(rng, session)
        cached = compute_results(session)
        fresh = _fresh(session)
        _same(cached.cells, fresh.cells)
        _same(cached.cluster_summary, fresh.cluster_summary)
        _same(cached.image_summary, fresh.image_summary)
        assert cached.counts == fresh.counts, action
        np.testing.assert_array_equal(cached.qc_labels, fresh.qc_labels)
        _same(pooled_cells(batch), _pooled_fresh(batch))


def test_excluded_measurements_match_a_direct_measurement():
    session = _batch(1).images[0]
    target = session.objects[0].object_id
    session.qc.exclude(target, "test")
    compute_results(session)             # warm the geometry cache
    cached = excluded_cell_measurements(session)
    direct = morphometry.measure_cells(
        session.object_labels, [o for o in session.objects if o.object_id in session.qc.excluded_ids],
        [], session.calibration,
    )
    _same(cached, direct)


def test_an_exclusion_reuses_the_geometry_of_untouched_cells(monkeypatch):
    session = _batch(1).images[0]
    compute_results(session)
    calls = {"cells": 0, "clusters": 0}
    original_cell = morphometry._cell_geometry
    original_cluster = morphometry._cluster_geometry_cropped

    def count_cell(*args, **kwargs):
        calls["cells"] += 1
        return original_cell(*args, **kwargs)

    def count_cluster(*args, **kwargs):
        calls["clusters"] += 1
        return original_cluster(*args, **kwargs)

    monkeypatch.setattr(morphometry, "_cell_geometry", count_cell)
    monkeypatch.setattr(morphometry, "_cluster_geometry_cropped", count_cluster)

    target = next(o.object_id for o in session.objects if session.qc.is_included(o.object_id))
    session.qc.exclude(target, "test")
    compute_results(session)
    assert calls["cells"] == 0, "no cell's own geometry changed"
    assert calls["clusters"] <= 3, "only the edited cluster should be re-measured"


def test_recalibrating_discards_the_geometry():
    session = _batch(1).images[0]
    before = compute_results(session).cells
    session.calibration = calibration_from_resolution(0.5)
    after = compute_results(session).cells
    assert "area_um2" in after and "area_um2" not in before


def test_pooled_tables_are_memoised_until_something_changes():
    batch = _batch(2)
    first = pooled_cells(batch)
    assert pooled_cells(batch) is first
    assert pooled_clusters(batch) is pooled_clusters(batch)

    batch.images[0].qc.exclude(batch.images[0].objects[0].object_id, "test")
    assert pooled_cells(batch) is not first

    second = pooled_cells(batch)
    batch.images[1].well = "C3"
    third = pooled_cells(batch)
    assert third is not second
    assert set(third[third.image_id == "f1.png"].well) == {"C3"}
