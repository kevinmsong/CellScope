"""Project files: schema migration, damaged files, atomic writes, and autosave."""

from __future__ import annotations

import io
import json
import threading
import time
import zipfile

import numpy as np
import pytest

import src.project as project
from src.autosave import Autosaver, recovery_path
from src.perf import image_record, synthetic_field
from src.pipeline import compute_results, run_segmentation
from src.project import ProjectError, load_project, read_header, save_project
from src.review import signature
from src.types import AnalysisSession, BatchSession


@pytest.fixture
def batch():
    b = BatchSession(label="Plate 4")
    for index in range(2):
        rgb = synthetic_field(200, 200, 6, seed=index, ruler=False)
        s = AnalysisSession()
        s.image = image_record(rgb, "f{}.png".format(index))
        s.channel = "red"
        b.add(s)
        run_segmentation(s, rgb[..., 0].astype(np.float32), engine="threshold_watershed")
    b.images[0].qc.exclude(b.images[0].objects[0].object_id, "artifact")
    b.images[0].reviewed = True
    b.images[0].reviewed_signature = signature(b.images[0])
    return b


def _manifest(path):
    with zipfile.ZipFile(path) as archive:
        return json.loads(archive.read("project.json"))


def _rewrite(path, manifest, drop=()):
    """Rewrite a project with an edited manifest (and optionally members removed)."""
    with zipfile.ZipFile(path) as archive:
        members = {n: archive.read(n) for n in archive.namelist() if n not in drop}
    members["project.json"] = json.dumps(manifest).encode()
    with zipfile.ZipFile(path, "w") as archive:
        for name, data in members.items():
            archive.writestr(name, data)


def _as_schema_1(manifest):
    """What a 0.1 project looked like: no header, display copies stored, no new fields."""
    batch = manifest["batch"]
    for image in batch["fields"]["images"]:
        fields = image["fields"]
        for key in ("qc_params", "annotation", "segmented_with"):
            fields.pop(key, None)
        fields["display_rgb"] = {"array": "arrays/display0.npy"}
        for obj in fields["objects"]:
            obj["fields"].pop("touches_annotation", None)
    for key in ("qc_params", "calibration_confirmed"):
        batch["fields"].pop(key, None)
    return {"schema": 1, "batch": batch}


# --------------------------------------------------------------------------- #


def test_round_trip_keeps_results_and_review(batch, tmp_path):
    before = [compute_results(s).cells for s in batch.images]
    loaded = load_project(save_project(batch, tmp_path / "p.cellscope"))
    assert loaded.label == "Plate 4"
    for original, restored, cells in zip(batch.images, loaded.images, before):
        np.testing.assert_array_equal(original.object_labels, restored.object_labels)
        assert restored.qc.excluded_ids == original.qc.excluded_ids
        assert restored.segmented_with == original.segmented_with
        assert restored.display_rgb is not None, "display copies are rebuilt"
        assert not restored.object_labels.flags.writeable
        import pandas as pd

        pd.testing.assert_frame_equal(compute_results(restored).cells, cells)
    assert loaded.images[0].reviewed
    assert signature(loaded.images[0]) == loaded.images[0].reviewed_signature


def test_header_records_versions_and_skips_display_copies(batch, tmp_path):
    path = save_project(batch, tmp_path / "p.cellscope")
    manifest = _manifest(path)
    assert manifest["schema"] == 2
    assert manifest["app"]["cellscope"]
    assert "display_rgb" not in json.dumps(manifest)
    header = read_header(path)
    assert header["label"] == "Plate 4" and header["n_images"] == 2


def test_a_schema_1_project_opens_with_defaults(batch, tmp_path):
    path = tmp_path / "old.cellscope"
    save_project(batch, path)
    manifest = _as_schema_1(_manifest(path))
    _rewrite(path, manifest)
    buffer = io.BytesIO()
    np.save(buffer, np.zeros((10, 10, 3), np.uint8))
    with zipfile.ZipFile(path, "a") as archive:
        archive.writestr("arrays/display0.npy", buffer.getvalue())

    warnings: list[str] = []
    loaded = load_project(path, warnings)
    assert any("0.1" in w for w in warnings)
    session = loaded.images[0]
    assert session.segmented_with == ""
    assert session.annotation is None
    assert session.display_rgb.shape[:2] == (200, 200), "rebuilt, not the stored copy"
    assert session.reviewed, "an upgrade must not silently un-review an image"
    from src.review import validate_review

    validate_review(session)
    assert session.reviewed


def test_a_newer_schema_is_refused_with_a_clear_message(batch, tmp_path):
    path = tmp_path / "future.cellscope"
    save_project(batch, path)
    manifest = _manifest(path)
    manifest["schema"] = 7
    manifest["app"]["cellscope"] = "3.0"
    _rewrite(path, manifest)
    with pytest.raises(ProjectError, match="CellScope 3.0. Upgrade"):
        load_project(path)


def test_unknown_fields_are_dropped_with_a_warning(batch, tmp_path):
    path = tmp_path / "newer.cellscope"
    save_project(batch, path)
    manifest = _manifest(path)
    manifest["batch"]["fields"]["images"][0]["fields"]["future_setting"] = 3
    _rewrite(path, manifest)
    warnings: list[str] = []
    loaded = load_project(path, warnings)
    assert len(loaded.images) == 2
    assert any("future_setting" in w for w in warnings)


def test_unknown_record_types_are_refused(batch, tmp_path):
    path = tmp_path / "odd.cellscope"
    save_project(batch, path)
    manifest = _manifest(path)
    manifest["batch"]["fields"]["qc_params"] = {"type": "Mystery", "fields": {}}
    _rewrite(path, manifest)
    with pytest.raises(ProjectError, match="Mystery"):
        load_project(path)


@pytest.mark.parametrize("damage", ["truncate", "not_zip", "no_manifest", "missing_array",
                                    "bad_json", "bad_array"])
def test_damaged_files_raise_a_project_error(batch, tmp_path, damage):
    path = tmp_path / "damaged.cellscope"
    save_project(batch, path)
    data = path.read_bytes()
    if damage == "truncate":
        path.write_bytes(data[: len(data) // 2])
    elif damage == "not_zip":
        path.write_bytes(b"hello")
    elif damage == "no_manifest":
        _rewrite(path, {}, drop=("project.json",))
        with zipfile.ZipFile(path) as archive:
            names = archive.namelist()
        with zipfile.ZipFile(path, "w") as archive:
            for name in names:
                if name != "project.json":
                    archive.writestr(name, b"")
    elif damage == "missing_array":
        _rewrite(path, _manifest(path), drop=("arrays/0.npy",))
    elif damage == "bad_json":
        with zipfile.ZipFile(path) as archive:
            members = {n: archive.read(n) for n in archive.namelist()}
        members["project.json"] = b"{not json"
        with zipfile.ZipFile(path, "w") as archive:
            for name, blob in members.items():
                archive.writestr(name, blob)
    elif damage == "bad_array":
        with zipfile.ZipFile(path) as archive:
            members = {n: archive.read(n) for n in archive.namelist()}
        members["arrays/0.npy"] = b"\x93NUMPY garbage"
        with zipfile.ZipFile(path, "w") as archive:
            for name, blob in members.items():
                archive.writestr(name, blob)
    with pytest.raises(ProjectError):
        load_project(path)


def test_pickled_arrays_are_never_loaded(batch, tmp_path):
    path = tmp_path / "pickle.cellscope"
    save_project(batch, path)
    buffer = io.BytesIO()
    np.save(buffer, np.array([{"evil": True}], dtype=object), allow_pickle=True)
    with zipfile.ZipFile(path) as archive:
        members = {n: archive.read(n) for n in archive.namelist()}
    members["arrays/0.npy"] = buffer.getvalue()
    with zipfile.ZipFile(path, "w") as archive:
        for name, blob in members.items():
            archive.writestr(name, blob)
    with pytest.raises(ProjectError):
        load_project(path)


def test_a_locked_target_is_retried(batch, tmp_path, monkeypatch):
    calls = []
    real_replace = project.os.replace

    def flaky(source, target):
        calls.append(1)
        if len(calls) < 3:
            raise PermissionError("file in use by OneDrive")
        return real_replace(source, target)

    monkeypatch.setattr(project.os, "replace", flaky)
    path = save_project(batch, tmp_path / "p.cellscope")
    assert len(calls) == 3
    assert load_project(path).label == "Plate 4"


def test_a_failed_save_keeps_the_previous_file(batch, tmp_path, monkeypatch):
    path = tmp_path / "p.cellscope"
    save_project(batch, path)
    original = path.read_bytes()

    def explode(*args, **kwargs):
        raise OSError("disk full")

    monkeypatch.setattr(project.np, "save", explode)
    with pytest.raises(OSError):
        save_project(batch, path)
    assert path.read_bytes() == original
    assert not list(tmp_path.glob(project.TEMP_PREFIX + "*"))


def test_stale_temporaries_are_cleaned(tmp_path):
    import os

    old = tmp_path / (project.TEMP_PREFIX + "abc.tmp")
    new = tmp_path / (project.TEMP_PREFIX + "def.tmp")
    other = tmp_path / "keep.tmp"
    for f in (old, new, other):
        f.write_bytes(b"x")
    os.utime(old, (time.time() - 7200, time.time() - 7200))
    assert project.cleanup_temporaries(tmp_path) == 1
    assert not old.exists() and new.exists() and other.exists()


def test_uncompressed_arrays_round_trip(batch, tmp_path):
    path = save_project(batch, tmp_path / "fast.cellscope", compress_arrays=False)
    with zipfile.ZipFile(path) as archive:
        kinds = {i.filename: i.compress_type for i in archive.infolist()}
    assert kinds["arrays/0.npy"] == zipfile.ZIP_STORED
    assert kinds["project.json"] == zipfile.ZIP_DEFLATED
    assert len(load_project(path).images) == 2


# --------------------------------------------------------------------------- #
# Autosave
# --------------------------------------------------------------------------- #


def _wait(saver, batch):
    saver.request(batch, wait=True)


def test_autosave_writes_once_and_then_skips_until_something_changes(batch, tmp_path, monkeypatch):
    writes = []
    real = project.write_project

    def counting(*args, **kwargs):
        writes.append(1)
        return real(*args, **kwargs)

    import src.autosave as autosave

    monkeypatch.setattr(autosave, "write_project", counting)
    saver = Autosaver(tmp_path)
    _wait(saver, batch)
    assert len(writes) == 1
    assert recovery_path(tmp_path, batch).exists()
    assert saver.request(batch).startswith("Autosaved")
    _wait(saver, batch)
    assert len(writes) == 1, "nothing changed"
    assert not saver.is_dirty(batch)

    batch.images[1].well = "B7"
    assert saver.is_dirty(batch)
    assert "unsaved changes" in saver.status(batch)
    _wait(saver, batch)
    assert len(writes) == 2
    assert load_project(recovery_path(tmp_path, batch)).images[1].well == "B7"
    saver.shutdown()


def test_autosave_does_not_block_the_caller(batch, tmp_path, monkeypatch):
    import src.autosave as autosave

    release = threading.Event()
    real = project.write_project

    def slow(*args, **kwargs):
        release.wait(5)
        return real(*args, **kwargs)

    monkeypatch.setattr(autosave, "write_project", slow)
    saver = Autosaver(tmp_path)
    started = time.perf_counter()
    assert saver.request(batch) == "Autosaving…"
    assert time.perf_counter() - started < 1.0
    assert saver.request(batch) == "Autosaving…"   # no second concurrent write
    release.set()
    _wait(saver, batch)
    assert saver.status(batch).startswith("Autosaved")
    saver.shutdown()


def test_an_edit_during_a_save_is_saved_next_time(batch, tmp_path, monkeypatch):
    import src.autosave as autosave

    release = threading.Event()
    real = project.write_project

    def slow(*args, **kwargs):
        release.wait(5)
        return real(*args, **kwargs)

    monkeypatch.setattr(autosave, "write_project", slow)
    saver = Autosaver(tmp_path)
    saver.request(batch)
    batch.images[0].review_note = "edited while saving"     # after the snapshot
    release.set()
    saver._state(batch).pending.result()
    assert saver.is_dirty(batch), "the snapshot predates the edit"
    _wait(saver, batch)
    loaded = load_project(recovery_path(tmp_path, batch))
    assert loaded.images[0].review_note == "edited while saving"
    saver.shutdown()


def test_a_failed_autosave_is_reported_and_retried(batch, tmp_path, monkeypatch):
    import src.autosave as autosave

    failures = [OSError("disk full")]
    real = project.write_project

    def flaky(*args, **kwargs):
        if failures:
            raise failures.pop()
        return real(*args, **kwargs)

    monkeypatch.setattr(autosave, "write_project", flaky)
    saver = Autosaver(tmp_path)
    assert "failed" in saver.request(batch, wait=True)
    assert saver.is_dirty(batch)
    assert saver.request(batch, wait=True).startswith("Autosaved")
    saver.shutdown()


def test_recoveries_list_labels(batch, tmp_path):
    saver = Autosaver(tmp_path)
    _wait(saver, batch)
    found = saver.recoveries()
    assert found[0]["label"] == "Plate 4" and found[0]["n_images"] == 2
    saver.shutdown()


def test_a_snapshot_taken_mid_edit_is_consistent(batch, tmp_path):
    """Edits made under the batch lock never interleave with a snapshot."""
    saver = Autosaver(tmp_path)
    stop = threading.Event()
    session = batch.images[1]
    ids = [o.object_id for o in session.objects]

    def edit():
        i = 0
        while not stop.is_set():
            with batch.lock():
                session.qc.exclude(ids[i % len(ids)], "x")
                session.qc.restore(ids[i % len(ids)], "y")
            i += 1

    worker = threading.Thread(target=edit)
    worker.start()
    try:
        for _ in range(5):
            _wait(saver, batch)
    finally:
        stop.set()
        worker.join()
    loaded = load_project(recovery_path(tmp_path, batch))
    assert loaded.images[1].qc.excluded_ids == set()
    saver.shutdown()
