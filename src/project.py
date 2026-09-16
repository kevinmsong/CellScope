"""Versioned, portable project archives: JSON metadata and non-pickled NumPy arrays.

A ``.cellscope`` file is a ZIP holding ``project.json`` and one ``.npy`` file per
array. Nothing is ever unpickled: records are rebuilt only from the dataclasses
in :mod:`src.types`, and arrays load with ``allow_pickle=False``.

Schema history
--------------
1. The first release.
2. Adds a header (``app`` versions, ``saved_at``) and stops storing the display
   copy of each image, which is rebuilt on load. New fields -- QC rules,
   burned-in annotations, the segmentation fingerprint -- are optional, so a
   schema 1 project opens with their defaults.

Compatibility rules
-------------------
* An older schema is migrated in memory; the file on disk is not touched.
* A newer schema is refused with a message naming the version that wrote it.
* Fields this version does not know, on record types it does know, are dropped
  with a warning, so a project from a slightly newer release still opens.
  An unknown record *type* is refused: guessing its meaning is not safe.
* A damaged file raises :class:`ProjectError` naming what was wrong, and a
  failed save never replaces the previous good file.
"""

from __future__ import annotations

import io
import json
import os
import platform
import tempfile
import time
import zipfile
import zlib
from dataclasses import fields, is_dataclass
from pathlib import Path
from typing import Any

import numpy as np

from . import __version__, types

SCHEMA = 2
REGISTRY = {name: cls for name, cls in vars(types).items()
            if isinstance(cls, type) and is_dataclass(cls)}
#: Fields that are derived, transient, or rebuilt on load. Never written, and
#: silently ignored if an older file contains them.
SKIP = {"results_cache", "geometry_cache", "view_cache", "derived_cache", "exported_token",
        "reference_detection", "undo_stack", "redo_stack", "display_rgb"}
#: Uncompressed size above which a project is refused, as a zip-bomb guard.
MAX_UNCOMPRESSED = 8 * 1024**3
#: Temporary files written next to a project start with this.
TEMP_PREFIX = ".cellscope-"


class ProjectError(ValueError):
    """A project file cannot be read: damaged, incomplete, or from a newer release."""


# --------------------------------------------------------------------------- #
# Writing
# --------------------------------------------------------------------------- #


def encode_project(batch) -> tuple[dict[str, Any], list[tuple[str, np.ndarray]]]:
    """Turn a batch into a JSON-ready manifest plus the arrays it references.

    No I/O happens here, so a caller can take a consistent snapshot while
    holding the batch lock and write it afterwards. Read-only arrays are
    referenced, not copied: CellScope never modifies them in place. A writable
    array is copied, since someone could still change it.
    """
    arrays: list[tuple[str, np.ndarray]] = []

    def encode(value):
        if isinstance(value, np.ndarray):
            name = f"arrays/{len(arrays)}.npy"
            arrays.append((name, value if not value.flags.writeable else value.copy()))
            return {"array": name}
        if is_dataclass(value) and not isinstance(value, type):
            return {"type": type(value).__name__, "fields": {
                f.name: encode(getattr(value, f.name)) for f in fields(value)
                if f.name not in SKIP}}
        if isinstance(value, (set, frozenset)):
            return {"collection": "set", "items": [encode(v) for v in sorted(value, key=repr)]}
        if isinstance(value, tuple):
            return {"collection": "tuple", "items": [encode(v) for v in value]}
        if isinstance(value, list):
            return [encode(v) for v in value]
        if isinstance(value, dict):
            return {str(k): encode(v) for k, v in value.items()}
        if isinstance(value, np.generic):
            return value.item()
        return value

    manifest = {
        "schema": SCHEMA,
        "app": {
            "cellscope": __version__,
            "python": platform.python_version(),
            "numpy": np.__version__,
        },
        "saved_at": types.utc_now(),
        "batch": encode(batch),
    }
    return manifest, arrays


def _replace_with_retry(source: str, target: Path, attempts: int = 6) -> None:
    """``os.replace``, retried briefly when another process holds the target.

    Sync clients (OneDrive) and virus scanners open files for a moment after
    they change; on Windows that makes a replace fail with a sharing error.
    """
    delay = 0.05
    for attempt in range(attempts):
        try:
            os.replace(source, target)
            return
        except PermissionError:
            if attempt == attempts - 1:
                raise
            time.sleep(delay)
            delay *= 2


def write_project(manifest, arrays, path, *, compression_level: int = 6,
                  compress_arrays: bool = True) -> str:
    """Write an encoded project atomically.

    The archive is built in a temporary file beside the target, flushed to disk,
    and moved into place; a failure at any point leaves the previous file as it
    was. ``compress_arrays=False`` stores arrays uncompressed, which makes
    autosaves several times faster at the cost of a larger file.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(dir=path.parent, prefix=TEMP_PREFIX, suffix=".tmp")
    os.close(fd)
    try:
        with zipfile.ZipFile(temporary, "w", zipfile.ZIP_DEFLATED,
                             compresslevel=compression_level) as archive:
            array_compression = zipfile.ZIP_DEFLATED if compress_arrays else zipfile.ZIP_STORED
            for name, array in arrays:
                buffer = io.BytesIO()
                np.save(buffer, array, allow_pickle=False)
                archive.writestr(name, buffer.getvalue(), compress_type=array_compression)
            archive.writestr("project.json", json.dumps(manifest))
        with open(temporary, "rb+") as handle:
            os.fsync(handle.fileno())
        _replace_with_retry(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)
    return str(path.resolve())


def save_project(batch, path, *, compression_level: int = 6, compress_arrays: bool = True) -> str:
    """Atomically save without modifying the last good project on failure."""
    lock = batch.lock() if hasattr(batch, "lock") else None
    if lock is not None:
        with lock:
            manifest, arrays = encode_project(batch)
    else:
        manifest, arrays = encode_project(batch)
    return write_project(manifest, arrays, path, compression_level=compression_level,
                         compress_arrays=compress_arrays)


def cleanup_temporaries(directory, older_than_seconds: float = 3600) -> int:
    """Remove temporary files left behind by an interrupted save."""
    directory = Path(directory)
    if not directory.is_dir():
        return 0
    removed = 0
    cutoff = time.time() - older_than_seconds
    for candidate in directory.glob(TEMP_PREFIX + "*.tmp"):
        try:
            if candidate.stat().st_mtime < cutoff:
                candidate.unlink()
                removed += 1
        except OSError:
            continue
    return removed


# --------------------------------------------------------------------------- #
# Reading
# --------------------------------------------------------------------------- #


def _migrate(manifest: dict[str, Any], warnings: list[str]) -> dict[str, Any]:
    """Bring an older manifest up to the current schema, in memory."""
    if manifest.get("schema") == 1:
        # Schema 1 stored display copies, which SKIP now drops on decode; every
        # other field added since has a default.
        manifest = {**manifest, "schema": 2, "app": {"cellscope": "0.1"}}
        warnings.append(
            "Opened a project saved by CellScope 0.1. It will be saved in the current format."
        )
    return manifest


def read_header(path) -> dict[str, Any]:
    """Schema, versions, label and image count, without loading any arrays."""
    try:
        with zipfile.ZipFile(path) as archive:
            manifest = json.loads(archive.read("project.json"))
    except (zipfile.BadZipFile, KeyError, json.JSONDecodeError, OSError, zlib.error) as error:
        raise ProjectError("{} is not a readable CellScope project: {}".format(
            Path(path).name, error)) from error
    batch = (manifest.get("batch") or {}).get("fields", {})
    images = batch.get("images", [])
    return {
        "schema": manifest.get("schema"),
        "app": manifest.get("app", {}),
        "saved_at": manifest.get("saved_at"),
        "label": batch.get("label", ""),
        "n_images": len(images) if isinstance(images, list) else 0,
    }


def load_project(path, warnings: list[str] | None = None):
    """Read only known dataclasses; never extract paths or execute pickle payloads.

    Problems that do not prevent opening are appended to ``warnings``.
    """
    warnings = warnings if warnings is not None else []
    name = Path(path).name
    try:
        archive = zipfile.ZipFile(path)
    except (zipfile.BadZipFile, OSError) as error:
        raise ProjectError("{} is damaged or is not a CellScope project ({}).".format(
            name, error)) from error

    with archive:
        if sum(i.file_size for i in archive.infolist()) > MAX_UNCOMPRESSED:
            raise ProjectError("Project exceeds the 8 GiB uncompressed limit.")
        try:
            manifest = json.loads(archive.read("project.json"))
        except KeyError as error:
            raise ProjectError("{} has no project.json; it is incomplete.".format(name)) from error
        except (json.JSONDecodeError, zipfile.BadZipFile, zlib.error, EOFError) as error:
            raise ProjectError("{}: project.json is damaged ({}).".format(name, error)) from error
        if not isinstance(manifest, dict):
            raise ProjectError("{}: project.json is not a CellScope manifest.".format(name))

        schema = manifest.get("schema")
        if isinstance(schema, int) and schema > SCHEMA:
            written_by = (manifest.get("app") or {}).get("cellscope", "a newer release")
            raise ProjectError(
                "Unsupported project version {}: {} was saved by CellScope {}. "
                "Upgrade CellScope to open it.".format(schema, name, written_by)
            )
        if schema not in (1, 2):
            raise ProjectError("Unsupported project version {!r} in {}.".format(schema, name))
        manifest = _migrate(manifest, warnings)
        dropped: dict[str, set[str]] = {}

        def load_array(member):
            try:
                return np.load(io.BytesIO(archive.read(member)), allow_pickle=False)
            except KeyError as error:
                raise ProjectError("{} is incomplete: {} is missing.".format(name, member)) from error
            except (ValueError, EOFError, zipfile.BadZipFile, zlib.error) as error:
                raise ProjectError("{}: {} is damaged ({}).".format(name, member, error)) from error

        def decode(value):
            if isinstance(value, list):
                return [decode(v) for v in value]
            if not isinstance(value, dict):
                return value
            if set(value) == {"array"}:
                return load_array(value["array"])
            if set(value) == {"type", "fields"}:
                cls = REGISTRY.get(value["type"])
                if cls is None:
                    raise ProjectError("Unknown project record type {!r}; it was probably "
                                       "written by a newer CellScope.".format(value["type"]))
                known = {f.name for f in fields(cls)}
                kept = {}
                for key, item in value["fields"].items():
                    if key in SKIP:
                        continue
                    if key not in known:
                        dropped.setdefault(cls.__name__, set()).add(key)
                        continue
                    kept[key] = decode(item)
                try:
                    return cls(**kept)
                except TypeError as error:
                    raise ProjectError("{}: a {} record is malformed ({}).".format(
                        name, cls.__name__, error)) from error
            if set(value) == {"collection", "items"}:
                constructors = {"set": set, "tuple": tuple}
                if value["collection"] not in constructors:
                    raise ProjectError("Unknown collection {!r} in {}.".format(
                        value["collection"], name))
                return constructors[value["collection"]](decode(value["items"]))
            return {k: decode(v) for k, v in value.items()}

        batch = decode(manifest.get("batch"))

    for type_name, keys in sorted(dropped.items()):
        warnings.append("Ignored {} field(s) this version does not know: {}.".format(
            type_name, ", ".join(sorted(keys))))
    if not isinstance(batch, types.BatchSession):
        raise ProjectError("{} does not contain a CellScope batch.".format(name))
    _validate(batch)
    return batch


def _validate(batch) -> None:
    """Structural checks that a hand-edited or damaged project cannot pass."""
    from .image_io import make_display

    for session in batch.images:
        for record in (session.image, session.nuclear_image, session.reference_image):
            if record:
                if record.pixels.shape[:2] != (record.height, record.width):
                    raise ProjectError("Image dimensions do not match project metadata.")
                record.pixels.flags.writeable = False
        if session.raw_labels is not None:
            session.raw_labels.flags.writeable = False
        if session.object_labels is not None:
            if session.image is None or session.object_labels.shape != session.image.pixels.shape[:2]:
                raise ProjectError("Mask dimensions do not match the image.")
            if session.object_labels.dtype.kind not in "iu" or session.object_labels.min() < 0:
                raise ProjectError("Invalid label mask.")
            ids = {o.object_id for o in session.objects}
            if set(np.unique(session.object_labels)) - {0} != ids:
                raise ProjectError("Mask IDs do not match object records.")
            session.object_labels.flags.writeable = False
        if session.image is not None:
            session.display_rgb, session.display_scale = make_display(session.image)


# --------------------------------------------------------------------------- #
# Presets
# --------------------------------------------------------------------------- #


PRESET_SCHEMA = 2
#: Parameter groups a preset of each schema must contain. Schema 1 predates
#: the QC rules, which are then left at their defaults.
PRESET_GROUPS = {
    1: {"preprocess_params", "segmentation_params", "split_params", "cluster_params"},
    2: {"preprocess_params", "segmentation_params", "split_params", "cluster_params",
        "qc_params"},
}


def preset_dict(batch):
    return {"schema": PRESET_SCHEMA, "channel": batch.channel,
            "nuclear_channel": batch.nuclear_channel,
            "parameters": {name: {f.name: getattr(getattr(batch, name), f.name)
                                   for f in fields(getattr(batch, name))}
                           for name in batch.SHARED_PARAMS if name != "calibration"}}


def apply_preset(batch, data):
    schema = data.get("schema")
    if schema not in PRESET_GROUPS:
        raise ValueError("Unsupported preset version.")
    allowed_channels = {"grayscale", "red", "green", "blue"}
    if data.get("channel") not in allowed_channels or data.get("nuclear_channel") not in allowed_channels | {"none"}:
        raise ValueError("Preset contains an invalid channel.")
    expected = PRESET_GROUPS[schema]
    if not isinstance(data.get("parameters"), dict) or set(data["parameters"]) != expected:
        raise ValueError("Preset must contain exactly these parameter groups: {}.".format(
            ", ".join(sorted(expected))))
    # Validate all constructor fields before changing the batch.
    values = {name: type(getattr(batch, name))(**params)
              for name, params in data["parameters"].items()
              if name in batch.SHARED_PARAMS and name != "calibration"}
    for name, value in values.items():
        setattr(batch, name, value)
    batch.channel = data["channel"]
    batch.nuclear_channel = data["nuclear_channel"]
    batch.apply_shared(*values)
    for session in batch.images:
        batch.apply_channels(session)
        session.reviewed = False
    return batch
