"""Versioned, portable project archives: JSON metadata and non-pickled NumPy arrays."""
from dataclasses import fields, is_dataclass
import io
import json
import os
from pathlib import Path
import tempfile
import zipfile

import numpy as np
from . import types

SCHEMA = 1
REGISTRY = {name: cls for name, cls in vars(types).items()
            if isinstance(cls, type) and is_dataclass(cls)}
SKIP = {"results_cache", "reference_detection", "undo_stack", "redo_stack"}


def save_project(batch, path):
    """Atomically save without modifying the last good project on failure."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(dir=path.parent, suffix=".tmp")
    os.close(fd)
    try:
        with zipfile.ZipFile(temporary, "w", zipfile.ZIP_DEFLATED) as archive:
            counter = 0
            def encode(value):
                nonlocal counter
                if isinstance(value, np.ndarray):
                    name = f"arrays/{counter}.npy"
                    counter += 1
                    buffer = io.BytesIO()
                    np.save(buffer, value, allow_pickle=False)
                    archive.writestr(name, buffer.getvalue())
                    return {"array": name}
                if is_dataclass(value):
                    return {"type": type(value).__name__, "fields": {
                        f.name: encode(getattr(value, f.name)) for f in fields(value)
                        if f.name not in SKIP}}
                if isinstance(value, (set, tuple)):
                    return {"collection": type(value).__name__, "items": [encode(v) for v in value]}
                if isinstance(value, list):
                    return [encode(v) for v in value]
                if isinstance(value, dict):
                    return {str(k): encode(v) for k, v in value.items()}
                if isinstance(value, np.generic):
                    return value.item()
                return value
            archive.writestr("project.json", json.dumps({"schema": SCHEMA, "batch": encode(batch)}))
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)
    return str(path.resolve())


def load_project(path):
    """Read only known dataclasses; never extract paths or execute pickle payloads."""
    with zipfile.ZipFile(path) as archive:
        if sum(i.file_size for i in archive.infolist()) > 8 * 1024**3:
            raise ValueError("Project exceeds the 8 GiB uncompressed limit.")
        manifest = json.loads(archive.read("project.json"))
        if manifest.get("schema") != SCHEMA:
            raise ValueError("Unsupported project version.")
        def decode(value):
            if isinstance(value, list):
                return [decode(v) for v in value]
            if not isinstance(value, dict):
                return value
            if set(value) == {"array"}:
                return np.load(io.BytesIO(archive.read(value["array"])), allow_pickle=False)
            if set(value) == {"type", "fields"}:
                cls = REGISTRY.get(value["type"])
                if cls is None:
                    raise ValueError("Unknown project record type.")
                allowed = {f.name for f in fields(cls)} - SKIP
                if not set(value["fields"]) <= allowed:
                    raise ValueError("Unknown project fields.")
                return cls(**{k: decode(v) for k, v in value["fields"].items()})
            if set(value) == {"collection", "items"}:
                constructors = {"set": set, "tuple": tuple}
                return constructors[value["collection"]](decode(value["items"]))
            return {k: decode(v) for k, v in value.items()}
        batch = decode(manifest["batch"])
    if not isinstance(batch, types.BatchSession):
        raise ValueError("Not a CellScope batch project.")
    for session in batch.images:
        for record in (session.image, session.nuclear_image, session.reference_image):
            if record:
                if record.pixels.shape[:2] != (record.height, record.width):
                    raise ValueError("Image dimensions do not match project metadata.")
                record.pixels.flags.writeable = False
        if session.raw_labels is not None:
            session.raw_labels.flags.writeable = False
        if session.object_labels is not None:
            if session.image is None or session.object_labels.shape != session.image.pixels.shape[:2]:
                raise ValueError("Mask dimensions do not match the image.")
            if session.object_labels.dtype.kind not in "iu" or session.object_labels.min() < 0:
                raise ValueError("Invalid label mask.")
            ids = {o.object_id for o in session.objects}
            if set(np.unique(session.object_labels)) - {0} != ids:
                raise ValueError("Mask IDs do not match object records.")
    return batch


def preset_dict(batch):
    return {"schema": 1, "channel": batch.channel, "nuclear_channel": batch.nuclear_channel,
            "parameters": {name: {f.name: getattr(getattr(batch, name), f.name)
                                   for f in fields(getattr(batch, name))}
                           for name in batch.SHARED_PARAMS if name != "calibration"}}


def apply_preset(batch, data):
    if data.get("schema") != 1:
        raise ValueError("Unsupported preset version.")
    allowed_channels = {"grayscale", "red", "green", "blue"}
    if data.get("channel") not in allowed_channels or data.get("nuclear_channel") not in allowed_channels | {"none"}:
        raise ValueError("Preset contains an invalid channel.")
    allowed_params = set(batch.SHARED_PARAMS) - {"calibration"}
    if not isinstance(data.get("parameters"), dict) or set(data["parameters"]) != allowed_params:
        raise ValueError("Preset must contain all four analysis parameter groups.")
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
