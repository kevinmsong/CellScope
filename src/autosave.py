"""Background autosave that only writes when something changed.

The interface asks :class:`Autosaver` to save on a timer and after long runs.
Each request is cheap: it compares the batch's change token with the one last
written and returns at once when nothing changed. When something did change,
a snapshot is taken under the batch lock -- arrays are read-only and shared,
so this is fast -- and a single background thread writes it. The interface
never waits for the disk, and two saves never run at once.

Writes are atomic (see :func:`src.project.write_project`), so an interrupted
autosave leaves the previous recovery file intact. Arrays are compressed at
level 1: on a 39-image project that is 1.8 s in the background for a 33 MB
file, against 0.5 s for 263 MB uncompressed -- too large for a synced folder.
"""

from __future__ import annotations

import hashlib
import threading
import time
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path

from .project import cleanup_temporaries, encode_project, read_header, write_project
from .workflow import change_token


def recovery_path(directory, batch) -> Path:
    """The recovery file for a batch. Derived from its ID, never from user text."""
    key = hashlib.sha256(batch.batch_id.encode()).hexdigest()[:24]
    return Path(directory) / (key + ".cellscope")


@dataclass
class _State:
    token: object = None
    saved_at: float | None = None
    path: str = ""
    error: str = ""
    pending: Future | None = None
    pending_token: object = None
    seconds: float = 0.0


class Autosaver:
    """Dirty-tracked, non-blocking project autosave."""

    def __init__(self, directory, compress_arrays: bool = True):
        self.directory = Path(directory)
        self.compress_arrays = compress_arrays
        self._states: dict[str, _State] = {}
        self._guard = threading.Lock()
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="cellscope-autosave")
        cleanup_temporaries(self.directory)

    def _state(self, batch) -> _State:
        with self._guard:
            return self._states.setdefault(batch.batch_id, _State())

    def _write(self, manifest, arrays, path, state: _State, token) -> None:
        started = time.perf_counter()
        try:
            write_project(manifest, arrays, path, compression_level=1,
                          compress_arrays=self.compress_arrays)
        except Exception as error:
            state.error = "{}: {}".format(type(error).__name__, error)
            raise
        state.token = token
        state.saved_at = time.time()
        state.path = str(path)
        state.error = ""
        state.seconds = time.perf_counter() - started

    def request(self, batch, wait: bool = False) -> str:
        """Save the batch in the background if it changed since the last save.

        Returns a short status line for the interface. ``wait=True`` blocks
        until the write finishes, for use before the batch is replaced.
        """
        if batch is None or batch.is_empty:
            return "Autosave: nothing to save yet."
        state = self._state(batch)
        if state.pending is not None and not state.pending.done():
            if wait:
                state.pending.result()
            else:
                return "Autosaving…"
        with batch.lock():
            token = change_token(batch)
            if token == state.token:
                return self.status(batch)
            manifest, arrays = encode_project(batch)
        path = recovery_path(self.directory, batch)
        state.pending_token = token
        state.pending = self._executor.submit(self._write, manifest, arrays, path, state, token)
        if wait:
            try:
                state.pending.result()
            except Exception:
                return self.status(batch)
        return self.status(batch) if wait else "Autosaving…"

    def status(self, batch) -> str:
        if batch is None or batch.is_empty:
            return "Autosave: nothing to save yet."
        state = self._state(batch)
        if state.pending is not None and not state.pending.done():
            return "Autosaving…"
        if state.error:
            return "Autosave failed: {}. Save the project manually.".format(state.error)
        if state.saved_at is None:
            return "Not autosaved yet."
        dirty = change_token(batch) != state.token
        return "{} {}{}".format(
            "Autosaved" if not dirty else "Last autosave",
            time.strftime("%H:%M:%S", time.localtime(state.saved_at)),
            " · unsaved changes" if dirty else "",
        )

    def is_dirty(self, batch) -> bool:
        return change_token(batch) != self._state(batch).token

    def mark_clean(self, batch, path: str = "") -> None:
        """Record that ``batch`` as it stands has just been saved elsewhere."""
        state = self._state(batch)
        state.token = change_token(batch)
        state.saved_at = time.time()
        state.path = path or state.path

    def recoveries(self) -> list[dict]:
        """Recovery files, newest first, with the label and size each holds."""
        found = []
        for path in sorted(self.directory.glob("*.cellscope"),
                           key=lambda p: p.stat().st_mtime, reverse=True):
            try:
                header = read_header(path)
            except Exception:
                header = {"label": "(unreadable)", "n_images": 0, "saved_at": None}
            found.append({
                "name": path.name,
                "label": header.get("label") or "(unnamed)",
                "n_images": header.get("n_images", 0),
                "saved": time.strftime("%Y-%m-%d %H:%M", time.localtime(path.stat().st_mtime)),
            })
        return found

    def shutdown(self) -> None:
        self._executor.shutdown(wait=True)
