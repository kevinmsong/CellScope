# Development

## Setup and checks

```powershell
python -m pip install -e ".[cellpose,dev]"
python -m pytest -q                 # about 570 tests, ~2 minutes
python -m ruff check .
python tools/bench.py               # quick performance suite; see PERFORMANCE.md
```

**Test data.** Tests build synthetic masks and fields in memory. The tests that
need Cellpose, Gradio or local images skip when those are missing. CI
(`.github/workflows/tests.yml`) runs lint, the tests and the launcher on a
CPU-only Linux runner without Cellpose.

**What the suite does not cover.** It does not establish segmentation accuracy
on biological data. Validate settings against manually reviewed images before
relying on automated counts.

**Real-app check.** Before a release, drive the app itself: launch
`python -m cellscope`, run a Cellpose batch, review with zoom and shortcuts,
then export and save. The browser-side viewer is not covered by the Python
tests.

## Architecture

```
cellscope/            the application (imports Gradio)
  cli.py              `cellscope` / `python -m cellscope`
  ui/shell.py         header, workflow stepper, tabs, cross-tab wiring
  ui/*_tab.py         one module per tab: layout, handlers, wiring
  ui/views.py         shared views: status, stepper, navigator, image table
  ui/viewer.py        browser-side review viewer (HTML layers + JS)
  ui/theme.py         tokens, stylesheet, theme, HTML helpers
src/                  the science (never imports Gradio)
  types.py            session, parameter and record dataclasses
  pipeline.py         prepare → infer → finalize → commit; batch pipeline; cached results
  segmentation.py     Cellpose adapter (model cache, inference lock, OOM recovery), splitting
  morphometry.py      cell and cluster measurement (window-local, cached geometry)
  clustering.py       contact graph and clusters
  qc.py               exclusion rules and log
  scalebar.py         reference-bar detection and burned-in annotation detection
  review.py           review signature, undo/redo, corrections
  workflow.py         stepper state, staleness, change token
  project.py          .cellscope schema 2, migration, atomic writes
  autosave.py         dirty-tracked background autosave
  export.py           archives and manifest
  quality.py          readiness and replicate-level summaries
  report.py           PDF
  nuclei.py           DAPI segmentation and contact counts
  perf.py             benchmark workloads and measurement
tools/bench.py        benchmark CLI
app.py, professional_ui.py, nuclei_ui.py, ui_theme.py   compatibility shims
```

**Dependency direction.**

- `shell` imports the tabs; the tabs import `views`, `theme` and `src`.
- Nothing imports `shell` or the shims.
- `tests/test_ui.py` checks these boundaries, and also that `src` never
  imports Gradio.

## Caching

**The rule.** Every cache is a `(key, value)` pair, read only when its key
matches exactly. Keys are built from monotonic counters
(`segmentation_version`, `qc.version`), frozen parameter records, and the
identity of read-only label arrays. A cache therefore cannot return a stale
value; there is nothing to invalidate by hand.

| Layer | Key | Holds |
|---|---|---|
| `compute_results` | segmentation version, QC version, calibration, cluster parameters | QC labels, clusters, tables |
| Geometry | mask identity, segmentation version, calibration, contact distance | per-cell and per-cluster geometry, reused across QC edits |
| Viewer | results key, display size, view options, selection | encoded image layers, hover data |
| Batch (`memoised`) | every image's results key, well, review state | pooled tables, figures |

**Label arrays are read-only.** An edit builds a new array. That is what makes
undo snapshots and autosave snapshots cheap: they hold references, not copies.

## Concurrency

**GPU.** Inference is serialised by `_INFERENCE_LOCK`.

**Batch runs.** CPU threads prepare the next image and finalize the previous one
around that single GPU stage. Results are committed in input order under the
batch lock, so the output does not depend on timing.

**Everything else.** Interface handlers that change state take the same lock
(`views.locked`). Autosave takes it only to snapshot, then writes outside it.

## Scientific invariants

1. Source pixels and raw segmentation are never modified. QC and corrections
   work on copies.
2. Measurements use the current QC-approved mask. Excluded cells never hold a
   cluster together.
3. An unresolved group has no per-cell measurements, and its cell count is NA.
4. Pixel and physical units are never pooled.
5. Replicate summaries go fields → wells → biological replicates → conditions.
   Cells are never replicates, and no tests are run.
6. Changing a setting never re-segments. Stale images are flagged instead.
7. Optimised implementations must match the originals: `tests/reference_impl.py`
   keeps frozen copies, and `tests/test_golden_equivalence.py` compares them.
8. Automatic exclusions are logged with their reason and never override a
   manual restore.
9. Projects never unpickle; a failed save never replaces the previous file.

## Known limits

- **Undo depth.** Undo keeps 12 steps per image, and it is not saved with the
  project.
- **Cancellation.** A single Cellpose inference cannot be interrupted; Cancel
  stops after it.
- **Tile batch size.** On a GPU, the tile batch is chosen from free memory, and
  batched bfloat16 inference can round slightly differently from single tiles.
  The size used is recorded per image.
- **Boundary correction.** Corrections use polygons; there is no freehand brush.
- **Scale-bar detection.** It recognises saturated yellow or white bars near the
  top or bottom edge. Other annotation styles need a manual exclusion.
- **DAPI segmentation.** It is a threshold-and-watershed estimate.

Never stage image data, `projects/`, `output/`, model weights or credentials.
Review `git diff --cached --stat` before committing.
