# Development

## Validation

```powershell
.venv\Scripts\python.exe -m pytest -q
```

Tests construct synthetic masks and signals in memory. They cover morphology,
calibration, clustering, segmentation, batch aggregation, exports, Gradio callback
contracts, project round-trips, manual corrections, undo/redo, and nuclear contact.
Gradio-specific tests are skipped when Gradio is missing; use the project virtual
environment for complete coverage. The suite does not establish segmentation
accuracy on a biological dataset; validate settings against manually reviewed
images before relying on automated counts.

A CPU-only CI workflow installs the scientific/UI dependencies without Cellpose.
Cellpose/CUDA behavior remains dependent on the local installation and hardware.

## Architecture

| Module | Responsibility |
|---|---|
| `app.py` | Main Gradio workflow and cell-review callbacks |
| `ui_theme.py` | Light/dark theme, semantic colors and appearance persistence |
| `professional_ui.py` | Projects, presets, design, reports and review controls |
| `nuclei_ui.py` | Independent DAPI analysis and nuclear QC interface |
| `src/types.py` | Session and parameter records |
| `src/pipeline.py` | Cell segmentation and QC-approved measurement pipeline |
| `src/review.py` | Review signatures, history, view mapping and corrections |
| `src/project.py` | Versioned portable projects and presets |
| `src/quality.py` | Readiness and experimental summaries |
| `src/nuclei.py` | Nuclear segmentation and direct-contact counts |
| `src/report.py` | PDF generation |
| `src/export.py` | CSV/mask/metadata archive generation |

## Important invariants

1. Source pixels and raw cell segmentation are never edited by QC/corrections.
2. Scientific results use the current QC-approved object mask.
3. Excluded cells do not hold accepted cell clusters together.
4. Nuclear segmentation and QC are independent of cytoplasmic cell analysis.
5. Direct nuclear contact includes diagonal adjacency, but no background gap.
6. Biological replicate summaries never treat individual cells as replicates.
7. No pickle is used to load projects; archive paths are never extracted.
8. Every correction invalidates review and derived results.

## Current limits

Undo holds up to 12 full mask snapshots, so memory use grows with image size.
View zoom enlarges the browser preview; measurements and corrections remain in
original coordinates. Manual boundary correction uses polygons rather than a
freehand brush. Experimental comparison tables currently summarize area.
Cancellation is cooperative between images; inference is not forcibly terminated.
Autosave requires a connected browser, except for explicit per-image batch saves.
Nuclear segmentation is an Otsu/watershed estimate with manually tunable parameters.

Never stage image data, `projects/`, `output/`, model weights or credentials.
Review `git diff --cached --stat` and the staged filenames before publishing.
