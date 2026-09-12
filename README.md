# CellScope

CellScope is a local application for reviewing fluorescence microscopy segmentation
and producing traceable cell measurements. It supports batch processing, manual
quality control, independent DAPI nuclear counts, and reproducible exports.

## Run locally

Use Python 3.10 or newer; Python 3.12 is tested. From this directory:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python app.py
```

Open the local URL printed in the terminal (normally `http://127.0.0.1:7860`).
Cellpose downloads model weights on its first use. A CUDA-compatible GPU is
optional. The explicitly named `threshold_watershed` engine works without
Cellpose and is useful for testing, but may perform poorly on crowded fields.

CellScope is intended to run on a trusted local computer. Project recovery files
contain image data and are available to users of that local server. Do not expose
the app publicly without adding authentication and storage isolation.

## A typical analysis

1. **Batch:** upload images, name the experiment, choose the cell channel, and
   calibrate with a known scale bar or pixel resolution. The optional nuclear
   guide helps cell segmentation; it is separate from DAPI counting.
2. **Segment:** inspect settings and run one image or the whole batch. Edge-touching
   cells are automatically excluded. Original segmentation masks remain intact.
3. **Review:** click cells to exclude or restore them. Gray outlines indicate
   excluded cells. Inspect linked measurements, zoom/pan, adjust opacity, undo/redo,
   and correct boundaries. Mark each image reviewed when satisfied.
4. **DAPI nuclei:** independently segment the blue channel (or a separate nuclear
   file), inspect nuclear masks, and review the count of directly touching nuclei.
5. **Quality & design:** check readiness and assign wells, conditions, and biological
   replicate labels. Compare well and replicate summaries without treating cells
   as independent biological replicates.
6. **Export:** download an analysis archive containing CSV measurements, masks,
   QC history, metadata, and a PDF report. A standalone PDF is also available on
   Quality & design.
7. **Projects & presets:** save a portable `.cellscope` project to resume later.
   Autosaves run every 60 seconds and after completed batch images. Recover them
   from the same tab after restarting the app.

## What is included

| Feature | Where to find it |
|---|---|
| Persistent System, Light, and Dark appearance | Header |
| Save/open projects and recover local autosaves | Projects & presets |
| Zoom, pan, original view, overlay opacity | Review > View and correction tools |
| Undo/redo; automatic invalidation of stale review | Review |
| Split, merge, and polygon boundary correction | Review > Correct segmentation |
| Click a cell or measurement row to inspect it | Review > Linked cell measurements |
| Reusable, named JSON analysis presets | Projects & presets |
| Cancellation, resume, and retry failed images | Processing |
| Readiness checks, well/replicate summaries, PDF | Quality & design |
| Separate direct-contact nuclear counts | DAPI nuclei |
| Audit-ready masks, tables, settings and reports | Export |

## How to interpret the results

- A resolved cell contributes individual morphology measurements. An unresolved
  group does not receive an invented cell count or per-cell measurements.
- Cell clusters are recomputed after exclusions; excluding a bridge cell can
  separate a cluster.
- DAPI counts are **nuclear segmentation estimates**, not cytoplasmic cell counts.
  Direct contact means nuclear masks share an edge or corner, with no background
  gap. Reports distinguish touching nuclei, touching groups, and contact pairs.
- Measurements use physical units only after calibration. Pixel and physical-unit
  results are never pooled together in the main batch summary.
- Experimental summaries pool fields within a well, average wells within each
  biological replicate, then compare replicate means. No cell-level p-values are
  calculated. Missing design labels are omitted rather than guessed.

See the [user guide](docs/USER_GUIDE.md) for correction and recovery workflows,
[formats](docs/FORMATS.md) for project/export contents, and
[development guide](docs/DEVELOPMENT.md) for validation and architecture.

## Data stays out of Git

This repository contains source code, documentation, and synthetic tests only.
Sample microscopy images, project archives, autosaves, model weights, generated
PDFs, screenshots, and analysis outputs are ignored. Tests generate their own
small synthetic arrays in memory; no sample image files are committed.
