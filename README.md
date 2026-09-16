# CellScope

CellScope is a local application for segmenting fluorescence microscopy images,
reviewing the result cell by cell, and producing traceable morphology
measurements. It keeps single cells, unresolved groups and excluded objects
distinct, never invents a cell count it cannot see, and exports everything
needed to show how each number was produced.

## Install (Windows)

You need Python 3.10–3.12 ([python.org](https://www.python.org/downloads/)).
Tick **Add python.exe to PATH** in the installer. Then, in PowerShell, from this
folder:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
```

**With an NVIDIA GPU** (recommended for Cellpose). Install the CUDA build of
PyTorch first, then CellScope with Cellpose:

```powershell
python -m pip install torch --index-url https://download.pytorch.org/whl/cu124
python -m pip install -e ".[cellpose]"
```

**CPU only.** Skip the first line. Cellpose runs on the CPU, several times more
slowly:

```powershell
python -m pip install -e ".[cellpose]"
```

**Without Cellpose.** `python -m pip install -e .` installs CellScope with only
its threshold-and-watershed engine. That engine is useful for testing, but
performs poorly on crowded fields.

`requirements.txt` lists the same runtime dependencies, including Cellpose, for
tools that expect one.

Tested with Python 3.12, Cellpose 4.2 (`cpsam`), PyTorch 2.5 (CUDA 12.4) and
Gradio 5.50. Cellpose 3.x is not supported. Cellpose downloads its model weights
(about 1.2 GB) the first time it runs.

## Run

```powershell
cellscope                    # or: python -m cellscope, or python app.py
```

Open the address it prints, normally <http://127.0.0.1:7860>.

| Option | Effect |
|---|---|
| `--port 7861` | Use a different port |
| `--cpu` | Never use the GPU |
| `--projects D:\cellscope-projects` | Where autosaves are kept (default `projects\`) |
| `--autosave-interval 30` | Seconds between checks for unsaved changes |
| `--no-preload` | Do not load the Cellpose model at start-up |
| `--inbrowser` | Open a browser tab |
| `--version` | Print CellScope, Cellpose, PyTorch and GPU versions |

CellScope is meant for a trusted local computer. It has no authentication, and
autosaves contain image data. Do not expose it on a network, or use `--share`,
unless you understand that anyone who can reach it can see and change the data.

## A typical analysis

The bar across the top shows where you are: **Import → Calibrate → Segment →
Review → Design → Export**. A single *Next* line always says what to do next.

1. **Import & calibrate.**
   - Name the experiment, add images, and choose the cell channel and an optional
     nuclear guide channel.
   - Calibrate using one of these:
     - a scale bar detected on the image;
     - two clicks on the bar;
     - a separate ruler frame;
     - a µm-per-pixel value.

     Or confirm that you want pixel units.
2. **Segment.**
   - Use **Preview on this image** to try settings on one representative field.
     It shows before-and-after counts. Keep or discard the result.
   - Then **Run batch**. Changing a setting never re-runs anything by itself.
     Images segmented with different settings are marked *stale*, and a batch
     run re-segments only new, failed and stale images.
3. **Review.**
   - Click cells to exclude or restore them, or inspect them.
   - Zoom with the mouse wheel and drag to pan.
   - Correct boundaries by splitting, merging or drawing a polygon, with undo and
     redo, then mark each image reviewed. Press <kbd>?</kbd> for the keyboard
     shortcuts.
   - Cells touching the image edge, or a scale bar burned into the image, are
     excluded automatically. They are logged and can be restored.
4. **Design & compare.** Give each image a well, condition and biological
   replicate. Summaries follow fields → wells → biological replicates →
   conditions, and a dot plot shows one point per replicate.
5. **Export.** A single archive holds measurements, masks, overlays, QC history,
   a PDF report and `manifest.json`. The manifest records the software versions,
   device, parameters, calibration, QC decisions, corrections and checksums.
6. **Projects.** Save a portable `.cellscope` project. Changes are also autosaved
   in the background and can be recovered after a restart.

**DAPI nuclei** counts nuclei independently of the cell segmentation.

## How to interpret the results

- **Unresolved groups.** A resolved cell has its own morphology measurements. A
  group that could not be separated gets group-level measurements only, and its
  cell count is reported as NA, never estimated.
- **Clusters and exclusions.** Clusters are recomputed after every exclusion, so
  excluding a bridging cell can split a cluster.
- **Units.** Physical units are used only after calibration. Pixel and physical
  values are never mixed in one column or one summary.
- **What counts as n.** Cells are not biological replicates. Replicate summaries
  pool fields within a well, weight wells equally within a replicate, and weight
  replicates equally within a condition. Condition *n* is the number of
  biological replicates. No significance tests are run.
- **DAPI counts.** These are nuclear segmentation estimates, not cell counts.

See the [user guide](docs/USER_GUIDE.md) for the workflow in detail,
[formats](docs/FORMATS.md) for project and export contents,
[performance](docs/PERFORMANCE.md) for benchmarks, and the
[development guide](docs/DEVELOPMENT.md) for architecture and testing.

## Data stays out of Git

This repository contains source code, documentation and synthetic tests only.
Microscopy images, projects, autosaves, model weights, reports, screenshots and
analysis outputs are ignored. Tests generate their own synthetic data.
