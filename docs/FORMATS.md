# Project and export formats

## Portable projects (`.cellscope`)

A ZIP archive with `project.json` and one NumPy array per file under `arrays/`.

**Safety**

- Arrays are read with `allow_pickle=False`.
- Records are rebuilt only from CellScope's own dataclasses.
- Archive paths are never extracted.
- The uncompressed size is limited to 8 GiB.

**Schema 2** (CellScope 0.2). `project.json` holds:

```json
{"schema": 2, "app": {"cellscope": "0.2.0", "python": "…", "numpy": "…"},
 "saved_at": "…", "batch": { … }}
```

**What a project stores**

- Source pixels with their SHA-256, calibration, channels and every parameter
  group, including the automatic exclusion rules.
- Raw and working masks, object records, burned-in scale-bar detections, and the
  fingerprint of the settings each segmentation used.
- Exclusions, QC and correction logs, review state and notes.
- Experimental design and DAPI analysis.

**What is rebuilt on load instead of stored:** display copies, measurement
caches and undo history.

**Compatibility**

| Case | Behaviour |
|---|---|
| Schema 1 (CellScope 0.1) | Migrated in memory; results and reviews are unchanged |
| Newer schema | Refused, naming the version that wrote it |
| Unknown field on a known record | Dropped, with a warning |
| Unknown record type | Refused |
| Truncated, damaged or incomplete file | `ProjectError` naming the problem |
| Masks that do not match the image or the object list | Refused |

**Writes.** A project is written to a temporary file beside its target, flushed,
and moved into place, with retries while a sync client or virus scanner briefly
holds the file. A failed save leaves the previous file untouched. Temporary files
left by an interrupted save are removed on the next start.

## Analysis archive (`batch_analysis.zip`)

| File | Contents |
|---|---|
| `manifest.json` | Machine-readable provenance (below) |
| `README.txt` | A short guide to the archive |
| `batch_summary.csv` | Every accepted cell pooled (descriptive) |
| `per_image_summary.csv` | One row per image, including exclusions by reason |
| `all_cells.csv` | Accepted, resolved cells tagged with image and well |
| `all_clusters.csv` | Clusters; `cell_count` is NA when a member is unresolved |
| `design_images.csv`, `design_wells.csv`, `design_replicates.csv`, `design_conditions.csv` | The replicate hierarchy for cell area, with n at each level |
| `well_summary.csv`, `replicate_summary.csv`, `condition_summary.csv` | The same hierarchy in the 0.1 layout |
| `qc_log.csv` | Every exclusion and restoration, with its reason |
| `readiness.csv` | Checks outstanding at export time |
| `nuclei_counts.csv` | Independent DAPI counts |
| `report.pdf` | Readable report: status, replicate tables and dot plot, overlays, definitions, methods, QC history |
| `metadata.json` | Batch provenance (0.1 layout, extended) |

**Per-image folders.** Each segmented image has a folder `images/<name>/`
containing:

| File | Contents |
|---|---|
| `cell_measurements.csv` | Per-cell measurements; includes `touches_border` and `touches_annotation` |
| `cluster_measurements.csv` | Per-cluster measurements |
| `image_summary.csv` | One row describing the image |
| `contact_graph.csv` | Which cells touch which |
| `metadata.json` | Full per-image provenance, including software versions, automatic QC rules and the burned-in annotation |
| `masks/raw_labels.tif` | The segmentation engine's output |
| `object_labels.tif` | After splitting and manual correction, before exclusions |
| `masks/qc_labels.tif` | What was measured |
| `overlays/segmentation_overlay.png` | The overlay image |
| `edit_log.json` | Manual corrections |
| `nuclei_*` | Nuclear masks, measurements and metadata, when analysed |

**How IDs relate**

- IDs link masks to measurements.
- Nuclear IDs and cell IDs are separate namespaces.
- Unresolved groups never receive per-cell measurements.

### `manifest.json` (manifest_schema 1)

| Key | Contents |
|---|---|
| `software` | Versions of CellScope, Python, the OS, numpy, scipy, scikit-image, pandas, Pillow, tifffile, Cellpose, PyTorch, CUDA and the GPU (when PyTorch ran), Gradio, ReportLab, Plotly |
| `batch` | ID, label, image/segmented/reviewed counts, units, calibration consistency, whether calibration was confirmed |
| `statistics` | The replicate aggregation rule, the pooled-summary rule, the rule for cluster cell counts, and `"tests": "none"` |
| `shared_parameters` | Batch default settings |
| `images[]` | One entry per image (below) |
| `files[]` | Path, size and SHA-256 of every other file in the archive |

Each `images[]` entry records:

- the source image (name, size, SHA-256), design labels and channels;
- calibration and every parameter group;
- inference details: engine, model, device, tile batch, analysis scale, any
  out-of-memory recovery, and stage timings;
- the segmentation fingerprint, and whether settings changed since segmentation;
- the burned-in scale-bar detection;
- QC exclusions by reason, correction counts and operations;
- review state, note and signature;
- counts, any error, and the folder in the archive.

## Presets

JSON with `schema` 2, a name, channel choices, and the preprocessing,
segmentation, splitting, clustering and automatic QC settings. Schema 1 presets,
which have no QC group, still apply with default rules. Presets never contain
images, calibration or QC decisions.
