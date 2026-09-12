# Project and export formats

## Portable projects

A `.cellscope` file is a ZIP archive containing `project.json` (schema version 1)
and NumPy arrays under `arrays/`. Arrays are read with `allow_pickle=False`;
loading a project does not execute Python objects or extract archive paths.
Known dataclass names and fields are validated, and the reader rejects unknown
schema versions, invalid cell label dimensions, and mismatched object IDs.
The uncompressed archive limit is 8 GiB.

Projects preserve source pixels and their recorded hashes, calibration, parameters,
cell and nuclear masks, exclusions, QC/correction logs, review notes and experimental
design. Source pixels and raw cell segmentation are restored as read-only arrays.
Derived measurement caches, temporary detection proposals, and undo/redo stacks
are rebuilt or reset rather than serialized.

Local saves use a temporary file and atomic replacement, so a failed write does
not replace the previous good project. Back up downloaded projects separately;
autosave is recovery storage, not a substitute for backup.

## Analysis archive

`batch_analysis.zip` includes the following root files:

| File | Contents |
|---|---|
| `batch_summary.csv` | Pooled accepted-cell measurements |
| `per_image_summary.csv` | Counts and measurements for each segmented image |
| `all_cells.csv` | Accepted resolved cells with image and well provenance |
| `all_clusters.csv` | QC-approved cell clusters |
| `qc_log.csv` | Cell exclusion/restore history across images |
| `readiness.csv` | Checks captured at export time |
| `well_summary.csv` | Cell area summarized within wells |
| `replicate_summary.csv` | Equal-weight well means within biological replicates |
| `condition_summary.csv` | Replicate count, mean and between-replicate SD |
| `nuclei_counts.csv` | Separate DAPI counts for analyzed images |
| `report.pdf` | Readable summary, overlays, definitions, methods and QC history |
| `metadata.json` | Batch and image provenance, settings and design labels |

DAPI-only images also retain their nuclear masks and metadata in mixed batches.
The DAPI tab offers a standalone `nuclei_analysis.zip` download without requiring
cell segmentation.

Each cell-segmented image has a folder under `images/`, including its measurement
CSVs and metadata, plus:

- `masks/raw_labels.tif`: original segmentation engine output.
- `object_labels.tif`: split/manually corrected object mask before exclusions.
- `masks/qc_labels.tif`: labels remaining after cell exclusions.
- `edit_log.json`: manual correction and undo/redo history.
- `nuclei_labels.tif`: independent pre-QC DAPI masks, when analyzed.
- `nuclei_measurements.csv`: nuclear IDs, inclusion, direct contact, groups and area.
- `nuclei_metadata.json`: nuclear method/settings, counts, exclusions and QC log.

IDs link masks to measurements. Excluded labels remain in pre-QC masks and are
identified in the corresponding QC metadata. Nuclear IDs and cell IDs are
separate namespaces; matching numbers do not imply a nucleus belongs to that cell.
Unresolved cell groups do not receive invented individual cell measurements.

## Presets

Presets are JSON with schema version 1, a name, channel choices and parameter
records. They contain no images and no calibration. Preserve the JSON alongside
an experiment protocol if a named analysis configuration must be shared.
