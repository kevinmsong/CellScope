# Performance

How fast CellScope is, how that was measured, and how to check that a change has
not made it slower.

## Running the benchmarks

```powershell
python tools/bench.py                          # quick: 1 and 10 synthetic fields, UI latency
python tools/bench.py --suite full             # adds 50 fields, a 4096 px field, a dense field
python tools/bench.py --engine cellpose --images "images/some folder" --limit 10
```

- **Inputs.** Synthetic fields are seeded, so repeated runs see identical inputs.
  `--images` runs on local lab images, which never enter the repository.
- **Output.** Each run writes JSON to `output/performance/` (gitignored) and
  prints a Markdown table.
- **UI latency.** This is the server-side cost of one user gesture, including
  every chained follow-up event, measured by calling the callbacks directly. It
  is the median of five repeats after a warm-up run.
- **Synthetic fields.** They use the fast `threshold_watershed` engine, so their
  timings isolate CellScope's own code from Cellpose.

## Machine

| Component | Detail |
|---|---|
| CPU | Intel Core i7 (Tiger Lake-H, 16 logical cores) |
| GPU | NVIDIA T1200 Laptop, 4 GB |
| OS | Windows 11 |
| Software | Python 3.12.3, numpy 1.26.4, torch 2.5.1+cu124, Cellpose 4.2.1 (`cpsam`) |

The repository sits inside a OneDrive folder, which adds file-system noise to the
save and export timings.

## Baseline (before the performance pass, commit `22d3b91`)

### Synthetic fields, 800×800, threshold engine

| Stage | 10 images (median per image) | 50 images (median per image) |
|---|---|---|
| Preprocess | 16 ms | 16 ms |
| Split | 112 ms | 107 ms |
| Contact graph | 14 ms | 16 ms |
| `measure_cells` | 71 ms | 78 ms |
| `measure_clusters` | 316 ms | 345 ms |
| `compute_results`, cold | 399 ms | 435 ms |
| Overlay | 82 ms | 85 ms |
| Throughput | 256 images/min | 262 images/min |
| Peak RSS | 619 MB | 1092 MB (+399 MB during run) |

### Large and crowded images

| Workload | Measure | Value |
|---|---|---|
| 4096×4096, 915 objects | Preprocess | 0.76 s |
| | Split | 1.62 s |
| | `measure_cells` | 0.84 s |
| | **`measure_clusters`** | **89.1 s** |
| | `compute_results`, cold | **92.9 s** |
| | Peak RSS | 2149 MB |
| 2048×2048, 1693 dense objects | `compute_results`, cold | 8.49 s |
| | Peak RSS | 1786 MB |

### Local lab images (10 × 800×800), Cellpose `cpsam` on GPU

| Measure | Value |
|---|---|
| Model load | 5.7 s |
| Inference per image | ~8.6 s (batch total 86.4 s) |
| Throughput | 6.9 images/min |
| `compute_results`, cold | 137 ms |
| Peak VRAM, reserved | 1902 MB |
| Peak RSS | 1877 MB |

### UI gestures (server time per gesture)

| Gesture | Synthetic, 10 images | Local, 10 images |
|---|---|---|
| Next image | 112–137 ms | 88 ms |
| Change opacity | 98–114 ms | 84 ms |
| Change zoom | 100–114 ms | 81 ms |
| Inspect click | 109–123 ms | 95 ms |
| **Exclude click** | **496–545 ms** | 204 ms |
| **Undo** | **489–549 ms** | 218 ms |
| Results tab | 89–110 ms | 134 ms |

### Persistence (10 images)

| Operation | Synthetic | Local |
|---|---|---|
| Autosave (zip level 1) | 0.46 s | 0.49 s |
| Manual save (zip level 6) | 2.19 s | 1.06 s |
| Load project | 0.25 s | 0.26 s |
| PDF report | 3.43 s | 2.11 s |
| Batch export | 4.31 s | 2.81 s |

### What the baseline shows

- **Cluster geometry dominates measurement.** `measure_clusters` makes full-image
  passes for every cluster, so its cost grows with image area × cluster count. On a
  4096 px field, the first exclusion therefore stalls the interface for about 90 s.
- **Cosmetic controls re-render everything.** Opacity and zoom each rebuild the
  whole overlay, the review table and every batch-wide status view.
- **A single exclusion re-measures the image.** Exclude and undo both re-measure
  every cell, and undo also rebuilds the cluster geometry.
- **Throughput with Cellpose is bounded by inference.** On this 4 GB card,
  inference is over 95% of the per-image time. CPU pipelining around it can
  shorten a batch by a few percent at most.
