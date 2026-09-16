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

## After the performance pass (CellScope 0.2.0)

These numbers come from the same machine, workloads and commands as the
baseline.

### Measurement

| Workload | Stage | Before | After | Change |
|---|---|---|---|---|
| 4096×4096, 915 objects | `measure_clusters` | 89.1 s | 0.66 s | 135× faster |
| | `compute_results`, cold | 92.9 s | 1.73 s | 54× faster |
| | Contact graph | 0.32 s | 0.19 s | |
| | Overlay | 0.35 s | 0.26 s | |
| 2048×2048, 1693 objects | `compute_results`, cold | 8.49 s | 3.68 s | 2.3× faster |
| 800×800 synthetic, per image | `measure_clusters` | 316–345 ms | 53 ms | 6× faster |
| | `compute_results`, cold | 399–435 ms | 137 ms | 3× faster |
| | Contact graph | 14–16 ms | 11 ms | |
| | Overlay | 82–85 ms | 67 ms | |
| Local images, Cellpose | `compute_results`, cold | 137 ms | 49 ms | 2.8× faster |

After an exclusion, only the cluster that changed is re-measured. Every other
cell and cluster reuses its cached geometry.

### Batch throughput

| Workload | Before | After |
|---|---|---|
| 10 synthetic fields, threshold engine | 256 images/min | 389 images/min |
| 50 synthetic fields, threshold engine | 262 images/min | 417 images/min |
| 4096 px field, threshold engine | 9.6 images/min | 10.7 images/min |
| 10 local images, Cellpose on GPU | 6.9 images/min | 7.2 images/min |

The CPU engine gains 1.5–1.6× from preparing and finalising images on worker
threads. Cellpose is limited by the network itself: mask reconstruction is
negligible, and bfloat16 is already the faster precision on this card (7.7 s
against 9.0 s per image in float32). The Cellpose model now loads in 3.3 s,
down from 5.7 s, because it is loaded once and cached.

### Memory

| Measure | Before | After |
|---|---|---|
| Peak VRAM during the Cellpose batch (reserved) | 1902 MB | 1902 MB |
| GPU memory, whole device (`nvidia-smi`) | — | 2.3 GB of 4 GB at peak |
| Peak RSS, 50 synthetic fields | 1092 MB | 1116 MB |
| Peak RSS, 4096 px field | 2149 MB | 2091 MB |
| Undo snapshot of a mask | full copy | shared reference |

Label arrays are now read-only and copy-on-write, so undo snapshots and autosave
snapshots hold references instead of copying masks.

### Interface gestures

These are server milliseconds per gesture, measured as in the baseline.
*Before* includes the chained follow-up events each gesture triggered; *after*
is the single combined response.

| Gesture | Before, synthetic | After, synthetic | Before, local | After, local |
|---|---|---|---|---|
| Change opacity | 98–114 | **0** (browser only) | 84 | **0** |
| Zoom or pan | 100–114 | **0** (browser only) | 81 | **0** |
| Inspect click | 109–123 | 9 | 95 | 9 |
| Exclude click | 496–545 | 130 | 204 | 89 |
| Undo | 489–549 | 129 | 218 | 93 |
| Next image | 112–137 | 115 | 88 | 89 |
| Results tab (repeat visit) | 89–110 | 0.6 | 134 | 0.6 |
| Status strip | 2.4–2.9 | 0.2 | 5.0 | 0.2 |

**Browser check.** In the real app, dragging and changing opacity sent no
requests, and a zoomed view kept its position across updates.

**Where "next image" goes.** It now also encodes the three layers the viewer
needs, which is why it did not get faster. The encoded layers are cached, so
returning to an image is faster than the first visit.

### Persistence

| Operation (10 images) | Before | After |
|---|---|---|
| Autosave write | 0.46 s, blocking | 0.33 s, in the background; skipped when nothing changed |
| Manual save | 2.19 s | 1.30 s |
| Load project | 0.25 s | 0.20 s |
| Export | 4.31 s | 4.57 s (now also writes the manifest, checksums and replicate tables) |

On a real 39-image project, taking the autosave snapshot holds the batch lock
for 29 ms, and the compressed write then takes 1.8 s in the background.

## Scientific regression check

The same local images were segmented and measured with the baseline commit
(`22d3b91`) and with 0.2.0, using identical settings. Masks, objects and QC
decisions were compared, and per-cell and per-cluster tables were compared at a
relative tolerance of 1e-12.

| Engine | Images | Identical | Differing |
|---|---|---|---|
| threshold + watershed | 97 | 93 | 4, all with a burned-in scale bar |
| Cellpose (`cpsam`, GPU) | 12 | 9 | 3, all with a burned-in scale bar |

**Masks.** In every image, masks and object counts were identical.

**Where the results differ.** The only differences are the new automatic
exclusion of objects touching the burned-in scale bar. The old version measured
the bar and its lettering as cells. Every remaining cell's geometry is unchanged.

**The new column.** Cell tables gained a `touches_annotation` column; no
existing column changed meaning.

**Crash safety.** Killing a process eight times in the middle of saving a
39-image project always left a project that loaded, and the orphaned temporary
files were removed by the next start's clean-up.
