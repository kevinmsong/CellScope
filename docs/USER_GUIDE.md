# User guide

## Projects and recovery

**Save project** writes an atomic local recovery file and offers a download. The
portable file contains the original images, nuclear/reference files, masks,
calibration, channels, parameters, exclusions, review notes, design labels and
nuclear analysis. It can be opened on another installation of the same version.

Autosave runs every 60 seconds while the browser is connected and after each
completed image in a normal batch or resumed run. The local `projects/` directory
survives an app restart. Use **Refresh recovery list**, select a file, then
**Recover selected autosave**. Recovery names are stable project identifiers;
open one to see the experiment and images. Saving or opening another project
preserves the current nonempty project first.

An interrupted inference is not resumable halfway through a single image. Resume
skips completed images and retries pending or failed images. Changes made since
the last autosave can be lost if the browser/server terminates; save manually at
important checkpoints. Undo history is intentionally session-only; the applied
edits and audit history survive saving and reopening.

## Calibrating from a separate ruler

Enter the ruler's measured length in pixels and the physical length it represents
in µm. The conversion is `µm per pixel = known µm / measured ruler pixels`.
An 800x600 reference can calibrate an 800x800 sample: image width/height and crop
size do not change that ratio. Both dimensions are retained in calibration metadata.

The reference must still have the same magnification and pixel scaling as the
sample. Entering the ruler's µm length does not compensate for independently
resizing one image or changing camera binning. Use a matching reference or enter
the sample's known resolution directly in those cases.

## Reviewing cells

The default click action toggles inclusion. Excluded cells remain outlined in
gray, so click their interior again to restore them. **Inspect** selects a cell
without changing QC. Its boundary is highlighted in white, and its measurements
appear below the image. Clicking a row in **Linked cell measurements** selects
the corresponding object. Excluded-cell measurements are diagnostic only.

Open **View and correction tools** to zoom (up to 8x), pan with horizontal/vertical
position controls, adjust overlay opacity, or show the original image. **Fit image**
returns to the full field. Zoom enlarges the display preview, not the measurement
mask; very small structures may need a higher-resolution source/display workflow.
The image's fullscreen button opens a larger view.

Undo/redo operates on complete exclusion or correction actions, including bulk
filters. Up to 12 full-mask checkpoints are kept in memory. Ctrl+Z and
Ctrl+Shift+Z work when the Undo/Redo controls are visible and the focus is outside
a text input. New edits clear redo history. Re-segmentation starts a new history.

**Mark reviewed** is available after segmentation. QC edits, corrections and
re-segmentation clear that status. Changes to calibration, channels, or analysis
settings invalidate the recorded review signature when status/readiness is
refreshed. Results remain available, but readiness identifies outstanding review.

## Correcting segmentation

Corrections edit the working object mask. They never edit the original source
pixels or raw segmentation mask. Each operation is recorded with time, IDs and
original-image coordinates.

- **Merge:** enter two or more touching IDs. They must have consistent inclusion
  status. The merged object keeps the first ID. Review it as one resolved cell.
- **Split:** enter one ID; choose **Collect correction points** and click two
  distinct seeds inside it. The split uses watershed on the distance transform.
  Inspect both resulting boundaries before accepting the result.
- **Replace boundary:** enter one ID and click at least three polygon vertices in
  order. The polygon becomes that cell's complete boundary. Overlap with another
  cell is rejected. This is a polygon replacement, not a freehand brush.

Points are drawn in yellow and also shown as editable JSON. Use **Clear points**
before another correction. A split child inherits an excluded parent's QC state.
Undo restores the previous mask and inclusion decisions. Automatic edge exclusion
runs after segmentation; inspect manual corrections for newly introduced edge
contacts. **Reset all QC** restores all objects, including edge objects, while
retaining an audit trail. Re-segmenting reapplies the automatic edge filter.

## DAPI nuclei

Choose **Image channel / blue** for a composite RGB image. For a grayscale DAPI
image, choose grayscale. A separate nuclear file must first be attached on Batch;
select **Separate nuclear file** and its appropriate channel.

Nuclear segmentation uses Gaussian smoothing, Otsu thresholding, a minimum area
filter, and distance-transform watershed. Seed separation controls how many peaks
are allowed within a connected DAPI region. These are estimates: inspect crowded
or unevenly stained nuclei carefully, and tune settings to the image resolution.
This method can over-split or under-split irregular nuclei.

**Direct contact** includes shared pixel edges and corners. Even a one-pixel
background gap does not count. The app reports:

| Count | Meaning |
|---|---|
| Included nuclei | Segmented nuclei remaining after nuclear QC |
| Touching nuclei | Included nuclei contacting at least one other included nucleus |
| Touching groups | Connected components containing at least two nuclei |
| Isolated nuclei | Included nuclei with no direct nuclear contact |
| Contact pairs | Unique pairs of directly contacting included nuclei |

Edge nuclei are excluded automatically. Click a nucleus to exclude/restore it;
contact groups are recomputed immediately. Nuclear exclusions are independent
of cell exclusions. **Export nuclear counts and masks** downloads a separate
archive even when no cytoplasmic cell analysis has been run. Re-running nuclei replaces the previous nuclear masks and
nuclear QC log. Save a project before a run if you need to preserve both versions.

## Presets and processing

Presets store the batch's applied channel, preprocessing, segmentation, splitting,
and clustering settings. They exclude calibration, images and QC decisions.
Download a named JSON preset, then upload and apply it to another batch. Inspect
the populated Segment controls before running. A setting changed in a widget but
not applied by running segmentation is not yet part of the batch preset.

**Resume pending images** uses each image's saved settings and skips completed
images. **Retry failed images** processes failures only. Cancellation stops future
work; the current inference call may finish first. Completed images are saved.
Remaining-time estimates extrapolate elapsed time and may be inaccurate during
model loading or when images differ greatly in size.

## Experimental design and readiness

Refresh readiness before editing design. Each image needs a well, condition and
biological replicate label to enter experimental comparisons. Use consistent
labels for fields from the same well and technical wells from the same specimen.

Fields pool within wells, so wells with more fields contribute more cells to that
well's mean. Each well then has equal weight within its biological replicate;
each biological replicate has equal weight within a condition. Between-replicate
SD is NA for a single replicate. Pixel and physical-unit groups stay separate.
These summaries currently compare cell area; the full cell table provides other
metrics for additional analyses.

Readiness flags missing segmentation/calibration, unreviewed images, changed
channels, differing settings, unresolved groups, and more than 50% excluded
objects. These are prompts for inspection, not automatic diagnoses.
