# User guide

The workflow bar at the top of every page shows six steps. Each step is marked
done (✓), current, needing attention (!), a problem (✕) or optional, and the
*Next* line says what to do next.

## 1. Import and calibrate

**Adding images.** Add JPEG, PNG or TIFF files; you can add more later. Images
are recognised by content, so the same file added twice is skipped, while two
different files that happen to share a name are both kept.

**Channels.** Choose the **cell channel** (what is measured) and, optionally, a
**nuclear guide** channel. The guide helps separate touching cells but is never
measured. Both settings apply to the whole batch. To override a single image,
edit its row in the image table. A separate nuclear file can be attached to the
active image.

**Calibration.** Choose one method:

- **Scale bar in image.** **Detect bar in this image** finds a yellow or white
  bar drawn on the image. You can also click the bar's two ends. Enter the
  length the caption states (for example 50 µm) and apply.
- **Reference frame.** Upload a separate ruler image taken at the same
  magnification; it may be a different size from the samples. Confirm the
  detected pixel length and the µm value.
- **Enter resolution.** Type µm per pixel; X and Y may differ.
- **Work in pixels.** Confirms that results will be in px and px² only.

Calibration is never applied silently. A batch that mixes calibrated and
uncalibrated images is flagged, and its pooled cells are not combined.

## 2. Segment

The settings column lists the controls you are most likely to change first.
Automatic exclusions, speed and hardware, preprocessing and touching-cell
controls are in the collapsed sections below them.

1. **Preview on this image** segments a copy of the active image. The
   **Before and after** table compares these values with the image's current
   segmentation:
   - objects and accepted cells;
   - exclusions at the border, at the scale bar, and otherwise;
   - unresolved groups and multi-cell clusters;
   - median area;
   - runtime.

   Choose **Keep preview** or **Discard preview**. Nothing changes until you keep
   it. **Segment this image now** runs and keeps the result in one step.
2. **Use these settings for the batch** makes them the batch defaults *without
   running anything*. Images already segmented with other settings are marked
   **stale**, and the note under the settings says how many would change.
3. **Run batch** segments images that need it: new, failed or stale ones.
   Choose *All images* to re-run everything; this clears their review.
   - **Progress** shows each image's state, object count and runtime, along with
     an estimate of the time left.
   - **Cancel** lets the image on the GPU finish, then stops; finished images
     keep their results.
   - **Retry failed images** runs only the failures.

**Automatic exclusions.** After segmentation, two rules run:

- objects touching the **image edge** are excluded;
- objects touching a **scale bar or caption burned into the image**, or lying
  within 2 px of it, are excluded.

Only saturated yellow or white bars in the outer quarter of the frame are
recognised. Objects that are the bar or its lettering are excluded by the same
rule. So is any cell whose strip was merged into a lettering object. Both rules
are logged and shown on the overlay. The scale-bar zone has a dotted outline.
You can switch either rule off, then use **Apply rules to all images**. This
updates exclusions without re-segmenting and never overrides a cell you
restored by hand.

**GPU memory.** If the GPU runs out of memory, the image is retried with a
smaller tile batch, and then on the CPU. The image's badge and the export
record which happened. With the `auto` engine, CellScope falls back to the
threshold engine only when Cellpose cannot be loaded at all. Any other Cellpose
failure is reported as that image's error.

## 3. Review

The viewer runs in the browser, so viewing controls respond immediately:

- **Zoom:** scroll wheel, <kbd>+</kbd>/<kbd>-</kbd>, **Fit** (<kbd>F</kbd>) or
  **1:1** (<kbd>1</kbd>).
- **Pan:** drag the image.
- **Overlay:** adjust its opacity, or press **Original** (<kbd>O</kbd>) to hide
  it.
- **Hover:** shows an object's number, whether it is a cell or an unresolved
  group, whether it is included, its area, and why it was excluded.

The **click mode** decides what a click does:

| Mode | Key | A click… |
|---|---|---|
| Toggle inclusion | <kbd>T</kbd> | excludes an included cell or restores an excluded one |
| Inspect | <kbd>I</kbd> | selects a cell without changing anything |
| Collect correction points | <kbd>P</kbd> | adds a point for a split or boundary |

**The selected cell.**

- It has a white outline, its values appear under **Selected cell**, and its row
  is highlighted in **Measurements for this image**.
- Clicking a row selects that cell, and you can also type its number.
- <kbd>X</kbd> excludes or restores it.
- Measurements of excluded cells are diagnostic only.

**Moving between images.** <kbd>←</kbd>/<kbd>→</kbd> move between images, and
**Next unreviewed** skips images already reviewed. <kbd>R</kbd> marks the image
reviewed, with an optional note.

**Undo.** <kbd>Ctrl</kbd>+<kbd>Z</kbd> and <kbd>Ctrl</kbd>+<kbd>Shift</kbd>+<kbd>Z</kbd>
undo and redo whole actions, including bulk filters. The last 12 are kept for
each image. Shortcuts are ignored while you type in a text box. Press
<kbd>?</kbd> for the full list.

**What invalidates a review.** Any exclusion, correction or re-segmentation
clears the reviewed mark. So does a later change to calibration, channels or
settings.

## Correcting segmentation

Corrections change the working mask only; the source image and the raw
segmentation are never modified. Each correction is logged with its time, IDs
and original-image coordinates. The **Cell IDs** box defaults to the selected
cell.

- **Merge:** two or more touching cells with the same inclusion state. The
  merged cell keeps the first ID.
- **Split:** one cell, plus two seed points inside it. The split uses a
  watershed on the distance transform.
- **Replace boundary:** one cell, plus three or more polygon vertices in order.
  Overlapping another cell is refused.

An edited cell that now touches the edge or a scale bar is excluded, as it
would be after segmentation, unless you restored it by hand. A split child
inherits an excluded parent's state. **Restore all** restores everything,
including automatic exclusions, and can be undone.

## 4. Design and compare

**Labelling images.** Give every image a **well**, **condition** and
**biological replicate**, then choose **Save design**. Reuse a replicate label
for technical wells of the same biological sample. Images missing a label are
listed and left out of the comparison, never guessed.

**How the comparison is built.** Choose a measurement, such as area, axes,
Feret diameter, perimeter, aspect ratio, circularity or solidity. Then:

1. Cells from all fields of a well are pooled into a **well** mean.
2. Wells count equally within a **biological replicate**.
3. Replicates count equally within a **condition**.

**What each level reports.** Every level shows its own n, mean, median and SD.
The condition *n* is the number of biological replicates. Calibrated and
uncalibrated images appear in separate rows. The dot plot shows one point per
replicate, with a bar at the condition mean.

**Cautions.** A note appears when a condition has fewer than three replicates
or includes unreviewed images. No significance tests are run: choose a test
that matches your design, with biological replicates as n.

**Readiness** lists outstanding checks. **Build PDF report** writes the report
on its own.

## 5. Results

Descriptive summaries and distributions of every accepted cell in the batch,
pooled. They describe the batch; they are not a between-condition comparison.

## 6. Export

**Build analysis archive** writes `batch_analysis.zip`; see
[formats](FORMATS.md). The workflow bar marks Export done until something
changes.

## Projects, autosave and recovery

**Saving.** **Save project** gives you a portable `.cellscope` file and updates
the local recovery copy. The file holds the images, masks, settings, QC
decisions, corrections, review notes, design and DAPI analysis.

**Autosave.** Autosave writes in the background whenever something has changed,
so it never pauses the interface. It checks every 30 seconds by default (see
`--autosave-interval`) and also after each image in a batch run. Saves are
atomic, so an interrupted save leaves the previous copy intact.

**Recovery.** **Autosaved projects** lists recovery copies by experiment name,
size and date. Opening one first saves the project that is currently open.
Undo history is not saved.

**Older projects.** Projects from CellScope 0.1 open with their results and
reviews intact. A newer project that this version cannot read is refused with
a message naming the version that wrote it.

**Presets.** A preset holds the channels and all analysis settings, including
the automatic exclusion rules, but not the calibration. Applying a preset marks
affected images stale; it does not re-run them.

## DAPI nuclei

**Source.** Choose **Image channel / blue** for an RGB composite, or
**Separate nuclear file** after attaching one on Import.

**Method.** Nuclei are found by Gaussian smoothing, an Otsu threshold, a minimum
area and a distance-transform watershed. **Seed separation** controls how
readily touching nuclei are split. These are estimates, so check crowded
fields.

**Direct contact** means the nuclear masks share an edge or a corner; any gap
disqualifies. The counts are:

| Count | Meaning |
|---|---|
| Included nuclei | Segmented nuclei remaining after nuclear QC |
| Touching nuclei | Included nuclei touching at least one other |
| Touching groups | Connected groups of at least two nuclei |
| Isolated nuclei | Included nuclei touching none |
| Contact pairs | Unique touching pairs |

**Nuclear QC.** Nuclei at the image edge are excluded automatically. Click a
nucleus to exclude or restore it. Nuclear and cell exclusions are independent,
and running the DAPI analysis again replaces the previous nuclear result.
