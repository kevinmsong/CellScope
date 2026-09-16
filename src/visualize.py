"""Overlays and plots (§13, §15).

The three segmentation outcomes get visually distinct boundaries, because the
one a researcher most needs to notice is the unresolved group -- the object
whose cell count CellScope is declining to guess.

Colours are chosen to stay distinguishable against the dark backgrounds typical
of fluorescence images and to remain separable for the most common forms of
colour vision deficiency (green / blue / orange rather than red / green).
"""

from __future__ import annotations

from typing import Iterable, Sequence

import numpy as np
from PIL import Image, ImageDraw
from skimage.morphology import binary_dilation
from skimage.segmentation import find_boundaries

#: Categorical slots 1-3 of the validated reference palette. Verified with the
#: palette validator at ``--pairs all`` (needed because scatter plots put every
#: pair on screen together): worst CVD dE 9.4, worst normal-vision dE 20.9, all
#: three above 3:1 contrast on a dark surface. Substituting a red for the
#: unresolved slot was tried and hard-failed (red vs orange: normal-vision
#: dE 6.8), so alarm is carried by line weight instead of hue -- see
#: ``UNRESOLVED_BOUNDARY_WIDTH``.
#: Overlays sit on near-black microscopy images, so the dark steps are used.
CATEGORY_COLOURS: dict[str, tuple[int, int, int]] = {
    "isolated": (57, 135, 229),     # slot 1, blue  - a resolved single cell
    "clustered": (217, 89, 38),     # slot 2, orange - resolved, touching neighbours
    "unresolved": (25, 158, 112),   # slot 3, aqua  - cell count deliberately not claimed
}

#: Light-mode steps of the same slots, for Plotly figures drawn on a light
#: chart surface. Only two categories ever appear in the plots, because
#: unresolved groups contribute no per-cell rows (§12).
#: Darker steps of the same two hues, for contrast against a white chart
#: surface: 6.32 and 5.18 against white, up from 4.42 and 3.20. Re-validated as
#: a pair at --pairs all: CVD dE 23.2 (protan), normal-vision dE well clear of
#: the floor, both above 3:1 on the surface.
PLOT_COLOURS: dict[str, str] = {
    "isolated": "#1f5fb0",
    "clustered": "#c2410c",
}

#: The unresolved boundary is drawn this many pixels thick. Identity must never
#: rest on colour alone, and this is the category a researcher most needs to
#: notice, so it carries a second, non-colour signal.
UNRESOLVED_BOUNDARY_WIDTH = 3

#: Marker symbols per series. Identity must not rest on hue alone, so the two
#: series differ in shape as well as colour -- readable in greyscale, in print,
#: and for any colour vision deficiency.
PLOT_SYMBOLS: dict[str, str] = {"isolated": "circle", "clustered": "diamond"}

#: A sans stack with real fallbacks, so figures do not silently drop to a serif
#: default on a machine without the first choice.
PLOT_FONT = "Inter, 'Helvetica Neue', Helvetica, Arial, sans-serif"

PLOT_SURFACE = "#ffffff"
PLOT_TEXT = "#0B1220"
PLOT_MUTED = "#4B5563"
PLOT_GRID = "#D8DEDA"

CATEGORY_LABELS = {
    "isolated": "isolated cell",
    "clustered": "cell in a cluster",
    "unresolved": "unresolved cluster (cell count NA)",
}


def categorise(objects, clusters) -> dict[int, str]:
    """Map each object ID to its display category."""
    cluster_of = {}
    for cluster in clusters:
        for member in cluster.member_ids:
            cluster_of[member] = cluster

    categories: dict[int, str] = {}
    for obj in objects:
        if not obj.is_resolved:
            categories[obj.object_id] = "unresolved"
            continue
        cluster = cluster_of.get(obj.object_id)
        categories[obj.object_id] = (
            "isolated" if cluster is not None and cluster.cluster_size == 1 else "clustered"
        )
    return categories


def _resize_labels(labels: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    """Nearest-neighbour resize, which preserves label values exactly."""
    if labels.shape[:2] == shape:
        return labels
    # int32 arrays become PIL mode "I" without being told (the explicit ``mode``
    # argument is deprecated in Pillow 11).
    resized = Image.fromarray(np.ascontiguousarray(labels, dtype=np.int32)).resize(
        (shape[1], shape[0]), Image.NEAREST
    )
    return np.asarray(resized).astype(np.int32)


def make_overlay(
    display_rgb: np.ndarray,
    labels: np.ndarray,
    objects: Iterable,
    clusters: Iterable,
    alpha: float = 0.28,
    show_cell_ids: bool = True,
    show_cluster_ids: bool = False,
    excluded_ids: Iterable[int] = (),
    annotation=None,
) -> np.ndarray:
    """Draw masks, boundaries and IDs onto a display-sized RGB image.

    ``labels`` may be at full resolution; it is resized to the display grid with
    nearest-neighbour interpolation so IDs survive intact. ``annotation`` (a
    burned-in scale bar) is outlined with a dotted line marking the zone whose
    contact excludes a cell.
    """
    objects = list(objects)
    clusters = list(clusters)
    base = np.asarray(display_rgb).astype(np.float32)
    if base.ndim == 2:
        base = np.repeat(base[:, :, None], 3, axis=2)

    small = _resize_labels(np.asarray(labels), base.shape[:2])
    categories = categorise(objects, clusters)
    masks = _overlay_masks(small, categories, excluded_ids)

    if masks.member.any():
        tinted = (1.0 - alpha) * base + alpha * masks.tint
        base = np.where(masks.member[..., None], tinted, base)
    base = np.where(masks.edge[..., None], masks.tint, base)
    if masks.excluded.any():
        base = np.where(masks.excluded[..., None], base * 0.35, base)
        base = np.where(masks.excluded_edge[..., None], _EXCLUDED_EDGE, base)
    outline = annotation_outline(annotation, labels.shape[:2] if hasattr(labels, "shape") else None,
                                 small.shape)
    if outline is not None:
        base = np.where(outline[..., None], _EXCLUDED_EDGE, base)
    image = Image.fromarray(np.clip(base, 0, 255).astype(np.uint8))
    if show_cell_ids or show_cluster_ids:
        _draw_ids(image, small, objects, clusters, categories, show_cell_ids, show_cluster_ids)
    return np.asarray(image)


def annotation_outline(annotation, full_shape, display_shape) -> np.ndarray | None:
    """A dotted outline of a burned-in annotation's exclusion zone, at display size."""
    if annotation is None or not getattr(annotation, "found", False) or full_shape is None:
        return None
    from scipy import ndimage as ndi

    r0, c0, r1, c1 = annotation.mask_bbox
    zone = np.zeros(tuple(full_shape), bool)
    zone[r0:r1, c0:c1] = annotation.mask[: r1 - r0, : c1 - c0]
    margin = max(1, int(annotation.margin_px))
    yy, xx = np.mgrid[-margin : margin + 1, -margin : margin + 1]
    zone[r0:r1, c0:c1] = ndi.binary_dilation(
        zone[r0:r1, c0:c1], structure=(yy**2 + xx**2) <= margin**2
    )
    small = _resize_labels(zone.astype(np.int32), tuple(display_shape[:2])) > 0
    edge = find_boundaries(small, mode="outer") & ~small
    rows, cols = np.indices(edge.shape)
    return edge & ((rows + cols) % 3 != 0)


_CATEGORY_ORDER = ("isolated", "clustered", "unresolved")
_EXCLUDED_EDGE = np.asarray((160, 160, 160), dtype=np.float32)
_CATEGORY_TINTS = np.asarray(
    [(0, 0, 0)] + [CATEGORY_COLOURS[c] for c in _CATEGORY_ORDER], dtype=np.float32
)


class _Masks:
    """Per-pixel overlay masks, computed with lookup tables in whole-image passes.

    Equivalent to testing each category's IDs with ``np.isin`` and running
    ``find_boundaries`` on each masked image: an inner boundary pixel is one
    whose 4-neighbourhood holds any other value, and masking only ever replaces
    *other* labels, so the boundary of a subset is the full boundary restricted
    to that subset. ``test_golden_equivalence`` checks this pixel for pixel.
    """

    __slots__ = ("category", "edge", "excluded", "excluded_edge", "member", "tint")


def _overlay_masks(small: np.ndarray, categories: dict[int, str], excluded_ids) -> _Masks:
    top = int(small.max()) if small.size else 0
    table = np.zeros(top + 1, np.uint8)
    for object_id, category in categories.items():
        if 0 < object_id <= top:
            table[object_id] = _CATEGORY_ORDER.index(category) + 1
    excluded_table = np.zeros(top + 1, bool)
    excluded_list = [int(i) for i in excluded_ids if 0 <= int(i) <= top]
    excluded_table[excluded_list] = True

    masks = _Masks()
    masks.category = table[small]
    masks.member = masks.category > 0
    masks.tint = _CATEGORY_TINTS[masks.category]
    masks.excluded = excluded_table[small]
    boundary = (
        find_boundaries(small, mode="inner")
        if (masks.member.any() or masks.excluded.any()) else np.zeros(small.shape, bool)
    )
    edge = boundary & masks.member
    unresolved = masks.category == _CATEGORY_ORDER.index("unresolved") + 1
    if UNRESOLVED_BOUNDARY_WIDTH > 1 and unresolved.any():
        thick = binary_dilation(
            edge & unresolved,
            np.ones((UNRESOLVED_BOUNDARY_WIDTH, UNRESOLVED_BOUNDARY_WIDTH), bool),
        ) & unresolved
        edge = edge | thick
    masks.edge = edge
    masks.excluded_edge = boundary & masks.excluded
    return masks


def overlay_layers(
    shape: tuple[int, int],
    labels: np.ndarray,
    objects: Iterable,
    clusters: Iterable,
    show_cell_ids: bool = True,
    show_cluster_ids: bool = False,
    excluded_ids: Iterable[int] = (),
    selected_id: int = 0,
    annotation=None,
) -> tuple[np.ndarray, np.ndarray]:
    """The overlay as two RGBA layers for a browser to composite.

    ``fill`` carries the category tints at full opacity; the viewer shows it at
    the chosen overlay opacity, which reproduces :func:`make_overlay`'s blend
    without a server round trip. ``lines`` carries everything drawn opaque --
    boundaries, the dimming of excluded cells, IDs, and the selected cell.
    """
    objects = list(objects)
    clusters = list(clusters)
    small = _resize_labels(np.asarray(labels), tuple(shape[:2]))
    categories = categorise(objects, clusters)
    masks = _overlay_masks(small, categories, excluded_ids)

    fill = np.zeros((*small.shape, 4), np.uint8)
    fill[..., :3] = masks.tint.astype(np.uint8)
    fill[..., 3] = np.where(masks.member, 255, 0)

    lines = np.zeros((*small.shape, 4), np.uint8)
    lines[masks.excluded] = (0, 0, 0, 166)             # base x 0.35
    lines[masks.edge, :3] = masks.tint[masks.edge].astype(np.uint8)
    lines[masks.edge, 3] = 255
    lines[masks.excluded_edge] = (160, 160, 160, 255)
    outline = annotation_outline(annotation, np.asarray(labels).shape[:2], small.shape)
    if outline is not None:
        lines[outline] = (160, 160, 160, 255)
    if selected_id:
        _draw_selection(lines, small, int(selected_id))

    image = Image.fromarray(lines, "RGBA")
    if show_cell_ids or show_cluster_ids:
        _draw_ids(image, small, objects, clusters, categories, show_cell_ids, show_cluster_ids)
    return fill, np.asarray(image)


def _draw_selection(lines: np.ndarray, small: np.ndarray, selected_id: int) -> None:
    """A thick two-tone outline around one object, drawn inside its bounding box."""
    from scipy import ndimage as ndi

    rows, cols = np.nonzero(small == selected_id)
    if not len(rows):
        return
    r0, r1 = max(0, rows.min() - 3), min(small.shape[0], rows.max() + 4)
    c0, c1 = max(0, cols.min() - 3), min(small.shape[1], cols.max() + 4)
    own = small[r0:r1, c0:c1] == selected_id
    outer = ndi.binary_dilation(own, iterations=3) & ~ndi.binary_dilation(own, iterations=1)
    inner = find_boundaries(own, mode="inner")
    window = lines[r0:r1, c0:c1]
    window[outer] = (0, 0, 0, 255)
    window[inner] = (255, 255, 255, 255)


def _centroids(small_labels: np.ndarray, ids: Sequence[int]) -> dict[int, tuple[float, float]]:
    """Centroid of each label in display coordinates, computed in one pass."""
    centroids: dict[int, tuple[float, float]] = {}
    if not len(ids) or not small_labels.size:
        return centroids
    # Coordinate sums are integers, exact in float64, so these means equal
    # the per-object ``mean()`` of pixel coordinates bit for bit.
    flat = small_labels.ravel()
    height, width = small_labels.shape
    counts = np.bincount(flat)
    cols = np.bincount(flat, weights=np.tile(np.arange(width, dtype=np.float64), height))
    rows = np.bincount(flat, weights=np.repeat(np.arange(height, dtype=np.float64), width))
    for object_id in ids:
        if 0 < object_id < len(counts) and counts[object_id]:
            centroids[object_id] = (
                float(cols[object_id] / counts[object_id]),
                float(rows[object_id] / counts[object_id]),
            )
    return centroids


def _draw_ids(image, small_labels, objects, clusters, categories, show_cell_ids, show_cluster_ids):
    draw = ImageDraw.Draw(image)
    cluster_of = {m: c.cluster_id for c in clusters for m in c.member_ids}
    ids = [o.object_id for o in objects]
    centroids = _centroids(small_labels, ids)

    for obj in objects:
        position = centroids.get(obj.object_id)
        if position is None:
            continue
        parts = []
        if show_cell_ids:
            parts.append(str(obj.object_id))
        if show_cluster_ids and obj.object_id in cluster_of:
            parts.append("c{}".format(cluster_of[obj.object_id]))
        if not parts:
            continue
        text = " ".join(parts)
        colour = CATEGORY_COLOURS[categories.get(obj.object_id, "clustered")]
        x, y = position
        # A dark outline keeps the label readable over bright fluorescence.
        for dx, dy in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            draw.text((x + dx, y + dy), text, fill=(0, 0, 0))
        draw.text((x, y), text, fill=colour)


def overlay_legend() -> list[tuple[str, str]]:
    """(hex colour, description) pairs for the UI legend."""
    return [
        ("#{:02X}{:02X}{:02X}".format(*CATEGORY_COLOURS[key]), CATEGORY_LABELS[key])
        for key in ("isolated", "clustered", "unresolved")
    ]


# --------------------------------------------------------------------------- #
# Figures (§15)
# --------------------------------------------------------------------------- #
#
# Deliberately absent: trend lines, fitted curves, p-values, error bars across
# cells. §15 limits the MVP to descriptive plots and §16 forbids treating cells
# from one field as independent replicates. Adding a regression line here would
# quietly invite exactly that reading.
#
# The chart surface is pinned light rather than following the app theme, so the
# validated contrast of the palette actually holds wherever the app is run.


def _blank_figure(message: str):
    import plotly.graph_objects as go

    figure = go.Figure()
    figure.add_annotation(
        text=message, showarrow=False,
        font=dict(size=13, color=PLOT_MUTED, family=PLOT_FONT),
    )
    return _style(figure, "", "")


def _style(figure, x_title: str, y_title: str, show_legend: bool = False):
    figure.update_layout(
        paper_bgcolor=PLOT_SURFACE,
        plot_bgcolor=PLOT_SURFACE,
        font=dict(color=PLOT_TEXT, size=12, family=PLOT_FONT),
        margin=dict(l=56, r=20, t=36, b=48),
        showlegend=show_legend,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0, font=dict(size=11)),
        bargap=0.06,
        xaxis=dict(title=x_title, gridcolor=PLOT_GRID, zeroline=False,
                   linecolor=PLOT_GRID, tickfont=dict(color=PLOT_MUTED)),
        yaxis=dict(title=y_title, gridcolor=PLOT_GRID, zeroline=False,
                   linecolor=PLOT_GRID, tickfont=dict(color=PLOT_MUTED)),
    )
    return figure


def _split_by_status(cell_df):
    """(isolated rows, clustered rows) -- the distinction §15 asks for."""
    if "cell_status" not in cell_df:
        return cell_df, cell_df.iloc[0:0]
    return (
        cell_df[cell_df["cell_status"] == "isolated"],
        cell_df[cell_df["cell_status"] != "isolated"],
    )


def histogram_by_status(cell_df, column: str, x_title: str, title: str = ""):
    """Overlaid histogram of one metric, split isolated vs clustered."""
    import plotly.graph_objects as go

    if cell_df is None or not len(cell_df) or column not in cell_df:
        return _blank_figure("No measured cells yet")

    isolated, clustered = _split_by_status(cell_df)
    figure = go.Figure()
    for name, frame, key in (
        ("Isolated", isolated, "isolated"),
        ("In a cluster", clustered, "clustered"),
    ):
        if not len(frame):
            continue
        figure.add_trace(
            go.Histogram(
                x=frame[column].astype(float),
                name=name,
                marker=dict(
                    color=PLOT_COLOURS[key],
                    line=dict(
                        width=0 if key == "isolated" else 1.5,
                        color=PLOT_COLOURS[key],
                    ),
                ),
                opacity=0.55 if key == "clustered" else 0.85,
                hovertemplate="%{y} cells<br>" + x_title + " %{x}<extra>" + name + "</extra>",
            )
        )
    figure.update_layout(barmode="overlay", title=title)
    return _style(figure, x_title, "Cell count", show_legend=len(figure.data) > 1)


def scatter_by_status(cell_df, x_column, y_column, x_title, y_title, title=""):
    """Scatter of two metrics, split isolated vs clustered.

    No fitted line: §15 keeps the MVP descriptive, and a trend line over cells
    from a single field would imply an inference §16 rules out.
    """
    import plotly.graph_objects as go

    if cell_df is None or not len(cell_df):
        return _blank_figure("No measured cells yet")
    if x_column not in cell_df or y_column not in cell_df:
        return _blank_figure("Measurement not available")

    isolated, clustered = _split_by_status(cell_df)
    figure = go.Figure()
    for name, frame, key in (
        ("Isolated", isolated, "isolated"),
        ("In a cluster", clustered, "clustered"),
    ):
        if not len(frame):
            continue
        figure.add_trace(
            go.Scatter(
                x=frame[x_column].astype(float),
                y=frame[y_column].astype(float),
                mode="markers",
                name=name,
                customdata=frame["cell_id"],
                marker=dict(
                    color=PLOT_COLOURS[key],
                    symbol=PLOT_SYMBOLS[key],
                    size=9,
                    opacity=0.8,
                    # A surface ring keeps overlapping points readable.
                    line=dict(width=2, color=PLOT_SURFACE),
                ),
                hovertemplate=(
                    "Cell %{customdata}<br>" + x_title + " %{x:.4g}<br>"
                    + y_title + " %{y:.4g}<extra>" + name + "</extra>"
                ),
            )
        )
    figure.update_layout(title=title)
    return _style(figure, x_title, y_title, show_legend=len(figure.data) > 1)


def cluster_size_figure(cluster_df, title=""):
    """Cells per cluster. Discrete counts, so bars on integer ticks.

    Unresolved clusters are counted separately: their cell count is NA, so they
    cannot appear on a cells-per-cluster axis without inventing a value.
    """
    import plotly.graph_objects as go

    if cluster_df is None or not len(cluster_df):
        return _blank_figure("No clusters yet")

    known = cluster_df[cluster_df["cell_count"].notna()]
    counts = known["cell_count"].astype(int).value_counts().sort_index()
    unknown = int(len(cluster_df) - len(known))

    figure = go.Figure()
    if len(counts):
        figure.add_trace(
            go.Bar(
                x=[str(i) for i in counts.index],
                y=counts.to_numpy(),
                marker=dict(color=PLOT_COLOURS["clustered"], line=dict(width=0)),
                hovertemplate="%{y} clusters of %{x} cell(s)<extra></extra>",
                text=counts.to_numpy(),
                textposition="outside",
            )
        )
    if unknown:
        figure.add_trace(
            go.Bar(
                x=["NA"],
                y=[unknown],
                marker=dict(
                    color="#{:02x}{:02x}{:02x}".format(*CATEGORY_COLOURS["unresolved"]),
                    line=dict(width=0),
                ),
                hovertemplate="%{y} unresolved cluster(s), cell count NA<extra></extra>",
                text=[unknown],
                textposition="outside",
            )
        )
    figure.update_layout(title=title)
    return _style(figure, "Cells per cluster", "Cluster count")


def build_all_figures(cell_df, cluster_df, calibration):
    """The five figures §15 asks for, labelled in whatever units are in force."""
    from .calibration import area_label, length_label

    length = length_label(calibration)
    area = area_label(calibration)
    area_column = "area_um2" if calibration.is_calibrated else "area_px2"
    major_column = "major_axis_um" if calibration.is_calibrated else "major_axis_px"
    minor_column = "minor_axis_um" if calibration.is_calibrated else "minor_axis_px"

    return {
        "area_distribution": histogram_by_status(
            cell_df, area_column, "Cell area ({})".format(area), "Cell area distribution"
        ),
        "major_axis_distribution": histogram_by_status(
            cell_df, major_column, "Major axis ({})".format(length),
            "Major-axis length distribution",
        ),
        "area_vs_major": scatter_by_status(
            cell_df, major_column, area_column,
            "Major axis ({})".format(length), "Area ({})".format(area),
            "Area versus major-axis length",
        ),
        "length_vs_width": scatter_by_status(
            cell_df, major_column, minor_column,
            "Major axis ({})".format(length), "Minor axis ({})".format(length),
            "Length versus width",
        ),
        "cluster_size_distribution": cluster_size_figure(
            cluster_df, "Cluster-size distribution"
        ),
    }
