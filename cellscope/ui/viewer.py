"""The review viewer: layered overlay images composited by the browser.

The server renders three images per view -- the display image, a fill layer of
category tints, and a line layer with boundaries, dimming, IDs and the selected
cell -- plus a label map for hovering. The browser stacks them and handles zoom,
pan, overlay opacity and "show original" itself, so those gestures never reach
the server. A click is converted to display-pixel coordinates in the browser
and sent through a hidden textbox; the server maps it to a cell exactly as the
previous image-click handler did.

Rendered layers are cached per session, keyed on the results key and the view
options, so redrawing after navigation costs only a dictionary lookup.
"""

from __future__ import annotations

import base64
import io
import json
from html import escape

import numpy as np
from PIL import Image

from src.pipeline import compute_results, excluded_cell_measurements, results_key
from src.qc import exclusion_reasons
from src.visualize import _resize_labels, overlay_layers

from .views import display_base

#: Layer images kept per session (views of the same image with other options).
LAYER_CACHE_SIZE = 6


def _png(array: np.ndarray, mode: str | None = None) -> str:
    buffer = io.BytesIO()
    image = Image.fromarray(array) if mode is None else Image.fromarray(array).convert(mode)
    image.save(buffer, format="PNG", compress_level=1)
    return "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode("ascii")


def _jpeg(array: np.ndarray) -> str:
    buffer = io.BytesIO()
    Image.fromarray(np.ascontiguousarray(array)).save(buffer, format="JPEG", quality=92)
    return "data:image/jpeg;base64," + base64.b64encode(buffer.getvalue()).decode("ascii")


def _label_png(small: np.ndarray) -> str:
    """Object IDs packed into red/green (IDs up to 65535) for hover lookup."""
    ids = np.clip(small, 0, 65535).astype(np.uint32)
    rgb = np.zeros((*small.shape, 3), np.uint8)
    rgb[..., 0] = ids & 255
    rgb[..., 1] = (ids >> 8) & 255
    return _png(rgb)


def _cache(session, key, build):
    cache = session.view_cache.setdefault("viewer", {})
    if key in cache:
        value = cache.pop(key)
        cache[key] = value                       # most recently used last
        return value
    value = build()
    cache[key] = value
    while len(cache) > LAYER_CACHE_SIZE:
        cache.pop(next(iter(cache)))
    return value


def _object_info(session, results) -> dict[str, list]:
    """Per-object facts shown on hover: status, inclusion, area, reason."""
    reasons = exclusion_reasons(session.qc)
    cells = results.cells
    excluded = excluded_cell_measurements(session)
    area_column = "area_um2" if session.calibration.is_calibrated else "area_px2"
    unit = "µm²" if session.calibration.is_calibrated else "px²"
    areas = {}
    for frame in (cells, excluded):
        if len(frame) and area_column in frame:
            areas.update(dict(zip(frame["cell_id"].astype(int), frame[area_column].astype(float))))
    info = {}
    for obj in session.objects:
        included = session.qc.is_included(obj.object_id)
        area = areas.get(obj.object_id)
        info[str(obj.object_id)] = [
            "unresolved group" if not obj.is_resolved else "cell",
            included,
            "" if area is None else "{:.0f} {}".format(area, unit),
            "" if included else reasons.get(obj.object_id, ""),
        ]
    return info


MODE_HINTS = {
    "Toggle inclusion": "Click: exclude / restore",
    "Inspect": "Click: inspect",
    "Collect correction points": "Click: add correction point",
}


def viewer_html(session, show_ids: bool = True, show_clusters: bool = False,
                message: str = "") -> str:
    """The complete viewer for one image."""
    base = display_base(session)
    if base is None:
        return ('<div class="cs-viewer"><div class="cs-empty"><strong>No image</strong>'
                "Add images on the Import tab.</div></div>")
    height, width = base.shape[:2]
    base_uri = _cache(session, ("base", id(base)), lambda: _jpeg(base))

    fill_uri = lines_uri = labels_uri = ""
    info: dict = {}
    if session.has_segmentation:
        results = compute_results(session)
        rkey = repr(results_key(session))     # calibration params are a dict: not hashable
        selected = int(session.selected_object or 0)
        layer_key = ("layers", rkey, (height, width), bool(show_ids), bool(show_clusters),
                     selected, tuple(map(tuple, session.correction_points)))

        def build_layers():
            fill, lines = overlay_layers(
                (height, width), session.object_labels, results.objects, results.clusters,
                show_cell_ids=show_ids, show_cluster_ids=show_clusters,
                excluded_ids=session.qc.excluded_ids, selected_id=selected,
                annotation=session.annotation,
            )
            if session.correction_points:
                lines = _draw_points(lines, session, (height, width))
            return _png(fill), _png(lines)

        fill_uri, lines_uri = _cache(session, layer_key, build_layers)
        labels_uri = _cache(
            session, ("labels", session.segmentation_version, id(session.object_labels),
                      (height, width)),
            lambda: _label_png(_resize_labels(session.object_labels, (height, width))),
        )
        info = _cache(session, ("info", rkey), lambda: _object_info(session, results))
    elif session.correction_points:
        session.correction_points.clear()

    state = {
        "key": session.analysis_id,
        "w": width, "h": height,
        "sx": session.image.width / width if session.image else 1.0,
        "sy": session.image.height / height if session.image else 1.0,
        "mode": session.review_mode,
        "selected": int(session.selected_object or 0),
    }
    layers = '<img class="cs-base" alt="" src="{}">'.format(base_uri)
    if fill_uri:
        layers += '<img class="cs-fill" alt="" src="{}"><img class="cs-lines" alt="" src="{}">'.format(
            fill_uri, lines_uri)
    hidden = ""
    if labels_uri:
        hidden = '<img class="cs-labels" alt="" hidden src="{}" data-objects="{}">'.format(
            labels_uri, escape(json.dumps(info)))
    note = '<div class="cs-empty"><strong>Not segmented yet</strong>{}</div>'.format(
        escape(message)) if not session.has_segmentation and message else ""
    return (
        '<div class="cs-viewer" data-state="{state}">'
        '<div class="cs-mode">{mode}</div>'
        '<div class="cs-viewport" style="aspect-ratio:{w}/{h}">'
        '<div class="cs-stage" style="width:{w}px;height:{h}px">{layers}</div></div>'
        '<div class="cs-hover"></div>'
        '<div class="cs-toolbar" role="toolbar" aria-label="View controls">'
        '<button type="button" data-act="out" title="Zoom out (-)">−</button>'
        '<span class="cs-zoom">100%</span>'
        '<button type="button" data-act="in" title="Zoom in (+)">+</button>'
        '<button type="button" data-act="fit" title="Fit image (F)">Fit</button>'
        '<button type="button" data-act="actual" title="Actual pixels (1)">1:1</button>'
        '<span class="cs-sep">|</span>'
        '<label>Overlay <input type="range" min="0" max="80" step="2" data-act="alpha" '
        'aria-label="Overlay opacity"></label>'
        '<button type="button" data-act="original" aria-pressed="false" '
        'title="Show the original image (O)">Original</button>'
        '<span class="cs-spacer"></span>'
        '<span class="cs-help">Scroll to zoom · drag to pan · <kbd>?</kbd> shortcuts</span>'
        "</div>{hidden}{note}</div>"
    ).format(
        state=escape(json.dumps(state)), mode=escape(MODE_HINTS.get(session.review_mode, "")),
        w=width, h=height, layers=layers, hidden=hidden, note=note,
    )


def _draw_points(lines: np.ndarray, session, shape) -> np.ndarray:
    """Correction points drawn on a copy of the line layer."""
    from PIL import ImageDraw

    image = Image.fromarray(np.array(lines), "RGBA")
    draw = ImageDraw.Draw(image)
    scale_x = shape[1] / session.object_labels.shape[1]
    scale_y = shape[0] / session.object_labels.shape[0]
    points = [(x * scale_x, y * scale_y) for x, y in session.correction_points]
    if len(points) > 1:
        draw.line(points, fill=(250, 204, 21, 255), width=2)
    for index, (x, y) in enumerate(points, 1):
        draw.ellipse((x - 3, y - 3, x + 3, y + 3), fill=(250, 204, 21, 255))
        draw.text((x + 4, y + 4), str(index), fill=(250, 204, 21, 255),
                  stroke_width=1, stroke_fill=(0, 0, 0, 255))
    return np.asarray(image)


def parse_click(payload: str) -> tuple[int, int] | None:
    """Display-pixel coordinates from the viewer's click message."""
    try:
        data = json.loads(payload or "{}")
        return int(data["x"]), int(data["y"])
    except (ValueError, KeyError, TypeError):
        return None


VIEWER_JS = r"""
<script>
(() => {
  if (window.cellscopeViewer) return;
  const views = {};          // per-image zoom and pan, kept across re-renders
  let alpha = 0.28;
  try { const a = parseFloat(localStorage.getItem('cellscope-alpha')); if (!isNaN(a)) alpha = a; } catch (_) {}
  let original = false;

  const setBridge = (id, value) => {
    const root = document.getElementById(id);
    const field = root && root.querySelector('textarea, input');
    if (!field) return;
    field.value = value;
    field.dispatchEvent(new Event('input', { bubbles: true }));
  };

  function stateOf(viewer) {
    try { return JSON.parse(viewer.dataset.state || '{}'); } catch (_) { return {}; }
  }

  function apply(viewer) {
    const s = stateOf(viewer);
    const view = views[s.key];
    const port = viewer.querySelector('.cs-viewport');
    const stage = viewer.querySelector('.cs-stage');
    if (!view || !port || !stage) return;
    // A freshly rendered viewer has no layout yet; clamping now would lose the
    // pan position. The ResizeObserver calls back once it has a size.
    if (!port.clientWidth || !port.clientHeight) return;
    const fit = port.clientWidth / s.w;
    view.fit = fit;
    const scale = fit * view.zoom;
    const maxX = Math.max(0, s.w * scale - port.clientWidth);
    const maxY = Math.max(0, s.h * scale - port.clientHeight);
    view.x = Math.min(maxX, Math.max(0, view.x));
    view.y = Math.min(maxY, Math.max(0, view.y));
    stage.style.transform = `translate(${-view.x}px, ${-view.y}px) scale(${scale})`;
    stage.style.setProperty('--cs-alpha', alpha);
    stage.style.imageRendering = scale >= 1.5 ? 'pixelated' : 'auto';
    const label = viewer.querySelector('.cs-zoom');
    if (label) label.textContent = Math.round(scale * 100) + '%';
    const slider = viewer.querySelector('input[data-act=alpha]');
    if (slider) slider.value = Math.round(alpha * 100);
    viewer.classList.toggle('original', original);
    const button = viewer.querySelector('button[data-act=original]');
    if (button) button.setAttribute('aria-pressed', String(original));
  }

  function zoomAt(viewer, factor, cx, cy) {
    const s = stateOf(viewer);
    const view = views[s.key];
    if (!view) return;
    const port = viewer.querySelector('.cs-viewport');
    const rect = port.getBoundingClientRect();
    const px = cx === undefined ? rect.width / 2 : cx - rect.left;
    const py = cy === undefined ? rect.height / 2 : cy - rect.top;
    const before = view.fit * view.zoom;
    view.zoom = Math.min(32, Math.max(1, view.zoom * factor));
    const after = view.fit * view.zoom;
    view.x = (view.x + px) * after / before - px;
    view.y = (view.y + py) * after / before - py;
    apply(viewer);
  }

  function imagePoint(viewer, event) {
    const s = stateOf(viewer);
    const stage = viewer.querySelector('.cs-stage');
    const rect = stage.getBoundingClientRect();
    const x = Math.floor((event.clientX - rect.left) / rect.width * s.w);
    const y = Math.floor((event.clientY - rect.top) / rect.height * s.h);
    if (x < 0 || y < 0 || x >= s.w || y >= s.h) return null;
    return [x, y];
  }

  const labelCache = new WeakMap();
  function labelAt(viewer, point) {
    const img = viewer.querySelector('img.cs-labels');
    if (!img || !img.complete || !img.naturalWidth) return 0;
    let data = labelCache.get(img);
    if (!data) {
      try {
        const canvas = document.createElement('canvas');
        canvas.width = img.naturalWidth; canvas.height = img.naturalHeight;
        const context = canvas.getContext('2d', { willReadFrequently: true });
        context.drawImage(img, 0, 0);
        data = context.getImageData(0, 0, canvas.width, canvas.height);
        labelCache.set(img, data);
      } catch (_) { return 0; }
    }
    const i = (point[1] * data.width + point[0]) * 4;
    return data.data[i] + 256 * data.data[i + 1];
  }

  function objectsOf(viewer) {
    if (viewer._objects) return viewer._objects;
    const node = viewer.querySelector('img.cs-labels');
    try { viewer._objects = node ? JSON.parse(node.dataset.objects || '{}') : {}; } catch (_) { viewer._objects = {}; }
    return viewer._objects;
  }

  function hover(viewer, event) {
    const tip = viewer.querySelector('.cs-hover');
    if (!tip) return;
    const point = imagePoint(viewer, event);
    const id = point ? labelAt(viewer, point) : 0;
    const facts = id ? objectsOf(viewer)[String(id)] : null;
    if (!facts) { tip.style.display = 'none'; return; }
    const [kind, included, area, reason] = facts;
    tip.textContent = `#${id} · ${kind} · ${included ? 'included' : 'excluded'}` +
      (area ? ` · ${area}` : '') + (reason ? ` · ${reason}` : '');
    const box = viewer.getBoundingClientRect();
    tip.style.left = (event.clientX - box.left + 14) + 'px';
    tip.style.top = (event.clientY - box.top + 12) + 'px';
    tip.style.display = 'block';
  }

  function bind(viewer) {
    if (viewer._bound) { apply(viewer); return; }
    viewer._bound = true;
    const s = stateOf(viewer);
    if (!views[s.key]) views[s.key] = { zoom: 1, x: 0, y: 0, fit: 1 };
    const port = viewer.querySelector('.cs-viewport');
    let drag = null;
    port.addEventListener('wheel', (e) => {
      e.preventDefault();
      zoomAt(viewer, e.deltaY < 0 ? 1.25 : 0.8, e.clientX, e.clientY);
    }, { passive: false });
    port.addEventListener('pointerdown', (e) => {
      if (e.button !== 0) return;
      drag = { x: e.clientX, y: e.clientY, moved: false, id: e.pointerId };
      port.setPointerCapture(e.pointerId);
    });
    port.addEventListener('pointermove', (e) => {
      if (drag) {
        const dx = e.clientX - drag.x, dy = e.clientY - drag.y;
        if (!drag.moved && Math.hypot(dx, dy) < 4) return;
        drag.moved = true;
        viewer.classList.add('panning');
        const view = views[stateOf(viewer).key];
        view.x -= dx; view.y -= dy;
        drag.x = e.clientX; drag.y = e.clientY;
        apply(viewer);
        return;
      }
      hover(viewer, e);
    });
    port.addEventListener('pointerleave', () => {
      const tip = viewer.querySelector('.cs-hover'); if (tip) tip.style.display = 'none';
    });
    port.addEventListener('pointerup', (e) => {
      const wasDrag = drag && drag.moved;
      drag = null;
      viewer.classList.remove('panning');
      if (wasDrag) return;
      const point = imagePoint(viewer, e);
      if (!point) return;
      setBridge('cs-bridge', JSON.stringify({ x: point[0], y: point[1], t: Date.now() }));
    });
    viewer.querySelector('.cs-toolbar').addEventListener('click', (e) => {
      const act = e.target.closest('[data-act]')?.dataset.act;
      if (act === 'in') zoomAt(viewer, 1.5);
      else if (act === 'out') zoomAt(viewer, 1 / 1.5);
      else if (act === 'fit') { Object.assign(views[stateOf(viewer).key], { zoom: 1, x: 0, y: 0 }); apply(viewer); }
      else if (act === 'actual') { const v = views[stateOf(viewer).key]; zoomAt(viewer, (1 / v.fit) / v.zoom); }
      else if (act === 'original') { original = !original; refreshAll(); }
    });
    viewer.querySelector('input[data-act=alpha]').addEventListener('input', (e) => {
      alpha = e.target.value / 100;
      try { localStorage.setItem('cellscope-alpha', String(alpha)); } catch (_) {}
      refreshAll();
    });
    new ResizeObserver(() => apply(viewer)).observe(port);
    apply(viewer);
  }

  function refreshAll() { document.querySelectorAll('.cs-viewer').forEach(apply); }
  function scan() { document.querySelectorAll('.cs-viewer').forEach(bind); }
  new MutationObserver(scan).observe(document.body, { childList: true, subtree: true });
  scan();

  function visible(el) { return !!(el && el.getClientRects().length); }
  function press(id) {
    const el = document.getElementById(id);
    const button = el && (el.tagName === 'BUTTON' ? el : el.querySelector('button'));
    if (button && (visible(button) || el.classList.contains('cs-hidden-button'))) { button.click(); return true; }
    return false;
  }
  function currentViewer() {
    return [...document.querySelectorAll('.cs-viewer')].find(visible);
  }
  function setMode(value) {
    const radio = document.querySelector(`#cs-mode input[value="${value}"]`);
    if (radio) radio.click();
  }

  document.addEventListener('keydown', (e) => {
    const t = e.target;
    if (/INPUT|TEXTAREA|SELECT/.test(t.tagName) || t.isContentEditable) return;
    const viewer = currentViewer();
    if (!viewer) return;
    const key = e.key;
    const mod = e.ctrlKey || e.metaKey;
    let handled = true;
    if (mod && key.toLowerCase() === 'z') handled = press(e.shiftKey ? 'cs-redo' : 'cs-undo');
    else if (mod && key.toLowerCase() === 'y') handled = press('cs-redo');
    else if (mod) handled = false;
    else if (key === 'ArrowLeft') handled = press('cs-prev');
    else if (key === 'ArrowRight') handled = press('cs-next');
    else if (key === 'x' || key === 'X') handled = press('cs-toggle-selected');
    else if (key === 'i' || key === 'I') setMode('Inspect');
    else if (key === 't' || key === 'T') setMode('Toggle inclusion');
    else if (key === 'p' || key === 'P') setMode('Collect correction points');
    else if (key === 'r' || key === 'R') handled = press('cs-mark-reviewed');
    else if (key === 'f' || key === 'F') { const v = views[stateOf(viewer).key]; Object.assign(v, { zoom: 1, x: 0, y: 0 }); apply(viewer); }
    else if (key === '+' || key === '=') zoomAt(viewer, 1.5);
    else if (key === '-' || key === '_') zoomAt(viewer, 1 / 1.5);
    else if (key === '1') { const v = views[stateOf(viewer).key]; zoomAt(viewer, (1 / v.fit) / v.zoom); }
    else if (key === 'o' || key === 'O') { original = !original; refreshAll(); }
    else if (key === '?') { const help = document.getElementById('cs-shortcuts'); if (help) help.querySelector('button, .label-wrap')?.click(); }
    else handled = false;
    if (handled) e.preventDefault();
  });

  window.cellscopeViewer = { views, apply: refreshAll, setBridge };
})();
</script>
"""

SHORTCUTS_MD = """
| Key | Action |
|---|---|
| <kbd>←</kbd> / <kbd>→</kbd> | Previous / next image |
| <kbd>T</kbd> · <kbd>I</kbd> · <kbd>P</kbd> | Click mode: toggle inclusion · inspect · correction points |
| <kbd>X</kbd> | Exclude or restore the selected cell |
| <kbd>R</kbd> | Mark this image reviewed |
| <kbd>Ctrl</kbd>+<kbd>Z</kbd> / <kbd>Ctrl</kbd>+<kbd>Shift</kbd>+<kbd>Z</kbd> | Undo / redo |
| <kbd>+</kbd> / <kbd>-</kbd> · scroll | Zoom |
| <kbd>F</kbd> · <kbd>1</kbd> | Fit image · actual pixels |
| drag | Pan |
| <kbd>O</kbd> | Show the original image |
| <kbd>?</kbd> | Show or hide this list |
"""
