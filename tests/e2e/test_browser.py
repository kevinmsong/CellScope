"""End-to-end test: the real app, driven in a headless browser.

Starts ``python -m cellscope`` on a free port and walks the whole workflow on
synthetic fields (two with a burned-in scale bar):

    import -> calibrate from the burned-in bar -> preview and keep -> batch run
    -> review (inspect, zoom, pan, opacity, exclude, undo, shortcuts, split)
    -> design -> export -> save and reopen the project -> dark theme

Skipped unless Playwright and its Chromium are installed:

    python -m pip install playwright && python -m playwright install chromium
    python -m pytest tests/e2e -q
"""

from __future__ import annotations

import json
import os
import socket
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

import pytest

sync_api = pytest.importorskip("playwright.sync_api")

ROOT = Path(__file__).resolve().parents[2]
pytestmark = pytest.mark.e2e

# Label-map helpers evaluated in the page: the viewer encodes object IDs in
# the red and green channels of a hidden image.
FIND_CELL = """() => {
  const img = document.querySelector('img.cs-labels');
  if (!img || !img.naturalWidth) return null;
  const objects = JSON.parse(img.dataset.objects);
  const c = document.createElement('canvas'); c.width = img.naturalWidth; c.height = img.naturalHeight;
  const g = c.getContext('2d'); g.drawImage(img, 0, 0);
  const d = g.getImageData(0, 0, c.width, c.height).data;
  const s = {};
  for (let i = 0; i < d.length; i += 4) {
    const id = d[i] + 256 * d[i + 1];
    if (!id || !objects[id] || !objects[id][1] || objects[id][0] !== 'cell') continue;
    const p = i / 4, x = p % c.width, y = Math.floor(p / c.width);
    const e = s[id] || (s[id] = {n: 0, sx: 0, sy: 0, minx: 1e9, maxx: -1});
    e.n++; e.sx += x; e.sy += y; e.minx = Math.min(e.minx, x); e.maxx = Math.max(e.maxx, x);
  }
  let best = null;
  for (const [id, e] of Object.entries(s)) if (!best || e.n > best.n) best = {id: +id, ...e};
  return best && {id: best.id, n: best.n, cx: best.sx / best.n, cy: best.sy / best.n,
                  minx: best.minx, maxx: best.maxx};
}"""
ROW_INSIDE = """([id, x]) => {
  const img = document.querySelector('img.cs-labels');
  const c = document.createElement('canvas'); c.width = img.naturalWidth; c.height = img.naturalHeight;
  const g = c.getContext('2d'); g.drawImage(img, 0, 0);
  const d = g.getImageData(Math.round(x), 0, 1, c.height).data;
  const rows = []; for (let y = 0; y < c.height; y++) if (d[y*4] + 256*d[y*4+1] === id) rows.push(y);
  return rows.length ? rows[Math.floor(rows.length / 2)] : null;
}"""


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


@pytest.fixture(scope="module")
def browser():
    with sync_api.sync_playwright() as p:
        try:
            instance = p.chromium.launch()
        except Exception as error:          # Chromium not downloaded
            pytest.skip("Chromium is not installed: {}".format(error))
        yield instance
        instance.close()


@pytest.fixture(scope="module")
def images(tmp_path_factory):
    from PIL import Image

    from src.perf import synthetic_field

    folder = tmp_path_factory.mktemp("fields")
    paths = []
    for index, ruler in enumerate((True, True, False)):
        rgb = synthetic_field(800, 800, n_cells=14, seed=40 + index, ruler=ruler,
                              touching_fraction=0.2)
        path = folder / "field_{}.png".format(index + 1)
        Image.fromarray(rgb).save(path)
        paths.append(str(path))
    return paths


@pytest.fixture(scope="module")
def server(tmp_path_factory):
    projects = tmp_path_factory.mktemp("projects")
    log_path = projects / "server.log"
    port = _free_port()
    env = dict(os.environ, PYTHONUNBUFFERED="1")
    url = "http://127.0.0.1:{}".format(port)
    with open(log_path, "w", encoding="utf-8") as log:
        process = subprocess.Popen(
            [sys.executable, "-m", "cellscope", "--port", str(port), "--projects", str(projects),
             "--no-preload", "--cpu", "--autosave-interval", "5"],
            cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT,
        )
        try:
            deadline = time.time() + 120
            while True:
                if process.poll() is not None:
                    pytest.fail("server exited: " + log_path.read_text(encoding="utf-8"))
                if time.time() > deadline:
                    pytest.fail("server did not start")
                try:
                    urllib.request.urlopen(url, timeout=2)
                    break
                except OSError:
                    time.sleep(0.5)
            yield {"url": url, "projects": projects, "log": log_path}
        finally:
            process.kill()
            process.wait()


def _idle(page, pause=0.4):
    time.sleep(pause)
    deadline = time.time() + 180
    while time.time() < deadline and page.locator(".progress-text, .eta-bar, .generating").count():
        time.sleep(0.25)


def _wait_text(page, text, timeout=180_000):
    # innerText applies CSS text-transform, so compare case-insensitively.
    page.wait_for_function("t => document.body.innerText.toLowerCase().includes(t.toLowerCase())",
                           arg=text, timeout=timeout)
    _idle(page)


def _tab(page, name):
    page.get_by_role("tab", name=name, exact=True).click()
    _idle(page)


def _status(page):
    return page.locator("#cs-review-status").inner_text()


def _mode(page, key, label):
    """Switch the click mode with its shortcut and wait for the viewer to show it."""
    page.keyboard.press(key)
    page.wait_for_function(
        "l => (document.querySelector('.cs-viewer .cs-mode') || {}).textContent === l",
        arg=label, timeout=30_000)
    time.sleep(0.3)


def _wait_in(page, selector, text, timeout=30_000):
    try:
        page.wait_for_function(
            "([s, t]) => (document.querySelector(s) || {innerText: ''}).innerText.includes(t)",
            arg=[selector, text], timeout=timeout)
    except sync_api.TimeoutError:
        shown = {s: page.locator(s).inner_text() for s in ("#cs-review-status", "#cs-cell-detail")}
        pytest.fail("{!r} never appeared in {}; showing {}".format(text, selector, shown))


def test_full_workflow(browser, server, images):
    page = browser.new_page(viewport={"width": 1600, "height": 1500})
    errors = []
    page.on("pageerror", lambda e: errors.append(str(e)))
    page.goto(server["url"])
    page.wait_for_selector("#cs-home", timeout=60_000)
    _idle(page, 1)
    assert "Add images" in page.locator(".cs-next").first.inner_text()

    # ---- import and calibrate from the burned-in bar ----------------------
    page.locator("#cs-upload input[type=file]").set_input_files(images)
    _wait_text(page, "Added 3 image(s)")
    assert "2 image(s) carry a burned-in scale bar" in page.inner_text("body")
    page.get_by_label("Experiment name").fill("E2E plate")
    page.get_by_label("Experiment name").press("Tab")
    page.get_by_role("radio", name="red", exact=True).first.check()
    _idle(page)
    page.get_by_role("button", name="Detect bar in this image").click()
    _wait_text(page, "Detected a yellow bar 126 px")
    page.get_by_label("Bar represents (µm)").fill("50")
    page.get_by_role("button", name="Apply calibration").first.click()
    _wait_text(page, "Whole batch calibrated")
    assert "0.396825" in page.inner_text("body")

    # ---- segment: preview, keep, run the rest -----------------------------
    _tab(page, "Segment")
    page.get_by_role("radio", name="threshold_watershed").check()
    page.get_by_role("button", name="Preview on this image").click()
    _wait_text(page, "Nothing has changed yet")
    assert page.get_by_text("Objects detected").count()
    page.get_by_role("button", name="Keep preview").click()
    _wait_text(page, "Kept the preview")
    page.get_by_role("button", name="Run batch").click()
    _wait_text(page, "Finished:", timeout=300_000)
    assert "Finished: 2 of 2 image(s) segmented" in page.inner_text("body")
    assert "Review field_1.png (3 left)" in page.locator(".cs-next").first.inner_text()

    # ---- review ------------------------------------------------------------
    _tab(page, "Review")
    page.wait_for_selector(".cs-viewer img.cs-labels", state="attached", timeout=60_000)
    _idle(page, 1)
    # The active image is the last one run; go to the first (ruler) field.
    while "1 / 3" not in page.locator(".cs-nav").nth(1).inner_text():
        page.get_by_role("button", name="◀ Previous").last.click()
        _idle(page, 0.8)
    objects = json.loads(page.locator("img.cs-labels").get_attribute("data-objects"))
    ruler_reasons = [v[3] for v in objects.values() if "scale bar" in v[3]]
    assert ruler_reasons, "objects touching the burned-in bar are excluded with their reason"

    state = json.loads(page.locator(".cs-viewer").get_attribute("data-state"))
    target = page.evaluate(FIND_CELL)
    assert target, "an included cell to click"
    # A point certainly inside the cell (its centroid may not be).
    target["cx"] = (target["minx"] + target["maxx"]) / 2
    target["cy"] = page.evaluate(ROW_INSIDE, [target["id"], target["cx"]])
    assert target["cy"] is not None

    def click(x, y):
        page.locator(".cs-viewport").scroll_into_view_if_needed()
        box = page.locator(".cs-stage").bounding_box()
        page.mouse.click(box["x"] + (x + .5) * box["width"] / state["w"],
                         box["y"] + (y + .5) * box["height"] / state["h"])
        _idle(page, 0.8)

    page.locator("body").click(position={"x": 3, "y": 3})
    _mode(page, "i", "Click: inspect")
    click(target["cx"], target["cy"])
    _wait_in(page, "#cs-cell-detail", "Cell {}".format(target["id"]))

    # Zoom, pan and opacity are handled in the browser.
    page.locator(".cs-viewport").scroll_into_view_if_needed()
    port = page.locator(".cs-viewport").bounding_box()
    px = port["x"] + (target["cx"] + .5) * port["width"] / state["w"]
    py = port["y"] + (target["cy"] + .5) * port["width"] / state["w"]
    page.mouse.move(px, py)
    for _ in range(4):
        page.mouse.wheel(0, -120)
    time.sleep(0.3)
    zoom = int(page.locator(".cs-zoom").inner_text().rstrip("%"))
    assert zoom > 150
    box = page.locator(".cs-stage").bounding_box()
    under = box["x"] + (target["cx"] + .5) * box["width"] / state["w"]
    assert abs(under - px) < 3, "wheel zoom keeps the cell under the cursor"
    _mode(page, "t", "Click: exclude / restore")
    requests = []
    page.on("request", lambda r: requests.append(r.url) if "queue/join" in r.url else None)
    page.mouse.move(px, py)
    page.mouse.down()
    # Drag towards the centre, so the cell stays inside the view.
    page.mouse.move(px + (-40 if px > port["x"] + port["width"] / 2 else 40),
                    py + (-20 if py > port["y"] + port["height"] / 2 else 20), steps=4)
    page.mouse.up()
    page.locator(".cs-toolbar input[type=range]").fill("60")
    time.sleep(0.5)
    assert requests == [], "pan and opacity must not reach the server"

    click(target["cx"], target["cy"])      # toggle mode, zoomed and panned
    _wait_in(page, "#cs-review-status", "Excluded cell {}".format(target["id"]))
    assert int(page.locator(".cs-zoom").inner_text().rstrip("%")) == zoom, "zoom survives re-render"
    page.keyboard.press("Control+z")
    _wait_in(page, "#cs-review-status", "Undone.")
    page.keyboard.press("x")
    _wait_in(page, "#cs-review-status", "Excluded cell {}".format(target["id"]))
    page.keyboard.press("Control+z")
    _wait_in(page, "#cs-review-status", "Undone.")
    page.keyboard.press("Control+Shift+z")
    _wait_in(page, "#cs-review-status", "Redone.")
    page.keyboard.press("Control+z")
    _wait_in(page, "#cs-review-status", "Undone.")
    page.keyboard.press("f")
    assert page.locator(".cs-zoom").inner_text() != "{}%".format(zoom)

    # Split the largest cell with two seeds, then undo.
    _mode(page, "p", "Click: add correction point")
    state = json.loads(page.locator(".cs-viewer").get_attribute("data-state"))
    left = target["minx"] + (target["maxx"] - target["minx"]) * 0.2
    right = target["minx"] + (target["maxx"] - target["minx"]) * 0.8
    click(left, page.evaluate(ROW_INSIDE, [target["id"], left]))
    _wait_in(page, "#cs-review-status", "Point 1")
    click(right, page.evaluate(ROW_INSIDE, [target["id"], right]))
    _wait_in(page, "#cs-review-status", "Point 2")
    assert len(json.loads(page.get_by_label("Points (original pixels)").input_value())) == 2
    before = _status(page).split("\n")[0]
    page.get_by_text("Correct segmentation", exact=True).click()
    page.get_by_label("Cell IDs", exact=True).last.fill(str(target["id"]))
    page.get_by_role("button", name="Apply correction").click()
    _wait_text(page, "Split applied")
    page.keyboard.press("Control+z")
    _idle(page)
    assert _status(page).split("\n")[0] == before

    page.keyboard.press("r")
    _wait_text(page, "Marked field_1.png reviewed")
    page.keyboard.press("ArrowRight")
    page.wait_for_function(
        "() => [...document.querySelectorAll('.cs-nav')].some(n => n.innerText.includes('2 / 3'))",
        timeout=30_000)

    # ---- design, results, export -------------------------------------------
    _tab(page, "Design & compare")
    _wait_text(page, "Condition n is the number of biological replicates", timeout=30_000)
    _tab(page, "Results")
    _wait_text(page, "Cells pooled", timeout=30_000)
    _tab(page, "Export")
    page.get_by_role("button", name="Build analysis archive").click()
    _wait_text(page, "Wrote batch_analysis.zip")

    # ---- save, then reopen the saved project -------------------------------
    _tab(page, "Projects")
    page.get_by_role("button", name="Save project").click()
    _wait_text(page, "Project saved")
    saved = sorted(server["projects"].glob("*.cellscope"))
    assert saved, "a recovery copy exists"
    page.locator("input[type=file][accept*='.cellscope']").set_input_files(str(saved[0]))
    _idle(page, 1.5)
    page.get_by_role("button", name="Open this project").click()
    _wait_text(page, "Opened E2E plate (3 image(s))")
    assert "1 / 3" in page.locator(".cs-status").first.inner_text()      # one image reviewed

    # ---- dark theme ----------------------------------------------------------
    page.get_by_role("radio", name="Dark").check()
    time.sleep(0.5)
    assert page.evaluate("document.documentElement.classList.contains('dark')")
    page.get_by_role("radio", name="Light").check()

    assert not errors, errors
    log = server["log"].read_text(encoding="utf-8")
    assert "Traceback" not in log, log[-3000:]
    page.close()
