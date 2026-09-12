"""Presentation layer for the CellScope interface.

Kept out of ``app.py`` so the callback logic there stays readable, and out of
``src/`` because building a Gradio theme requires importing Gradio, which the
scientific package deliberately never does.

Colour decisions, measured rather than eyeballed
------------------------------------------------
Forest green is the *interface* accent only. The data palette in
``src/visualize.py`` is blue and orange with no green in it, so chrome never
competes with a measurement, and that pair validates at CVD dE 23.2 on white.

Contrast ratios against white:

    FOREST 9.11   FOREST_DARK 11.47   INK 18.72   MUTED 7.56   WARN 7.09
"""

from __future__ import annotations

import gradio as gr

from src.visualize import overlay_legend

FOREST = "#14532D"
FOREST_DARK = "#0F4229"
FOREST_TINT = "#E9F5ED"

INK = "#0B1220"
MUTED = "#4B5563"
LINE = "#C8D0CB"
WARN = "#92400E"
OK = "#15803D"
SURFACE = "#FFFFFF"
SURFACE_2 = "#F4F7F5"

#: Whole-interface scale. 0.9 shows about 10% more content per screen.
PAGE_ZOOM = 0.9

#: Content width before the zoom factor, and the gutter either side. At 0.9 this
#: yields ~2000 effective px, filling a 1920-wide screen with only the gutters
#: left over -- measurement tables are wide, and margin spent on empty page is
#: margin taken from the data.
CONTENT_WIDTH = 1800
SIDE_GUTTER = "6px"

SANS = "Inter, 'Helvetica Neue', Helvetica, Arial, sans-serif"
MONO = "'JetBrains Mono', Consolas, 'Courier New', monospace"

CSS = """
.gradio-container {
  font-family: %(sans)s;
  zoom: %(zoom)s;
  max-width: %(maxw)spx !important;
  margin-left: auto !important;
  margin-right: auto !important;
  padding-left: %(gutter)s !important;
  padding-right: %(gutter)s !important;
}
.gradio-container > .main,
.gradio-container .contain { padding-left: 0 !important; padding-right: 0 !important; }
@supports not (zoom: 1) {   /* older Firefox: fall back to type scale alone */
  .gradio-container { font-size: %(zoompct)s; }
}

#cs-header { display:flex; align-items:center; gap:.6rem; margin-bottom:.5rem;
  padding-bottom:.5rem; border-bottom:2px solid %(forest)s; flex-wrap:nowrap; }
#cs-header .cs-sub { font-size:.78rem; color:%(muted)s; }

/* The title is a button (it starts a new batch) but must still read as a
   heading: no chrome until hovered, when it picks up an underline to show it
   is clickable. */
#cs-home { font-size:1.3rem !important; font-weight:650 !important;
  letter-spacing:-.01em; color:%(forest_dark)s !important;
  background:none !important; border:none !important; box-shadow:none !important;
  padding:0 !important; margin:0 !important; min-width:0 !important;
  width:auto !important; flex:0 0 auto; cursor:pointer; }
#cs-home:hover { text-decoration:underline; text-underline-offset:3px; }

.cs-status { display:flex; flex-wrap:wrap; border:1px solid %(line)s; border-radius:8px;
  overflow:hidden; background:%(surface2)s; margin-bottom:.35rem; }
.cs-stat { flex:1 1 0; min-width:130px; padding:.45rem .7rem; border-right:1px solid %(line)s; }
.cs-stat:last-child { border-right:0; }
.cs-k { display:block; font-size:.62rem; text-transform:uppercase; letter-spacing:.07em;
  color:%(muted)s; margin-bottom:.1rem; }
.cs-v { display:block; font-size:.83rem; font-weight:600; color:%(ink)s;
  overflow:hidden; text-overflow:ellipsis; white-space:nowrap; }
.cs-stale .cs-v { color:%(warn)s; }
.cs-stale .cs-v::before { content:"⚠  "; }

/* The image navigator: which image the tab is acting on, never ambiguous. */
.cs-nav { display:flex; align-items:center; gap:.55rem; flex-wrap:wrap;
  border:1px solid %(line)s; border-left:3px solid %(forest)s; border-radius:6px;
  background:%(surface2)s; padding:.4rem .7rem; font-size:.82rem; color:%(ink)s; }
.cs-nav .cs-pos { font-variant-numeric:tabular-nums; color:%(muted)s; font-size:.75rem; }
.cs-nav .cs-name { font-weight:650; }
.cs-nav .cs-sep { color:%(line)s; }

.cs-badge { display:inline-flex; align-items:center; gap:.3rem; font-size:.7rem;
  padding:.1rem .45rem; border-radius:999px; border:1px solid %(line)s;
  background:%(surface)s; color:%(muted)s; white-space:nowrap; }
.cs-badge.ok { color:%(ok)s; border-color:%(ok)s; background:%(tint)s; }
.cs-badge.warn { color:%(warn)s; border-color:%(warn)s; }

/* Review strip: every image's state at a glance. */
.cs-strip { display:flex; flex-wrap:wrap; gap:.3rem; margin:.35rem 0; }
.cs-tile { display:inline-flex; align-items:center; gap:.35rem; font-size:.72rem;
  padding:.22rem .5rem; border-radius:5px; border:1px solid %(line)s;
  background:%(surface)s; color:%(ink)s; }
.cs-tile.done { background:%(tint)s; border-color:%(ok)s; }
.cs-tile.active { outline:2px solid %(forest)s; outline-offset:1px; }
.cs-tile.err { border-color:%(warn)s; color:%(warn)s; }
.cs-tile .n { color:%(muted)s; font-variant-numeric:tabular-nums; }

#cs-results { margin-left:-%(gutter)s; margin-right:-%(gutter)s; }
#cs-results > .tabitem,
#cs-results .form,
#cs-results .block { padding-left:2px !important; padding-right:2px !important; }
#cs-results .table-wrap { border-radius:6px; }

.gradio-container .table-wrap, .gradio-container .svelte-virtual-table-viewport {
  max-width: 100%% !important;
}
.gradio-container tbody td, .gradio-container thead th { white-space: nowrap; }

.cs-legend { display:flex; flex-wrap:wrap; gap:.4rem; margin:.45rem 0 .15rem; }
.cs-chip { display:inline-flex; align-items:center; gap:.4rem; font-size:.74rem; color:%(ink)s;
  padding:.18rem .55rem; border-radius:999px; border:1px solid %(line)s; background:%(surface)s; }
.cs-swatch { width:.72rem; height:.72rem; border-radius:3px; flex:none;
  box-shadow:inset 0 0 0 1px rgba(0,0,0,.25); }
.cs-swatch.thick { outline:2px solid currentColor; outline-offset:1px; }

/* Gradio renders dataframe cells in the mono font via var(--font-mono), which a
   hashed Svelte class sets. The theme repoints that variable; these rules cover
   markdown tables, whose own rule is a double class and needs !important.
   tabular-nums keeps a numeric column aligned, the one thing mono was buying. */
.gradio-container table,
.gradio-container th,
.gradio-container td,
.gradio-container thead th,
.gradio-container tbody td,
.gradio-container .cell-wrap,
.gradio-container .cell-wrap span,
.gradio-container .svelte-virtual-table-viewport span {
  font-family: %(sans)s !important;
  font-variant-numeric: tabular-nums;
  font-size: .8rem;
}
.gradio-container thead th {
  font-weight: 600; color: %(ink)s; background: %(surface2)s;
  text-transform: none; letter-spacing: 0;
}
.gradio-container tbody td { color: %(ink)s; }

/* Real monospace, restored only where it means something. */
.gradio-container code, .gradio-container pre, .gradio-container kbd {
  font-family: %(mono)s !important;
}

.cs-note { font-size:.79rem; color:%(muted)s; line-height:1.5; max-width:78ch; }
.cs-note strong, .cs-note code { color:%(ink)s; }
.cs-callout { border-left:3px solid %(forest)s; padding:.5rem .8rem; background:%(tint)s;
  border-radius:0 6px 6px 0; font-size:.8rem; color:%(ink)s; }
.cs-callout p { margin:0; }
.cs-warnbox { border-left:3px solid %(warn)s; padding:.5rem .8rem; background:#FEF6EC;
  border-radius:0 6px 6px 0; font-size:.8rem; color:%(ink)s; }
.cs-warnbox p { margin:0; }

.cs-empty { border:1px dashed %(line)s; border-radius:8px; padding:1.1rem;
  text-align:center; color:%(muted)s; font-size:.83rem; background:%(surface2)s; }
.cs-empty strong { display:block; color:%(ink)s; font-size:.9rem; margin-bottom:.2rem; }

.cs-h { font-size:.7rem; text-transform:uppercase; letter-spacing:.07em;
  color:%(muted)s; font-weight:650; margin:.55rem 0 .1rem; }

.tabs button.selected { color:%(forest_dark)s !important; font-weight:600; }
footer { display:none !important; }
""" % {  # noqa: UP031 - CSS braces make str.format unreadable
    "zoom": PAGE_ZOOM,
    "zoompct": "{:.0f}%".format(PAGE_ZOOM * 100),
    "maxw": int(round(CONTENT_WIDTH / PAGE_ZOOM)),
    "gutter": SIDE_GUTTER,
    "sans": SANS, "mono": MONO, "forest": FOREST, "forest_dark": FOREST_DARK,
    "tint": FOREST_TINT, "ink": INK, "muted": MUTED, "line": LINE, "warn": WARN,
    "ok": OK, "surface": SURFACE, "surface2": SURFACE_2,
}


# Custom HTML uses the same semantic colours as Gradio's native components.
_LIGHT_TOKENS = {
    FOREST: "accent", FOREST_DARK: "accent-text", FOREST_TINT: "tint",
    INK: "text", MUTED: "muted", LINE: "line", WARN: "warn", OK: "ok",
    SURFACE: "surface", SURFACE_2: "surface-alt", "#FEF6EC": "warn-bg",
}
for _colour, _token in _LIGHT_TOKENS.items():
    CSS = CSS.replace(_colour, "var(--cs-" + _token + ")")
CSS = """
:root { --cs-accent:#14532D; --cs-accent-text:#0F4229; --cs-tint:#E9F5ED;
 --cs-text:#0B1220; --cs-muted:#4B5563; --cs-line:#C8D0CB; --cs-warn:#92400E;
 --cs-ok:#15803D; --cs-surface:#FFFFFF; --cs-surface-alt:#F4F7F5; --cs-warn-bg:#FEF6EC; }
.dark { color-scheme:dark; --cs-accent:#86EFAC; --cs-accent-text:#86EFAC;
 --cs-tint:#16382A; --cs-text:#F1F5F9; --cs-muted:#CBD5E1; --cs-line:#475569;
 --cs-warn:#FCD34D; --cs-ok:#86EFAC; --cs-surface:#111827;
 --cs-surface-alt:#1F2937; --cs-warn-bg:#422D18; }
#cs-review-image { cursor:crosshair; }
""" + CSS

THEME_JS = """() => {
 const media = window.matchMedia('(prefers-color-scheme: dark)');
 const apply = (mode) => {
   const dark = mode === 'Dark' || (mode === 'System' && media.matches);
   document.documentElement.classList.toggle('dark', dark);
   document.body.classList.toggle('dark', dark);
   document.querySelectorAll('.gradio-container').forEach(el => el.classList.toggle('dark', dark));
 };
 let saved = 'System';
 try { saved = localStorage.getItem('cellscope-theme') || 'System'; } catch (_) {}
 window.cellscopeTheme = (mode) => {
   saved = mode;
   try { localStorage.setItem('cellscope-theme', mode); } catch (_) {}
   apply(mode);
 };
 media.addEventListener('change', () => apply(saved));
 apply(saved);
 return saved;
}"""


def build_theme():
    """Accessible light and dark surfaces with forest-green accents."""
    return gr.themes.Soft(
        primary_hue=gr.themes.colors.green,
        neutral_hue=gr.themes.colors.gray,
        font=[gr.themes.GoogleFont("Inter"), "Helvetica Neue", "Arial", "sans-serif"],
        # Dataframe cells read var(--font-mono) from a hashed, scoped Svelte
        # class that changes between releases, so overriding it by selector is
        # brittle. Setting the variable is what actually works; the CSS above
        # hands real monospace back to code and pre.
        font_mono=[
            gr.themes.GoogleFont("Inter"), "Helvetica Neue", "Arial", "sans-serif",
        ],
    ).set(
        body_background_fill=SURFACE,
        body_background_fill_dark="#111827",
        background_fill_primary=SURFACE,
        background_fill_primary_dark="#111827",
        background_fill_secondary=SURFACE_2,
        background_fill_secondary_dark="#1F2937",
        block_background_fill=SURFACE,
        block_background_fill_dark="#111827",
        border_color_primary=LINE,
        border_color_primary_dark="#475569",
        body_text_color=INK,
        body_text_color_dark="#F1F5F9",
        body_text_color_subdued=MUTED,
        body_text_color_subdued_dark="#CBD5E1",
        block_label_text_color=MUTED,
        block_label_text_color_dark="#CBD5E1",
        block_title_text_color=INK,
        block_title_text_color_dark="#F1F5F9",
        button_primary_background_fill=FOREST,
        button_primary_background_fill_dark="#166534",
        button_primary_background_fill_hover=FOREST_DARK,
        button_primary_background_fill_hover_dark="#15803D",
        button_primary_text_color="#FFFFFF",
        button_primary_text_color_dark="#FFFFFF",
        button_secondary_background_fill=SURFACE_2,
        button_secondary_background_fill_dark="#1F2937",
        checkbox_background_color_selected=FOREST,
        checkbox_background_color_selected_dark="#166534",
        slider_color=FOREST,
        slider_color_dark="#166534",
    )


def legend_html() -> str:
    """Colour key for the overlays, as chips rather than a run of glyphs."""
    chips = []
    for (colour, label), key in zip(
        overlay_legend(), ("isolated", "clustered", "unresolved")
    ):
        thick = " thick" if key == "unresolved" else ""
        chips.append(
            '<span class="cs-chip"><span class="cs-swatch{}" style="background:{};color:{}">'
            "</span>{}</span>".format(thick, colour, colour, label)
        )
    return '<div class="cs-legend">{}</div>'.format("".join(chips))


def empty_state(title: str, hint: str) -> str:
    """A placeholder that says what to do next rather than showing nothing."""
    return '<div class="cs-empty"><strong>{}</strong>{}</div>'.format(title, hint)


def callout(text: str, tone: str = "info") -> str:
    css_class = "cs-warnbox" if tone == "warn" else "cs-callout"
    return '<div class="{}"><p>{}</p></div>'.format(css_class, text)


def section(title: str) -> str:
    return '<div class="cs-h">{}</div>'.format(title)
