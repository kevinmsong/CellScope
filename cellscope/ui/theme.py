"""Presentation layer: theme, stylesheet, and small HTML building blocks.

Kept out of ``src/`` because building a Gradio theme requires importing Gradio,
which the scientific package deliberately never does.

Colour is used semantically and never alone: every state also carries a word
and a symbol.

    neutral  pending, not started       ○
    positive done, reviewed             ✓
    warning  action needed, stale       !
    error    failed, cannot pool        ✕

Forest green is the *interface* accent only. The data palette in
``src/visualize.py`` is blue, orange and aqua with no green in it, so chrome
never competes with a measurement. Contrast against white: accent 9.11,
text 18.72, muted 7.56, warning 7.09, error 6.47.
"""

from __future__ import annotations

from html import escape

import gradio as gr

from src.visualize import overlay_legend

FOREST = "#14532D"
FOREST_DARK = "#0F4229"

#: Whole-interface scale. 0.9 shows about 10% more content per screen.
PAGE_ZOOM = 0.9
CONTENT_WIDTH = 1800
SANS = "Inter, 'Helvetica Neue', Helvetica, Arial, sans-serif"
MONO = "'JetBrains Mono', Consolas, 'Courier New', monospace"

TOKENS_LIGHT = {
    "accent": "#14532D", "accent-text": "#0F4229", "accent-soft": "#E9F5ED",
    "text": "#0B1220", "muted": "#4B5563", "line": "#D5DBD7", "line-strong": "#AEB8B2",
    "surface": "#FFFFFF", "surface-alt": "#F5F7F6", "surface-sunken": "#EEF1EF",
    "ok": "#15803D", "ok-bg": "#E9F5ED",
    "warn": "#92400E", "warn-bg": "#FEF6EC",
    "err": "#B42318", "err-bg": "#FDECEA",
    "neutral": "#4B5563", "neutral-bg": "#F1F3F2",
    "viewer-bg": "#0B0F0D",
}
TOKENS_DARK = {
    "accent": "#86EFAC", "accent-text": "#86EFAC", "accent-soft": "#16382A",
    "text": "#F1F5F9", "muted": "#CBD5E1", "line": "#3A4452", "line-strong": "#566173",
    "surface": "#111827", "surface-alt": "#1B2331", "surface-sunken": "#0D131E",
    "ok": "#86EFAC", "ok-bg": "#16382A",
    "warn": "#FCD34D", "warn-bg": "#3B2A12",
    "err": "#FCA5A5", "err-bg": "#3F1D1D",
    "neutral": "#CBD5E1", "neutral-bg": "#1F2937",
    "viewer-bg": "#05070A",
}


def _tokens(values: dict[str, str]) -> str:
    return " ".join("--cs-{}:{};".format(k, v) for k, v in values.items())


CSS = """
:root { %(light)s }
.dark { color-scheme: dark; %(dark)s }

.gradio-container { font-family: %(sans)s; zoom: %(zoom)s;
  max-width: %(maxw)spx !important; margin: 0 auto !important;
  padding-left: 8px !important; padding-right: 8px !important; }
.gradio-container > .main, .gradio-container .contain { padding-left: 0 !important; padding-right: 0 !important; }
@supports not (zoom: 1) { .gradio-container { font-size: %(zoompct)s; } }
footer { display: none !important; }

/* ---- header -------------------------------------------------------- */
#cs-header { display:flex; align-items:center; gap:.75rem; padding:.2rem 0 .45rem;
  border-bottom:1px solid var(--cs-line); margin-bottom:.4rem; flex-wrap:nowrap; }
#cs-home { font-size:1.2rem !important; font-weight:700 !important; letter-spacing:-.01em;
  color:var(--cs-accent-text) !important; background:none !important; border:none !important;
  box-shadow:none !important; padding:0 !important; margin:0 !important; min-width:0 !important;
  width:auto !important; flex:0 0 auto; cursor:pointer; }
#cs-home:hover { text-decoration:underline; text-underline-offset:3px; }
.cs-sub { font-size:.76rem; color:var(--cs-muted); }
#cs-theme { flex:0 0 auto; }
#cs-theme .wrap { flex-wrap:nowrap !important; }

/* ---- workflow stepper ---------------------------------------------- */
.cs-steps { display:flex; gap:0; margin:.1rem 0 .35rem; padding:0; list-style:none;
  border:1px solid var(--cs-line); border-radius:8px; overflow:hidden; background:var(--cs-surface-alt); }
.cs-step { flex:1 1 0; min-width:0; display:flex; align-items:center; gap:.5rem;
  padding:.42rem .65rem; border-right:1px solid var(--cs-line); position:relative; }
.cs-step:last-child { border-right:0; }
.cs-step .cs-dot { flex:none; width:1.35rem; height:1.35rem; border-radius:50%%;
  display:inline-flex; align-items:center; justify-content:center; font-size:.72rem; font-weight:700;
  border:1.5px solid var(--cs-line-strong); color:var(--cs-muted); background:var(--cs-surface); }
.cs-step .cs-label { font-size:.8rem; font-weight:650; color:var(--cs-text); white-space:nowrap; }
.cs-step .cs-detail { display:block; font-size:.68rem; color:var(--cs-muted);
  white-space:nowrap; overflow:hidden; text-overflow:ellipsis; }
.cs-step .cs-text { min-width:0; }
.cs-step.done .cs-dot { background:var(--cs-ok); border-color:var(--cs-ok); color:var(--cs-surface); }
.cs-step.current { background:var(--cs-surface); box-shadow:inset 0 -3px 0 var(--cs-accent); }
.cs-step.current .cs-dot { border-color:var(--cs-accent); color:var(--cs-accent-text); }
.cs-step.warn .cs-dot { background:var(--cs-warn-bg); border-color:var(--cs-warn); color:var(--cs-warn); }
.cs-step.warn .cs-detail { color:var(--cs-warn); }
.cs-step.error .cs-dot { background:var(--cs-err-bg); border-color:var(--cs-err); color:var(--cs-err); }
.cs-step.error .cs-detail { color:var(--cs-err); }
.cs-step.optional .cs-dot { border-style:dashed; }
.cs-next { display:flex; align-items:baseline; gap:.5rem; font-size:.82rem; color:var(--cs-text);
  padding:.1rem .15rem .35rem; }
.cs-next b { color:var(--cs-accent-text); font-weight:700; text-transform:uppercase;
  letter-spacing:.06em; font-size:.66rem; }

/* ---- status strip ---------------------------------------------------- */
.cs-status { display:flex; flex-wrap:wrap; border:1px solid var(--cs-line); border-radius:8px;
  overflow:hidden; background:var(--cs-surface-alt); margin-bottom:.35rem; }
.cs-stat { flex:1 1 0; min-width:120px; padding:.38rem .65rem; border-right:1px solid var(--cs-line); }
.cs-stat:last-child { border-right:0; }
.cs-k { display:block; font-size:.6rem; text-transform:uppercase; letter-spacing:.07em;
  color:var(--cs-muted); margin-bottom:.08rem; }
.cs-v { display:block; font-size:.82rem; font-weight:600; color:var(--cs-text);
  overflow:hidden; text-overflow:ellipsis; white-space:nowrap; font-variant-numeric:tabular-nums; }
.cs-stale .cs-v { color:var(--cs-warn); }
.cs-stale .cs-v::before { content:"! "; font-weight:800; }

/* ---- navigator, badges, strip --------------------------------------- */
.cs-nav { display:flex; align-items:center; gap:.5rem; flex-wrap:wrap; border:1px solid var(--cs-line);
  border-left:3px solid var(--cs-accent); border-radius:6px; background:var(--cs-surface-alt);
  padding:.35rem .65rem; font-size:.82rem; color:var(--cs-text); }
.cs-nav .cs-pos { font-variant-numeric:tabular-nums; color:var(--cs-muted); font-size:.74rem; }
.cs-nav .cs-name { font-weight:650; overflow:hidden; text-overflow:ellipsis; max-width:40ch; white-space:nowrap; }
.cs-nav .cs-sep { color:var(--cs-line-strong); }
.cs-badge { display:inline-flex; align-items:center; gap:.25rem; font-size:.69rem; font-weight:600;
  padding:.08rem .45rem; border-radius:999px; border:1px solid var(--cs-line-strong);
  background:var(--cs-neutral-bg); color:var(--cs-neutral); white-space:nowrap; }
.cs-badge.ok { color:var(--cs-ok); border-color:var(--cs-ok); background:var(--cs-ok-bg); }
.cs-badge.warn { color:var(--cs-warn); border-color:var(--cs-warn); background:var(--cs-warn-bg); }
.cs-badge.err { color:var(--cs-err); border-color:var(--cs-err); background:var(--cs-err-bg); }
.cs-strip { display:flex; flex-wrap:wrap; gap:.28rem; margin:.3rem 0; }
.cs-tile { display:inline-flex; align-items:center; gap:.32rem; font-size:.7rem; max-width:22ch;
  padding:.18rem .45rem; border-radius:5px; border:1px solid var(--cs-line); background:var(--cs-surface);
  color:var(--cs-text); white-space:nowrap; overflow:hidden; text-overflow:ellipsis; }
.cs-tile .n { color:var(--cs-muted); font-variant-numeric:tabular-nums; }
.cs-tile .s { font-weight:800; }
.cs-tile.done { background:var(--cs-ok-bg); border-color:var(--cs-ok); }
.cs-tile.done .s { color:var(--cs-ok); }
.cs-tile.stale, .cs-tile.running { border-color:var(--cs-warn); }
.cs-tile.stale .s, .cs-tile.running .s { color:var(--cs-warn); }
.cs-tile.err { border-color:var(--cs-err); color:var(--cs-err); }
.cs-tile.active { outline:2px solid var(--cs-accent); outline-offset:1px; }

/* ---- sections, notes, callouts -------------------------------------- */
.cs-h { font-size:.68rem; text-transform:uppercase; letter-spacing:.08em; color:var(--cs-muted);
  font-weight:700; margin:.6rem 0 .15rem; }
.cs-note { font-size:.79rem; color:var(--cs-muted); line-height:1.5; max-width:80ch; }
.cs-note strong, .cs-note code { color:var(--cs-text); }
.cs-callout { border-left:3px solid var(--cs-accent); padding:.45rem .75rem; background:var(--cs-accent-soft);
  border-radius:0 6px 6px 0; font-size:.8rem; color:var(--cs-text); }
.cs-warnbox { border-left:3px solid var(--cs-warn); padding:.45rem .75rem; background:var(--cs-warn-bg);
  border-radius:0 6px 6px 0; font-size:.8rem; color:var(--cs-text); }
.cs-errbox { border-left:3px solid var(--cs-err); padding:.45rem .75rem; background:var(--cs-err-bg);
  border-radius:0 6px 6px 0; font-size:.8rem; color:var(--cs-text); }
.cs-callout p, .cs-warnbox p, .cs-errbox p { margin:0; }
.cs-empty { border:1px dashed var(--cs-line-strong); border-radius:8px; padding:1rem;
  text-align:center; color:var(--cs-muted); font-size:.82rem; background:var(--cs-surface-alt); }
.cs-empty strong { display:block; color:var(--cs-text); font-size:.9rem; margin-bottom:.2rem; }
.cs-kbd, kbd { font-family:%(mono)s; font-size:.7rem; padding:.05rem .32rem; border-radius:4px;
  border:1px solid var(--cs-line-strong); border-bottom-width:2px; background:var(--cs-surface); }
.cs-legend { display:flex; flex-wrap:wrap; gap:.35rem; margin:.4rem 0 .1rem; }
.cs-chip { display:inline-flex; align-items:center; gap:.38rem; font-size:.72rem; color:var(--cs-text);
  padding:.15rem .5rem; border-radius:999px; border:1px solid var(--cs-line); background:var(--cs-surface); }
.cs-swatch { width:.7rem; height:.7rem; border-radius:3px; flex:none; box-shadow:inset 0 0 0 1px rgba(0,0,0,.25); }
.cs-swatch.thick { outline:2px solid currentColor; outline-offset:1px; }
.cs-swatch.excluded { background:#555 !important; box-shadow:inset 0 0 0 2px #a0a0a0; }
.cs-swatch.ruler { background:transparent !important; border:1.5px dotted #a0a0a0; }

/* ---- buttons: primary, secondary, destructive ------------------------ */
.gradio-container button.stop, .gradio-container button[variant="stop"] {
  background:transparent !important; color:var(--cs-err) !important;
  border:1px solid var(--cs-err) !important; }
.gradio-container button.stop:hover { background:var(--cs-err-bg) !important; }
.cs-hidden-button { display:none !important; }

/* ---- tables ---------------------------------------------------------- */
.gradio-container table, .gradio-container th, .gradio-container td,
.gradio-container .cell-wrap, .gradio-container .cell-wrap span {
  font-family:%(sans)s !important; font-variant-numeric:tabular-nums; font-size:.78rem; }
.gradio-container thead th { font-weight:650; color:var(--cs-text); background:var(--cs-surface-alt); }
.gradio-container tbody td { white-space:nowrap; }
.gradio-container .table-wrap { max-width:100%% !important; }
.gradio-container code, .gradio-container pre, .gradio-container kbd { font-family:%(mono)s !important; }
.tabs button.selected { color:var(--cs-accent-text) !important; font-weight:650; }

/* ---- review viewer --------------------------------------------------- */
.cs-viewer { position:relative; border:1px solid var(--cs-line); border-radius:8px; overflow:hidden;
  background:var(--cs-viewer-bg); user-select:none; }
.cs-viewport { position:relative; width:100%%; overflow:hidden; cursor:crosshair; touch-action:none; }
.cs-viewer.panning .cs-viewport { cursor:grabbing; }
.cs-stage { position:absolute; left:0; top:0; transform-origin:0 0; will-change:transform; }
.cs-stage img { position:absolute; left:0; top:0; width:100%%; height:100%%;
  image-rendering:pixelated; pointer-events:none; -webkit-user-drag:none; }
.cs-stage .cs-fill { opacity:var(--cs-alpha, .28); }
.cs-viewer.original .cs-fill, .cs-viewer.original .cs-lines { visibility:hidden; }
.cs-toolbar { display:flex; align-items:center; gap:.35rem; flex-wrap:wrap; padding:.3rem .45rem;
  background:var(--cs-surface-alt); border-top:1px solid var(--cs-line); font-size:.74rem; color:var(--cs-muted); }
.cs-toolbar button { font:inherit; font-size:.74rem; color:var(--cs-text); background:var(--cs-surface);
  border:1px solid var(--cs-line-strong); border-radius:5px; padding:.12rem .5rem; cursor:pointer; min-width:1.9rem; }
.cs-toolbar button:hover { border-color:var(--cs-accent); }
.cs-toolbar button[aria-pressed="true"] { background:var(--cs-accent-soft); border-color:var(--cs-accent); }
.cs-toolbar input[type=range] { width:7rem; accent-color:var(--cs-accent); }
.cs-toolbar .cs-zoom { font-variant-numeric:tabular-nums; min-width:3.2rem; text-align:right; }
.cs-toolbar .cs-spacer { flex:1 1 auto; }
.cs-hover { position:absolute; pointer-events:none; z-index:5; padding:.15rem .4rem; border-radius:4px;
  font-size:.72rem; background:rgba(11,18,32,.88); color:#F1F5F9; white-space:nowrap; display:none; }
.cs-viewer .cs-mode { position:absolute; top:.4rem; left:.4rem; z-index:4; font-size:.7rem; font-weight:650;
  padding:.1rem .45rem; border-radius:4px; background:rgba(11,18,32,.78); color:#F1F5F9; }
.cs-viewer .cs-empty { margin:1rem; }
#cs-bridge, #cs-key-bridge { display:none !important; }
.cs-shortcuts { font-size:.75rem; color:var(--cs-muted); line-height:1.9; }

#cs-results { margin-left:-4px; margin-right:-4px; }
"""
CSS = CSS % {
    "light": _tokens(TOKENS_LIGHT), "dark": _tokens(TOKENS_DARK),
    "zoom": PAGE_ZOOM, "zoompct": "{:.0f}%".format(PAGE_ZOOM * 100),
    "maxw": int(round(CONTENT_WIDTH / PAGE_ZOOM)), "sans": SANS, "mono": MONO,
}

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
        # Dataframe cells read var(--font-mono); pointing it at the sans stack
        # is the one override that survives Gradio's hashed class names.
        font_mono=[gr.themes.GoogleFont("Inter"), "Helvetica Neue", "Arial", "sans-serif"],
        radius_size=gr.themes.sizes.radius_sm,
        spacing_size=gr.themes.sizes.spacing_sm,
    ).set(
        body_background_fill=TOKENS_LIGHT["surface"],
        body_background_fill_dark=TOKENS_DARK["surface"],
        background_fill_primary=TOKENS_LIGHT["surface"],
        background_fill_primary_dark=TOKENS_DARK["surface"],
        background_fill_secondary=TOKENS_LIGHT["surface-alt"],
        background_fill_secondary_dark=TOKENS_DARK["surface-alt"],
        block_background_fill=TOKENS_LIGHT["surface"],
        block_background_fill_dark=TOKENS_DARK["surface"],
        block_border_color=TOKENS_LIGHT["line"],
        block_border_color_dark=TOKENS_DARK["line"],
        block_shadow="none",
        block_shadow_dark="none",
        border_color_primary=TOKENS_LIGHT["line"],
        border_color_primary_dark=TOKENS_DARK["line"],
        body_text_color=TOKENS_LIGHT["text"],
        body_text_color_dark=TOKENS_DARK["text"],
        body_text_color_subdued=TOKENS_LIGHT["muted"],
        body_text_color_subdued_dark=TOKENS_DARK["muted"],
        block_label_text_color=TOKENS_LIGHT["muted"],
        block_label_text_color_dark=TOKENS_DARK["muted"],
        block_title_text_color=TOKENS_LIGHT["text"],
        block_title_text_color_dark=TOKENS_DARK["text"],
        block_title_text_weight="600",
        button_primary_background_fill=FOREST,
        button_primary_background_fill_dark="#166534",
        button_primary_background_fill_hover=FOREST_DARK,
        button_primary_background_fill_hover_dark="#15803D",
        button_primary_text_color="#FFFFFF",
        button_primary_text_color_dark="#FFFFFF",
        button_primary_border_color=FOREST,
        button_primary_border_color_dark="#166534",
        button_secondary_background_fill=TOKENS_LIGHT["surface"],
        button_secondary_background_fill_dark=TOKENS_DARK["surface-alt"],
        button_secondary_border_color=TOKENS_LIGHT["line-strong"],
        button_secondary_border_color_dark=TOKENS_DARK["line-strong"],
        checkbox_background_color_selected=FOREST,
        checkbox_background_color_selected_dark="#166534",
        slider_color=FOREST,
        slider_color_dark="#22C55E",
        # Plain field labels: the Soft theme's tinted chips gave every label the
        # visual weight of a button.
        block_title_background_fill="transparent",
        block_title_background_fill_dark="transparent",
        block_title_padding="0",
        block_title_radius="0",
        block_label_background_fill=TOKENS_LIGHT["surface-alt"],
        block_label_background_fill_dark=TOKENS_DARK["surface-alt"],
        input_background_fill=TOKENS_LIGHT["surface"],
        input_background_fill_dark=TOKENS_DARK["surface-sunken"],
    )


# --------------------------------------------------------------------------- #
# HTML helpers
# --------------------------------------------------------------------------- #

STATE_SYMBOLS = {"done": "✓", "current": "", "todo": "", "warn": "!", "error": "✕", "optional": ""}


def legend_html(excluded: bool = True, ruler: bool = True) -> str:
    """Colour key for the overlays, as chips rather than a run of glyphs."""
    chips = []
    for (colour, label), key in zip(overlay_legend(), ("isolated", "clustered", "unresolved")):
        thick = " thick" if key == "unresolved" else ""
        chips.append(
            '<span class="cs-chip"><span class="cs-swatch{}" style="background:{};color:{}">'
            "</span>{}</span>".format(thick, colour, colour, escape(label))
        )
    if excluded:
        chips.append('<span class="cs-chip"><span class="cs-swatch excluded"></span>'
                     "excluded (dimmed, grey outline)</span>")
    if ruler:
        chips.append('<span class="cs-chip"><span class="cs-swatch ruler"></span>'
                     "burned-in scale bar zone</span>")
    return '<div class="cs-legend">{}</div>'.format("".join(chips))


def empty_state(title: str, hint: str) -> str:
    """A placeholder that says what to do next rather than showing nothing."""
    return '<div class="cs-empty"><strong>{}</strong>{}</div>'.format(title, hint)


def callout(text: str, tone: str = "info") -> str:
    css_class = {"warn": "cs-warnbox", "error": "cs-errbox"}.get(tone, "cs-callout")
    return '<div class="{}"><p>{}</p></div>'.format(css_class, text)


def section(title: str) -> str:
    return '<div class="cs-h">{}</div>'.format(escape(title))


def badge(text: str, tone: str = "") -> str:
    symbol = {"ok": "✓ ", "warn": "! ", "err": "✕ "}.get(tone, "")
    return '<span class="cs-badge {}">{}{}</span>'.format(tone, symbol, escape(text))


def stepper_html(steps, action: str) -> str:
    """The workflow stepper and the one-line next action."""
    items = []
    for number, step in enumerate(steps, start=1):
        symbol = STATE_SYMBOLS.get(step.state, "") or str(number)
        state_word = {"done": "done", "current": "current step", "todo": "to do",
                      "warn": "needs attention", "error": "problem",
                      "optional": "optional"}[step.state]
        items.append(
            '<li class="cs-step {state}" aria-label="{label}: {word}">'
            '<span class="cs-dot" aria-hidden="true">{symbol}</span>'
            '<span class="cs-text"><span class="cs-label">{label}</span>'
            '<span class="cs-detail">{detail}</span></span></li>'.format(
                state=step.state, label=escape(step.label), word=state_word, symbol=symbol,
                detail=escape(step.detail or state_word),
            )
        )
    return (
        '<ol class="cs-steps" aria-label="Analysis workflow">{}</ol>'
        '<div class="cs-next" role="status"><b>Next</b><span>{}</span></div>'
    ).format("".join(items), escape(action))
