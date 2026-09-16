"""Compatibility shim: the theme now lives in ``cellscope.ui.theme``."""

from cellscope.ui.theme import (
    CSS,
    THEME_JS,
    build_theme,
    callout,
    empty_state,
    legend_html,
    section,
)

__all__ = ["CSS", "THEME_JS", "build_theme", "callout", "empty_state", "legend_html", "section"]
