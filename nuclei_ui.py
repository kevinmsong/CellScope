"""Compatibility shim: the DAPI tab now lives in ``cellscope.ui.nuclei_tab``."""

from cellscope.ui.nuclei_tab import nuclear_view

__all__ = ["nuclear_view"]
