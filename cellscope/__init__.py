"""CellScope application package: the interface and its command-line launcher.

The scientific code lives in :mod:`src`, which never imports Gradio; this
package holds everything that does. The version has a single source, in
``src/__init__.py``.
"""

from src import __version__

__all__ = ["__version__"]
