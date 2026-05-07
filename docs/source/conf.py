"""Sphinx configuration for the tabkernels package."""
from __future__ import annotations

import os
import sys
from datetime import datetime

# -- Path setup --------------------------------------------------------------
# Make the package importable so autodoc can introspect it.
sys.path.insert(0, os.path.abspath("../../src"))

# -- Project information -----------------------------------------------------
project = "tabkernels"
author = "Agus Sudjianto"
copyright = f"{datetime.now():%Y}, {author}"

try:
    from tabkernels.version import __version__ as _pkg_version
except Exception:  # pragma: no cover - docs build fallback
    _pkg_version = "0.1.0a1"

version = _pkg_version
release = _pkg_version

# -- General configuration ---------------------------------------------------
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx.ext.intersphinx",
    "sphinx.ext.mathjax",
    "sphinx_autodoc_typehints",
    "myst_parser",
]

templates_path = ["_templates"]
exclude_patterns: list[str] = []

# Source filename -> parser
source_suffix = {
    ".rst": "restructuredtext",
    ".md": "markdown",
}

# -- autodoc / autosummary ---------------------------------------------------
autosummary_generate = True
autodoc_default_options = {
    "members": True,
    "undoc-members": True,
    "show-inheritance": True,
    "inherited-members": False,
}
autodoc_member_order = "bysource"
autodoc_typehints = "description"
autodoc_class_signature = "separated"

# Don't choke on heavy/optional imports.
autodoc_mock_imports: list[str] = []

# -- Napoleon (NumPy + Google docstring support) -----------------------------
napoleon_google_docstring = True
napoleon_numpy_docstring = True
napoleon_include_init_with_doc = False
napoleon_include_private_with_doc = False
napoleon_use_param = True
napoleon_use_rtype = True

# -- Type hints --------------------------------------------------------------
always_document_param_types = True
typehints_fully_qualified = False

# -- Intersphinx -------------------------------------------------------------
intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable", None),
    "scipy": ("https://docs.scipy.org/doc/scipy", None),
    "sklearn": ("https://scikit-learn.org/stable", None),
    "torch": ("https://pytorch.org/docs/stable", None),
    "pandas": ("https://pandas.pydata.org/pandas-docs/stable", None),
    "matplotlib": ("https://matplotlib.org/stable", None),
}

# -- HTML output -------------------------------------------------------------
html_theme = "sphinx_rtd_theme"
html_static_path: list[str] = []
html_title = f"tabkernels {release}"
html_short_title = "tabkernels"

# Suppress noisy / benign warnings from third-party docstrings.
# - ref.python: harmless cross-reference noise from numpy/torch typehints
# - autosectionlabel.*: duplicate section labels across per-module rst files
# - misc.highlighting_failure: pygments lexer fallbacks
suppress_warnings = [
    "ref.python",
    "docutils",
]

# Treat duplicate object descriptions as benign (some symbols re-exported via
# package __init__).
nitpicky = False


# -- LaTeX role shims --------------------------------------------------------
# Some docstrings in tabkernels carry through LaTeX cross-reference roles
# (``\\cite{...}``, ``\\cref{...}``) from the monograph source. Register them
# as inert "literal" roles so docutils does not error on them while we keep
# the original source untouched.
def _latex_passthrough_role(name, rawtext, text, lineno, inliner, options=None, content=None):
    from docutils import nodes

    return [nodes.literal(rawtext, text)], []


def setup(app):  # noqa: D401 - sphinx hook
    """Register passthrough roles for LaTeX-flavoured docstring markup."""
    from docutils.parsers.rst import roles

    for role_name in ("cite", "citep", "citet", "cref", "Cref"):
        roles.register_local_role(role_name, _latex_passthrough_role)
    return {"version": _pkg_version, "parallel_read_safe": True}
