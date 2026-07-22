# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

# Configuration file for the Sphinx documentation builder.
# https://www.sphinx-doc.org/en/master/usage/configuration.html

import os
import subprocess

# -- Project information -----------------------------------------------------

project = "TT-MLIR"
copyright = "2025, Tenstorrent AI ULC"
author = "Tenstorrent"

# -- General configuration ---------------------------------------------------

extensions = [
    "sphinx_sitemap",
    "myst_parser",
]

source_suffix = {
    ".rst": "restructuredtext",
    ".md": "markdown",
}

myst_heading_anchors = 3
myst_enable_extensions = ["colon_fence"]

templates_path = ["_templates"]
exclude_patterns = [
    "SUMMARY.md",
    "_build",
    "specs/tensor-layout-interactive.html",
]

_MLIR_BASE = "https://docs.tenstorrent.com/tt-mlir/"

# -- Options for HTML output -------------------------------------------------

html_theme = "sphinx_rtd_theme"
html_theme_options = {
    "collapse_navigation": False,
    "titles_only": True,
    "navigation_depth": 2,
}
html_logo = "_static/tt_logo.svg"
html_favicon = "_static/favicon.png"
html_static_path = ["_static"]
html_css_files = ["https://docs.tenstorrent.com/_static/tt_theme.css"]
# Single-version site: published at a flat path, no version switcher.
html_baseurl = _MLIR_BASE
html_last_updated_fmt = "%b %d, %Y"

sitemap_locales = [None]
sitemap_url_scheme = "{link}"

html_context = {
    "logo_link_url": "https://docs.tenstorrent.com/",
    "search_site_base_url": _MLIR_BASE,
}
