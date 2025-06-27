# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULCMore actions
#
# SPDX-License-Identifier: Apache-2.0
# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------
# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here.
import sys
import os
from pathlib import Path

# Add the Python packages directory to the path
build_dir = Path(__file__).resolve().parents[3] / "build"
sys.path.insert(0, str(build_dir / "python_packages/ttir_builder"))

# Output paths
html_dir = build_dir / "docs/book/autogen/html/Module"
md_dir = build_dir / "docs/src/autogen/md/Module"

# Create output directories
html_dir.mkdir(parents=True, exist_ok=True)
md_dir.mkdir(parents=True, exist_ok=True)

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = "ttir-builder"
copyright = "2025, Julia Grim"
author = "Julia Grim"
release = "0.1"

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "sphinx.ext.duration",
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
]

# Autodoc settings
autodoc_default_options = {
    "members": True,
    "undoc-members": False,
    "show-inheritance": True,
    "inherited-members": True,
    "private-members": False,
}
autodoc_docstring_signature = True
autodoc_typehints = "description"
autodoc_member_order = "bysource"

# Napoleon settings
napoleon_numpy_docstring = True

# Autosummary settings
autosummary_generate = True

# Exclude patterns
exclude_patterns = ["modules.rst", "ttir_builder.rst"]

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "sphinx_rtd_theme"


def autodoc_skip_member(app, what, name, obj, skip, options):
    if hasattr(obj, "__autodoc_skip__") and obj.__autodoc_skip__:
        return True  # Skip this member
    return skip


def setup(app):
    app.add_css_file("tt_theme.css")
    app.connect("autodoc-skip-member", autodoc_skip_member)
