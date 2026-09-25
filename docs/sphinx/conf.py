# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
import sys
from pathlib import Path
from typing import Any

_conf_dir = Path(__file__).resolve().parent
for _ancestor in [_conf_dir, *_conf_dir.parents]:
    _sphinx_dir = _ancestor / "shared" / "sphinx"
    if (_sphinx_dir / "tt_theme.py").is_file():
        sys.path.insert(0, str(_sphinx_dir))
        break
    if _ancestor.name == "locals":
        _sphinx_dir = _ancestor.parent / "shared" / "sphinx"
        if (_sphinx_dir / "tt_theme.py").is_file():
            sys.path.insert(0, str(_sphinx_dir))
            break

from tt_theme import apply_html_theme_config, setup_sphinx_app

project = "TT-MLIR"
copyright = "2025 Tenstorrent AI ULC"
author = "TT-MLIR Team"
release = "0.1"

_local_static = _conf_dir / "_static"

extensions = [
    "myst_parser",
    "sphinx.ext.duration",
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
]

source_suffix = {
    ".rst": "restructuredtext",
    ".md": "markdown",
}

myst_enable_extensions = [
    "colon_fence",
]

# Permalink anchors only; do not use heading levels for sidebar depth.
myst_heading_anchors = 3
toc_object_entries = False

autodoc_default_options = {
    "members": True,
    "undoc-members": False,
    "show-inheritance": True,
    "inherited-members": False,
    "private-members": False,
}
autodoc_docstring_signature = True
autodoc_typehints = "description"
autodoc_member_order = "alphabetical"

napoleon_numpy_docstring = True

autosummary_generate = True

exclude_patterns = [
    "modules.rst",
    "base.rst",
    "ttir.rst",
    "stablehlo.rst",
    "src/SUMMARY.md",
    "src/specs/tensor-layout-interactive.html",
    "src/op-by-op-workflows.md",
    "builder/builder-utils.rst",
    "builder/stablehlo-builder.rst",
    "builder/ttir-builder.rst",
]

_theme = apply_html_theme_config(
    _conf_dir,
    local_static=_local_static,
    local_templates=_conf_dir / "_templates",
    local_logo=_local_static / "images" / "tt_logo.svg",
    local_favicon=_local_static / "images" / "favicon.png",
    html_context={"versions": None},
)

html_theme = _theme["html_theme"]
html_logo = _theme["html_logo"]
html_favicon = _theme["html_favicon"]
html_static_path = _theme["html_static_path"]
templates_path = _theme["templates_path"]
html_css_files = _theme["html_css_files"]
html_last_updated_fmt = "%b %d, %Y"
html_context = _theme["html_context"]

html_theme_options = {
    # Nested pages (Tools → ttir-builder → …) like mdBook SUMMARY.md.
    # Page hierarchy only (Tools → ttir-builder → …), not h2/h3 inside pages.
    "navigation_depth": 3,
    "collapse_navigation": False,
    "includehidden": True,
}


def autodoc_skip_member(
    app: Any, what: str, name: str, obj: Any, skip: bool, options: Any
) -> bool:
    if hasattr(obj, "__autodoc_skip__") and obj.__autodoc_skip__:
        return True
    return skip


def _page_title_only_toc(app: Any, docname: str, source: list[str]) -> None:
    """Sidebar: show page links only, not in-page h2/h3 (mdBook SUMMARY style)."""
    if not docname.startswith("src/"):
        return
    text = "".join(source)
    if text.lstrip().startswith("---"):
        return
    source[:] = ["---\ntocdepth: 1\n---\n\n", *source]


def setup(app: Any) -> None:
    setup_sphinx_app(app, _theme)
    app.connect("autodoc-skip-member", autodoc_skip_member)
    app.connect("source-read", _page_title_only_toc)
