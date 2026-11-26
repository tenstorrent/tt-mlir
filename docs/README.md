# Documentation

## Prerequisites

- Docs use `sphinx_rtd_theme` with the canonical Tenstorrent UI from [docs.tenstorrent.com](https://docs.tenstorrent.com/_static/tt_theme.css) (via `shared/sphinx/tt_theme.py` in the monorepo). Vendored assets under `docs/sphinx/_static/` are used for logo/templates and offline builds (`TT_DOCS_THEME_CSS=local`).
- User guides live in `docs/src/` (Markdown). For Sphinx HTML preview, `docs/sphinx/src` is a symlink to `docs/src/` so the full guide is included in the RTD-themed build.
- CMake’s `docs` target still uses [mdBook](https://github.com/rust-lang/mdBook) plus Sphinx markdown export for the Python builder API.
- Manual install: `python -m pip install -r docs/requirements.txt` (includes `myst-parser`, `sphinx-rtd-theme`).

## Build Sphinx HTML locally (full site)

Use a project venv so Homebrew’s `sphinx-build` is not picked up:

```bash
cd /path/to/tt-mlir
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r docs/requirements.txt

cd docs/sphinx
TT_DOCS_THEME_CSS=local python -m sphinx -b html . _build/html
python -m http.server 8000 -d _build/html
```

Open http://localhost:8000/index.html — the sidebar follows `docs/src/SUMMARY.md` (e.g. **Tools** → `ttmlir-opt`, `ttir-builder` → sub-pages), matching [docs.tenstorrent.com/tt-mlir](https://docs.tenstorrent.com/tt-mlir/tools.html).

If `docs/sphinx/src` is missing, recreate the symlink: `ln -sf ../src docs/sphinx/src`.

## Build (CMake / mdBook)

- Configure and build the `docs` target per the getting-started guide.
- Output is served with `mdbook serve build/docs` (default mdBook theme, not `sphinx_rtd_theme`).

## Theme assets

- Shared theme helper: monorepo `shared/sphinx/tt_theme.py`
- Project overrides: `docs/sphinx/_static/`, `docs/sphinx/_templates/`
