# tt-crank — Claude Code guidance

## Build & test

```sh
source venv/activate
./scripts/build [debug|release|san]   # configure + build
./scripts/test [sim]                  # run C++ tests (sim = route through ttsim)
./scripts/install-py                  # editable-install the Python extension
uv build --wheel                      # build a distributable wheel into dist/
pytest tests/python/                  # run Python tests
pytest tests/python/ --sim            # Python tests via ttsim
```

`./scripts/install-py` does an editable `uv pip install -e .` with `SKBUILD_BUILD_DIR` pointed at the existing `build/` symlink, so cmake artifacts stay consistent with a normal `./scripts/build` run. It bootstraps `build/` via `./scripts/build` if it doesn't exist yet. Run it once after the first build, then re-run whenever the C++ extension sources change.

`uv build --wheel` needs no wrapper: everything that differs from a dev build is in the `[[tool.scikit-build.overrides]]` block of `pyproject.toml`. It builds in `build_wheel/` and leaves the `build` symlink alone — wheels link with `$ORIGIN` RPATHs and dev builds with absolute build-tree paths, so sharing one dir relinks everything on each switch. tt-metal lives in the shared submodule tree, so it is not rebuilt; don't run a wheel build and `./scripts/build` concurrently.

## C++ conventions

### Error handling — never throw directly

Use the macros from `engine/assert.hpp`. Never use `throw` or `assert()`.

| Macro | When |
|---|---|
| `TT_FATAL(cond, ...)` | Always-on assertion; throws with backtrace on failure |
| `TT_ASSERT(cond, ...)` | Debug-only assertion; no-op in release builds |
| `TT_THROW(...)` | Unconditional throw with backtrace |

Format args work exactly like `std::format`:

```cpp
TT_FATAL(ptr != nullptr, "expected non-null ptr at index {}", i);
TT_ASSERT(x >= 0, "x must be non-negative, got {}", x);
```

Do **not** pre-format the message and pass it as the first variadic arg — `std::format_string` is `consteval` and requires a string literal:

```cpp
TT_FATAL(ok, "{}", std::format(...));  // wrong — std::string is not a format_string
TT_FATAL(ok, "value: {}", value);      // correct
```

### Casts — never use `static_cast` for numeric conversions

Use `as<To>(value)` from `engine/cast.hpp`:

```cpp
as<int>(some_uint32);          // checked in debug; bit-cast for pointers/refs
as<int>(some_enum);            // works — enum formatted via underlying type
```

`static_cast` is only acceptable inside `cast.hpp` itself or for non-numeric conversions (e.g., pointer upcasts in third-party API calls where `as<>` doesn't apply).

### Formatting — always use `std::format`, never `fmt::format`

```cpp
#include <format>

std::string s = std::format("value: {}", x);
```

Enum types are formattable out of the box via the generic `std::formatter` specialization in `assert.hpp` (formats as the underlying integer). Include `engine/assert.hpp` to get it.

`fmt::` APIs (`fmt::format`, `fmt::join`, `<fmt/ranges.h>`) are pulled in transitively by tt-mlir headers — do not use them directly in first-party code.

### Comment style — always `//`, never `/* */`

All comments use `//`. For multi-line comments, repeat `//` on each line:

```cpp
// Skips the top `skip` frames so backtrace() itself doesn't appear in output.
// Returns at most `size` frames.
inline std::vector<std::string> backtrace(size_t size = 64, size_t skip = 1) {
```

Never use `/* */` or `/** @brief ... */` block comments.

### MLIR result checks

Use `mlir::succeeded(result)` (not `!mlir::failed(result)`) for readability:

```cpp
TT_FATAL(mlir::succeeded(pm.run(module)), "pipeline failed: {}", diag_buffer);
```
