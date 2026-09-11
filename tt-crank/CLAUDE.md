# tt-crank — Claude Code guidance

tt-crank is a subproject of tt-mlir and is built from the tt-mlir root; it has no build scripts, virtualenv or vendored tt-mlir of its own. tt-mlir's root `CLAUDE.md` (environment, commands, code style) applies here as well — the notes below are specific to `tt-crank/`.

## Build & test

All commands run from the tt-mlir root.

```sh
source env/activate
cmake -G Ninja -B build -DCMAKE_CXX_COMPILER_LAUNCHER=ccache -DTTMLIR_ENABLE_CRANK=ON
cmake --build build
./tt-crank/scripts/test          # C++ unit tests on a device
./tt-crank/scripts/test sim      # … routed through ttsim
./tt-crank/scripts/test sim -- --gtest_filter='EngineCompileTest.*'   # forward gtest flags
```

- `-DTTMLIR_ENABLE_CRANK=ON` auto-enables `TTMLIR_ENABLE_RUNTIME`, `TTMLIR_ENABLE_OPMODEL` and the TTNN runtime (`TT_RUNTIME_ENABLE_TTNN`) with a `STATUS` line each; no other `TTMLIR_*` flags are needed.
- tt-crank's own options are `TT_CRANK_*`, passed on the same configure line: `TT_CRANK_BUILD_TESTS`, `TT_CRANK_BUILD_PYTHON_EXT`, `TT_CRANK_TRACY_ZONES` (needs `TT_RUNTIME_ENABLE_PERF_TRACE=ON`), `TT_CRANK_SIM_ARCH`, `TT_CRANK_TTSIM_VERSION`, and the sanitizer / clang-tidy switches in `cmake/Sanitizers.cmake` and `cmake/StaticAnalyzers.cmake`. There are no presets; the build type is the tt-mlir build's `CMAKE_BUILD_TYPE`.
- Outputs land under `build/tt-crank/`. The unit-test binary is `build/tt-crank/tests/tt_crank_unit_tests`; `scripts/test` runs it directly (set `BUILD_DIR` for a different tt-mlir build directory).
- tt-metal is the one tt-mlir vendors at `third_party/tt-metal/src/tt-metal`; nothing lives under `tt-crank/third_party/` except the tt-logger fetch.
- Runtime behaviour is switched by environment variables declared in `src/common/config.hpp` (`TT_CRANK_USE_SIMULATOR`, `TT_CRANK_BACKTRACE_DISABLED`, `TT_CRANK_ASSERT_ABORT_ENABLED`, …); presence of the variable is what counts. `TT_CRANK_USE_SIMULATOR=1` routes any runner through ttsim.
- Python installation (editable install of `tt_crank` plus the `tracy` / `ttnn` wrappers) and wheels follow in later commits. `pytest tests/python/` (run from `tt-crank/`, `--sim` for ttsim) needs that install.
- Formatting and lint go through tt-mlir's hooks, from the tt-mlir root: `pre-commit run --files <changed files>`.

## C++ conventions

### Error handling — never throw directly

Use the macros from `src/common/assert.hpp` (included as `"assert.hpp"`; `tt_crank_common` exports `src/common/` as an include directory). Never use `throw` or `assert()`.

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

Use `as<To>(value)` from `src/common/cast.hpp` (included as `"cast.hpp"`):

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

Enum types are formattable out of the box via the generic `std::formatter` specialization in `assert.hpp` (formats as the underlying integer). Include `"assert.hpp"` to get it.

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
