# tt-kurbla

Compiler & runtime frontend for Tenstorrent hardware. POC.

See [`architecture-overview.md`](architecture-overview.md) for the high-level design goals and the plan.

## Prerequisites

- Linux (x86_64)
- CMake ≥ 3.24, Ninja
- clang ≥ 20 via the tt-mlir toolchain (built / activated by `env/activate`)
- `ccache` (optional — used automatically if on `PATH`)

## First-time setup

The first build builds the LLVM/MLIR toolchain that `tt-mlir` depends on — multi-GB, ~30 min, **once per machine**.

```sh
git clone --recurse-submodules <repo> tt-kurbla
cd tt-kurbla
source venv/activate
./scripts/build
```

## Daily build & test

```sh
./scripts/build [preset]     # configure + build (default: debug)
./scripts/test [sim]         # run tests against the current build
```

```sh
# Or, with the env already activated in your shell:
cmake --preset <preset>
cmake --build --preset <preset>
```

Pass `--help` to either script for full usage.

## Preset cheat sheet

| Preset    | Build type     | Compiler | Extras                                       |
|---        |---             |---       |---                                           |
| `debug`   | Debug          | clang    | Default preset                               |
| `release` | Release        | clang    |                                              |
| `san`     | RelWithDebInfo | clang    | ASan + UBSan on first-party code (tt-mlir unsanitized) |

Build directories are created as `build_<preset>/`. `scripts/build` symlinks `build/` to the most recently built preset, which is what `scripts/test` uses.

### Running through ttsim

Every build already stages `tenstorrent/ttsim` alongside its SoC descriptor — no separate configure flag. Pass `sim` to `scripts/test` to route the runtime through the simulator:

```sh
./scripts/build                # build (debug preset)
./scripts/test sim             # run tests via ttsim
./scripts/test sim -- -R EngineCompileTest   # filter tests + ttsim
```

`TT_KURBLA_USE_SIMULATOR=1` is the underlying switch — set it directly to use ttsim with any other runner (Python extension, manual binary invocation, etc.).

Override the source tree of `tt-mlir` without editing the submodule:

```sh
cmake --preset debug -DTTMLIR_SOURCE_DIR_OVERRIDE=/path/to/sibling/tt-mlir
```

## Python tests

The Python test suite lives in `tests/python/` and runs with pytest. The build must exist first (`./scripts/build`), as pytest imports the compiled `tt_kurbla` extension.

```sh
source venv/activate
pytest tests/python/                  # run all Python tests
pytest tests/python/ --sim            # route through ttsim
pytest tests/python/ -k test_device   # filter by name
pytest tests/python/op_tests/         # run a subdirectory only
```

`--sim` sets `TT_KURBLA_USE_SIMULATOR=1` before the extension is loaded. It is equivalent to setting the variable manually, but must be passed as a pytest flag (not an env var) because the extension reads it in a static initializer at import time — before pytest option parsing runs.

## Layout

```
tt-kurbla/
├── CMakeLists.txt
├── CMakePresets.json
├── cmake/                 # CompilerWarnings, Sanitizers, StaticAnalyzers, Cache, SystemIncludes
├── scripts/
│   ├── build              # configure + build wrapper
│   └── test               # test runner wrapper (supports sim)
├── src/                   # public + private headers + sources, by component
│   ├── version.{hpp,cpp.in}
│   └── engine/            # compile / runtime API (POC)
├── tests/                 # unit tests, one file per src component
├── third_party/
│   ├── CMakeLists.txt     # ExternalProject_Add(tt-mlir), FetchContent(tt-logger)
│   └── tt-mlir/           # submodule
└── architecture-overview.md
```

Headers live alongside their `.cpp` under `src/<component>/...` — no separate `include/` tree. Internal/downstream callers include as `"engine/foo.hpp"`.

## Logging

Logging uses [`tt-logger`](https://github.com/tenstorrent/tt-logger) (spdlog-based). Control it at runtime via environment variables:

| Variable | Values | Default |
|---|---|---|
| `TT_LOGGER_LEVEL` | `debug`, `info`, `warn`, `error`, `critical`, `off` | `info` |
| `TT_LOGGER_FILE` | path to log file | stdout |
| `TT_LOGGER_TYPES` | comma-separated logger names, or `All` | all loggers |

```sh
# Enable debug logging
TT_LOGGER_LEVEL=debug ./scripts/test sim

# Debug output to a file
TT_LOGGER_LEVEL=debug TT_LOGGER_FILE=/tmp/kurbla.log ./scripts/test sim

# Filter to specific subsystems
TT_LOGGER_LEVEL=debug TT_LOGGER_TYPES=TTNN,Op ./scripts/test sim

# Show only tt-kurbla logs (suppress tt-mlir/tt-metal noise)
TT_LOGGER_LEVEL=debug TT_LOGGER_TYPES=Always ./scripts/test sim
```

`debug` log calls are compiled in only on `debug`/`san` builds — they are no-ops in `release`.

## Dev tooling

`venv/activate` sets up the project environment. Pre-commit hooks are installed on first `./scripts/build` run.

```sh
pre-commit run --all-files    # run all hooks across the tree
pre-commit uninstall          # remove the git hook
```

## Status

POC. See [`architecture-overview.md`](architecture-overview.md) for the design plan.
