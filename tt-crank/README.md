# tt-kurbla

Compiler & runtime frontend for Tenstorrent hardware. POC.

See [`architecture-overview.md`](architecture-overview.md) for the high-level design goals and the plan.

## Prerequisites

- Linux (x86_64)
- CMake ≥ 3.24, Ninja
- gcc ≥ 11 or clang ≥ 14 (system)
- clang ≥ 20 via the tt-mlir toolchain (built / activated by `env/activate`)
- `ccache` (optional — used automatically if on `PATH`)

## First-time setup

The first build builds the LLVM/MLIR toolchain that `tt-mlir` depends on — multi-GB, ~30 min, **once per machine**.

```sh
git clone --recurse-submodules <repo> tt-kurbla
cd tt-kurbla
./scripts/build      # init submodules → source env/activate → configure → build → test
```

## Daily build

After first-time setup, either of these works:

```sh
./scripts/build [preset]     # wrapper handles env activation + cmake invocation
```

```sh
# Or, with the env already activated in your shell:
cd third_party/tt-mlir && source env/activate && cd -
cmake --preset <preset>
cmake --build --preset <preset>
ctest --preset <preset>
```

## Preset cheat sheet

| Preset          | Build type     | Compiler | Extras                                                  |
|---              |---             |---       |---                                                      |
| `default`       | Release        | gcc      | The "just works" preset                                 |
| `dev`           | RelWithDebInfo | gcc      | `-Werror` + clang-tidy; the daily-dev preset            |
| `debug`         | Debug          | gcc      |                                                         |
| `clang-release` | Release        | clang    | Portability sanity                                      |
| `clang-debug`   | Debug          | clang    |                                                         |
| `san`           | RelWithDebInfo | gcc      | ASan + UBSan on first-party code (tt-mlir unsanitized)  |

Override the source tree of `tt-mlir` without editing the submodule:

```sh
cmake --preset default -DTTMLIR_SOURCE_DIR_OVERRIDE=/path/to/sibling/tt-mlir
```

## Layout

```
tt-kurbla/
├── CMakeLists.txt
├── CMakePresets.json
├── cmake/                 # CompilerWarnings, Sanitizers, StaticAnalyzers, Cache
├── scripts/build          # one-shot wrapper (env + cmake + test)
├── src/                   # public + private headers + sources, by component
│   ├── version.{hpp,cpp.in}
│   └── engine/            # compile / runtime API (POC)
├── tests/                 # unit tests, one file per src component
├── third_party/
│   ├── CMakeLists.txt     # ExternalProject_Add(tt-mlir)
│   └── tt-mlir/           # submodule
└── architecture-overview.md
```

Headers live alongside their `.cpp` under `src/<component>/...` — no separate `include/` tree. Internal/downstream callers include as `"engine/foo.hpp"`.

## Status

POC. See [`architecture-overview.md`](architecture-overview.md) for the design plan.
