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

## Benchmarks

Benchmarks live under `tests/python/benchmarks/` and run as a pytest target. The benchmark-only CLI flags are registered by `tests/python/benchmarks/conftest.py`, so you have to invoke pytest at-or-below that directory for them to be recognized.

```sh
pytest tests/python/                                   # sanity suite (this is what CI runs)
pytest tests/python/benchmarks/                        # run all benchmarks
pytest tests/python/benchmarks/test_mnist_linear.py    # one benchmark
```

The `benchmarks/` directory is in `norecursedirs` (`pytest.ini`), so plain `pytest tests/python` skips it during collection. Targeting a path inside `tests/python/benchmarks/` overrides that — pytest always collects from paths passed on the CLI.

Flags (defaults shown):

| Flag | Default | Purpose |
|---|---|---|
| `--mode={eager,compile}` | `eager` | `compile` wraps the model with `torch.compile(backend="tt")` |
| `--warmup=N` | `3` | Untimed warmup iterations before measurement starts |
| `--iters=N` | `20` | Timed iterations per benchmark |
| `--cpu-baseline` | off | Time the same model on CPU for a comparison row |
| `--accuracy` | off | Build a CPU reference and emit `pcc_before_warmup` / `pcc_after_warmup` |
| `--benchmark-json=PATH` | `benchmark_results.json` | Per-test JSON output for CI ingestion |
| `--profiler` | off | Capture a `torch.profiler` Chrome trace per benchmark (open in `chrome://tracing` or Perfetto) |
| `--profile-dir=DIR` | `./profile_data` | Where `--profiler` writes its traces |
| `--llm-batch-size=N` | `32` | Batch size for the LLM decode benchmark |
| `--llm-max-output-tokens=N` | fill 128-slot cache | Override the generate-step count; small values for smoke runs |

Results print as one card per test at session end, with `total_ms`, `iter_mean_ms`, `samples_per_sec` for CNN/prefill workloads and `ttft_ms`, `itl_p50_ms`, `itl_p95_ms`, `tokens_per_sec` for the decode benchmark. The same data lands in `--benchmark-json` keyed on `measurement_name`.

```sh
# typical run on real silicon
pytest tests/python/benchmarks/ --mode=eager --warmup=5 --iters=50

# under sim, with accuracy + cpu baseline for the workloads that can run
pytest tests/python/benchmarks/ --sim --accuracy --cpu-baseline

# capture a torch.profiler trace per benchmark for kernel-level inspection
pytest tests/python/benchmarks/ --profiler                       # → ./profile_data/
pytest tests/python/benchmarks/ --profiler --profile-dir=./traces
```

The runners emit Tracy signposts — named timestamps in the captured trace — at every phase boundary, so a post-run `tt-perf-report` invocation can slice the device-side perf for one phase (e.g. just the decode steps, or just the drain) instead of the whole region. The signposts are emitted unconditionally; if Tracy isn't loaded into the process they become no-ops.

CNN / prefill (`run_benchmark`) emits:

- `warmup_start` / `warmup_end`
- `dispatch_start` — start of the timed iter loop
- `drain_start` — start of the final `_sync` pass
- `end`

LLM generate loop (`run_llm_benchmark`) emits:

- `warmup_start` / `warmup_end`
- `prefill_start` / `prefill_end` (step 0)
- `decode_<i>_start` / `decode_<i>_end` for each step `i`
- `end`

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
