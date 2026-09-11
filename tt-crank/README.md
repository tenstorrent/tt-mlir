# tt-crank

Compiler & runtime frontend for `tt-mlir`

This is an experimental project which provides a thin frontend layer (enabling easier integration with `tt-mlir`) along with implementation of a torch backend (and possibly other integrations as well).

## Prerequisites

`tt-crank` is a subproject of `tt-mlir` - so everything `tt-mlir` requires is a pre-requisite.

We reuse the same python virtual environment as `tt-mlir`.

## Build & test

Everything runs from the tt-mlir root:

```sh
source env/activate
cmake -G Ninja -B build -DCMAKE_CXX_COMPILER_LAUNCHER=ccache -DTTMLIR_ENABLE_CRANK=ON
cmake --build build
./tt-crank/scripts/test          # C++ unit tests on a device
./tt-crank/scripts/test sim      # … routed through ttsim
```

`-DTTMLIR_ENABLE_CRANK=ON` adds `tt-crank/` to tt-mlir's build graph and auto-enables what it depends on, e.g. `TTMLIR_ENABLE_RUNTIME`, `TTMLIR_ENABLE_OPMODEL`, etc.

To run C++ unit tests you can use the following script:

```sh
./tt-crank/scripts/test
```

### Running through ttsim

Every build stages `tenstorrent/ttsim` next to its SoC descriptor under `build/tt-crank/ttsim_home/` — no separate configure flag. Pass `sim` to `scripts/test` to route the runtime through the simulator:

```sh
./tt-crank/scripts/test sim                                            # run tests via ttsim
./tt-crank/scripts/test sim -- --gtest_filter='EngineCompileTest.*'    # filter tests + ttsim
```

`TT_CRANK_USE_SIMULATOR=1` is the underlying switch, so it can be set directly to use ttsim in any other circumstance (Python extension, manual binary invocation, etc.).

## Python package

We support both editable and wheel installation.

Use:
```sh
./tt-crank/scripts/install-py
```

to install the editable (dev) package.

**NOTE:** the editable installation is required for running tests/benchmarks.

### Wheel

To build a self-contained wheel, run from the `tt-mlir` root:

```sh
pip wheel tt-crank/ --no-build-isolation --no-deps -w dist/
# or
uv build --wheel --no-build-isolation tt-crank/ -o dist/
```

The build requirements come with the dev requirements (installed by `install-py`). `tt-mlir` is configured and built separately in `tt-crank/build_wheel`, so the dev build is untouched. The wheel installs into any Python 3.12 environment with `pip install dist/tt_crank-*.whl`; the target machine needs `sfpi` (`tt-crank-install-sfpi` installs it) and tt-metal's system dependencies.

## Python tests

The Python test suite lives in `tests/python/` and runs with pytest from `tt-crank/`. It imports the compiled `tt_crank` extension, so the Python package has to be installed first (see above).

Some examples of running tests with different options:

```sh
pytest tt-crank/tests/python/                  # run all Python tests
pytest tt-crank/tests/python/ --sim            # route through ttsim
```

## Benchmarks

Benchmarks live under `tests/python/benchmarks/` and run as a pytest target. The benchmark-only CLI flags are registered by `tests/python/benchmarks/conftest.py`, so you have to invoke pytest at-or-below that directory for them to be recognized.

```sh
pytest tt-crank/tests/python/benchmarks/                        # run all benchmarks
pytest tt-crank/tests/python/benchmarks/test_mnist_linear.py    # one benchmark
```

**NOTE:** The `benchmarks/` directory is in `norecursedirs` (in `pytest.ini`), so plain `pytest tests/python` skips it during collection. Pytest will collect them once you run target the `benchmarks/` or it sub-dirs.

Flags (defaults shown):

| Flag | Default | Purpose |
|---|---|---|
| `--mode={eager,compile}` | `eager` | `compile` wraps the model with `torch.compile(backend="tt")` |
| `--warmup=N` | `3` | Untimed warmup iterations before measurement starts |
| `--iters=N` | `20` | Timed iterations per benchmark |
| `--cpu-baseline` | off | Time the same model on CPU for a comparison row |
| `--accuracy` | off | Build a CPU reference and emit `pcc_before_warmup` / `pcc_after_warmup` |
| `--benchmark-json=PATH` | `.data/benchmark_results.json` | Per-test JSON output for CI ingestion |
| `--profiler` | off | Capture a `torch.profiler` Chrome trace per benchmark (open in `chrome://tracing` or Perfetto) |
| `--profile-dir=DIR` | `.data/profile_data` | Where `--profiler` writes its traces |
| `--llm-batch-size=N` | `32` | Batch size for the LLM decode benchmark |
| `--llm-max-output-tokens=N` | fill 128-slot cache | Override the generate-step count; small values for smoke runs |

Results print as one card per test at session end, with `total_ms`, `iter_mean_ms`, `samples_per_sec` for CNN/prefill workloads and `ttft_ms`, `itl_p50_ms`, `itl_p95_ms`, `tokens_per_sec` for the decode benchmark. The same data lands in `--benchmark-json` keyed on `measurement_name`.

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

## Logging

Logging uses [`tt-logger`](https://github.com/tenstorrent/tt-logger) (spdlog-based). Control it at runtime via environment variables:

| Variable | Values | Default |
|---|---|---|
| `TT_LOGGER_LEVEL` | `debug`, `info`, `warn`, `error`, `critical`, `off` | `info` |
| `TT_LOGGER_FILE` | path to log file | stdout |
| `TT_LOGGER_TYPES` | comma-separated logger names, or `All` | all loggers |

```sh
# Enable debug logging
TT_LOGGER_LEVEL=debug ./tt-crank/scripts/test sim

# Debug output to a file
TT_LOGGER_LEVEL=debug TT_LOGGER_FILE=/tmp/crank.log ./tt-crank/scripts/test sim

# Filter to specific subsystems
TT_LOGGER_LEVEL=debug TT_LOGGER_TYPES=TTNN,Op ./tt-crank/scripts/test sim

# Show only tt-crank logs (suppress tt-mlir/tt-metal noise)
TT_LOGGER_LEVEL=debug TT_LOGGER_TYPES=Always ./tt-crank/scripts/test sim
```

`debug` log calls are compiled in only for `Debug` and `RelWithDebInfo` builds (the tt-mlir build's `CMAKE_BUILD_TYPE`) — they are no-ops in `Release`.
