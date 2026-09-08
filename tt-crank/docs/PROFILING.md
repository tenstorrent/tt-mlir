# Profiling tt-crank with Tracy

`tt-crank` uses tt-metal's Tracy profiler. Tracy support is part of every
`tt-crank` build, and the `tracy` CLI comes with the Python package (see the
README). `libtt_crank.so` has Tracy zones marking the hot paths of the compile /
run pipeline, so a trace shows where the time goes.

## Quick start

```sh
# From the tt-mlir root.
tracy -p -r -m pytest -svv tt-crank/tests/python/op_tests/test_elementwise.py
```

Output lands in `.tracy_artifacts/reports/<timestamp>/`:

| File | Description |
|---|---|
| `tracy_profile_log_host.tracy` | Host-side trace, openable in the Tracy GUI |
| `ops_perf_results_<ts>.csv` | Per-op performance results |
| `profile_log_device.csv` | Raw device-side data (silicon only; can be many GB) |

## Tracy CLI options

| Flag | Description |
|---|---|
| `-p` | Only profile explicitly enabled zones (recommended) |
| `-r` | Generate an ops report after the run |
| `--no-device` | Host-only profiling (skip device data) |
| `-o FOLDER` | Output folder for profiler artifacts (default: `.tracy_artifacts/`) |
| `-n NAME` | Append a custom name to the report filename |

`tracy --help` lists everything.

## Host-only profiling

`tracy --no-device` does not write a `.tracy` file on its own. Either open the
Tracy GUI before the run (it captures live), or run `tracy-capture` first:

```sh
# Terminal 1 (tt-mlir root)
./third_party/tt-metal/src/tt-metal/build/tools/profiler/bin/tracy-capture -o output.tracy

# Terminal 2
tracy -p --no-device -m pytest -svv tt-crank/tests/python/op_tests/test_elementwise.py
```

## Turning tt-crank zones off

Configure with `-DTT_CRANK_TRACY_ZONES=OFF`. The CLI keeps working; the trace
just has no `tt_crank::*` zones.

## Adding zones

C++:

```cpp
#include <tracy/Tracy.hpp>

void some_function() {
    ZoneScopedN("tt_crank::some_function");
    // ...
}
```

`ZoneScopedN` is a no-op when `TRACY_ENABLE` is undefined.

Python (phase markers):

```python
import tracy

tracy.signpost("warmup_complete")
```

[`tt-perf-report`](https://github.com/tenstorrent/tt-perf-report) can slice the
ops CSV between two signposts (`--start-signpost` / `--end-signpost`).

## Further reading

- [Tracy Profiler — TT-Metalium docs](https://docs.tenstorrent.com/tt-metal/latest/tt-metalium/tools/tracy_profiler.html)
- [Profiling TT-NN Operations](https://docs.tenstorrent.com/tt-metal/latest/ttnn/ttnn/profiling_ttnn_operations.html)
- [Tracy Profiler GitHub](https://github.com/wolfpld/tracy)
