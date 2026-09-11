# Profiling tt-kurbla with Tracy

tt-kurbla bundles tt-metal's Tracy profiler. Every build ships:

- the `tracy` console script (a thin wrapper around tt-metal's `tools/tracy`
  module — orchestrates a profiling run and writes the per-op CSV report),
- the `tracy-capture` binary (headless Tracy capture),
- `tracy-csvexport` (export `.tracy` files to CSV).

`libtt_kurbla.so` is also instrumented with Tracy zones by default. The zones
mark the hot paths in our compile / run pipeline so a recorded trace shows
where time is actually spent.

## Quick start

```bash
source venv/activate && source third_party/tt-mlir/env/activate

# Build (release recommended — zones are inlined at non-debug optimization).
./scripts/build release
./scripts/install-py

# Run any tt-kurbla workload under the tracy CLI.
tracy -p -r --sync-host-device \
    -m pytest -svv tests/python/op_tests/test_elementwise.py
```

Output lands in `.tracy_artifacts/reports/<timestamp>/`:

| File | Description |
|---|---|
| `tracy_profile_log_host.tracy` | Host-side trace, openable in the Tracy GUI |
| `ops_perf_results_<ts>.csv` | Per-op performance results (op name, duration, etc.) |
| `profile_log_device.csv` | Raw device-side data (silicon only; can be many GB) |

## Tracy CLI options

| Flag | Description |
|---|---|
| `-p` | Only profile explicitly enabled zones (recommended) |
| `-r` | Generate an ops report after the run |
| `--sync-host-device` | Synchronize host and device timelines (requires silicon) |
| `--no-device` | Host-only profiling (skip device data) |
| `-o FOLDER` | Output folder for profiler artifacts (default: `.tracy_artifacts/`) |
| `-n NAME` | Append a custom name to the report filename |

`tracy --help` lists everything.

## Toggling tt-kurbla zones off

For a clean release build with zero Tracy hooks in `libtt_kurbla.so`,
configure with `-DTT_KURBLA_TRACY_ZONES=OFF`. The CLI keeps working — it just
won't show `tt_kurbla::*` zones in the recorded trace.

```bash
cmake --preset release -DTT_KURBLA_TRACY_ZONES=OFF
cmake --build --preset release
```

`./scripts/build` doesn't pass extra `-D…` arguments today — call `cmake`
directly when you need to flip the flag.

## Host-only profiling

`tracy --no-device` skips the device timeline but currently does not write a
`.tracy` file on its own. Two workarounds (the same as tt-xla's):

### Workaround 1: connect with the Tracy GUI live

Open the Tracy GUI **before** starting the run. It connects to the profiler
in real time and captures the trace directly. Then run:

```bash
tracy -p --no-device -m pytest -svv tests/python/op_tests/test_elementwise.py
```

### Workaround 2: run `tracy-capture` in a second terminal

`tracy-capture` is the headless equivalent of the GUI. Start it first:

```bash
# Terminal 1: start the capture listener
./build/third_party/tt-mlir-install/bin/tracy-capture -o output.tracy
```

```bash
# Terminal 2: run the workload
tracy -p --no-device -m pytest -svv tests/python/op_tests/test_elementwise.py
```

`tracy-capture -h` lists the relevant flags (`-o` output path, `-a` address,
`-p` port, `-f` force overwrite, `-s` stop after N seconds).

## Adding more zones

To add a new zone in C++ code:

```cpp
#include <tracy/Tracy.hpp>

void some_function() {
    ZoneScopedN("tt_kurbla::some_function");
    // ...
}
```

`ZoneScopedN` is a no-op when `TRACY_ENABLE` is undefined, so it is safe to
add unconditionally. Use short, stable names — they show up in every trace.

From Python, use signposts to mark phases:

```python
import tracy

tracy.signpost("warmup_complete")
# ...
tracy.signpost("decode_start")
```

`tt-perf-report` (https://github.com/tenstorrent/tt-perf-report) can slice
the ops CSV between two signposts with `--start-signpost` / `--end-signpost`.

## Further reading

- [Tracy Profiler — TT-Metalium docs](https://docs.tenstorrent.com/tt-metal/latest/tt-metalium/tools/tracy_profiler.html)
- [Profiling TT-NN Operations](https://docs.tenstorrent.com/tt-metal/latest/ttnn/ttnn/profiling_ttnn_operations.html)
- [Tracy Profiler GitHub](https://github.com/wolfpld/tracy)
