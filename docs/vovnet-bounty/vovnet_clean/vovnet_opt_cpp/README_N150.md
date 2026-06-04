# VoVNet standalone — N150 benchmark

`ese_vovnet19b_dw`, bf16, batch=1, `optimization_level=2` (L1-sharded
activations). This is the MLIR→tt-alchemist→standalone-C++ pipeline output.

The 8×8 = 64-core shard layouts in here (`CoreCoord{7,7}`) are why this crashed
on **N300** (56 compute cores → `bank_manager.cpp:430: 64 > 56 L1 banks`).
On **N150** there are 64 compute cores, so those layouts fit — this should run
to completion.

## Prerequisites on the N150 box

- A working **tt-metal / TT-NN dev environment** with `find_package(TT-NN)`
  resolvable (i.e. `TT_METAL_HOME` set and metal built, or a `ttnn-install/`
  dropped next to this file — CMake adds it to the prefix path automatically).
- `clang++` >= 17, `cmake`, `ninja`.
- One Wormhole **N150** visible (`tt-smi` shows it). Single chip, device 0.

## Run

```sh
cd vovnet_opt_cpp
./run
```

`run` does `cmake -G Ninja -B build -DCMAKE_BUILD_TYPE=Release
-DCMAKE_CXX_COMPILER=clang++`, builds the `ttnn-standalone` target, then runs it.

Expected output (two lines):

```
[eager] <ms>/iter, <fps> FPS (b=1)
[trace] <ms>/iter, <fps> FPS (b=1)
```

- **eager** ~ low single-digit FPS (program-launch bound — expected, ignore).
- **trace** is the real number. Reference: reviewer got **~339 FPS** with this
  exact recipe (opt2 + metal trace) on N150.

## Knobs (env vars)

| var | default | meaning |
|---|---|---|
| `BENCH_MODE` | `both` | `eager` \| `trace` \| `both` |
| `BENCH_WARMUP` | `10` | warmup iters (compile + const-eval cache) |
| `BENCH_ITERS` | `50` | timed iters |
| `TRACE_REGION_SIZE` | `134217728` (128 MB) | DRAM trace region. Bump if capture errors with "out of trace region". |

Example, trace only, 200 iters:

```sh
BENCH_MODE=trace BENCH_ITERS=200 ./run
```

## What I changed vs raw codegen output

Two surgical edits to the generated files (everything else is untouched
tt-alchemist output):

1. **`ttnn-precompiled.hpp`** — `traceRegionSize` was `0` (trace impossible).
   Now reserves 128 MB DRAM (env-overridable). Without this, trace capture fails.
2. **`ttnn-standalone.cpp`** — replaced the single-shot `main()` with a
   warmup + eager-timing + metal-trace benchmark loop.

## If the build fails

The eager path uses only stable APIs and should always compile. The **trace**
section uses the public ttnn trace API:

```cpp
ttnn::begin_trace_capture(device, cq);
ttnn::end_trace_capture(device, tid, cq);
ttnn::execute_trace(device, tid, cq, blocking);
ttnn::release_trace(device, tid);
```

If your tt-metal version exposes these under a different name/namespace (they
occasionally move, e.g. `ttnn::operations::trace::*` or different `MeshTraceId`
type), that is the only spot to adjust — the `tid` is already `auto` so the type
won't fight you. As a stopgap, `BENCH_MODE=eager ./run` skips trace entirely and
still confirms the pipeline runs on N150.

## Weights / accuracy

Weights are `ones()` (codegen without exported tensors). Throughput is
unaffected, but this does **not** validate accuracy. For an accuracy check vs
torch-CPU, regenerate with exported tensors (see `../codegen_cpp_vovnet.py`,
add `"export_tensors": True`) and run on the N150.

## Regenerating fresh on the N150 (fallback / version-match)

If the prebuilt C++ here has a tt-metal version mismatch, regenerate it on the
N150 box itself:

```sh
python ../codegen_cpp_vovnet.py     # codegen_cpp, opt2, batch=1 -> ./vovnet_cpp
```

Then re-apply the two edits above (trace region + bench main) to the freshly
generated files and `./run`.
