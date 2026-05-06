# test_graph_tracker_race

Race reproducer for `tt::tt_metal::GraphTracker`. Two threads share the
process and the singleton:

- compile thread: `op_model::ttnn::executeConstraintQuery` →
  `ScopedGraphCapture::push_processor` / `pop_processor`
- runtime thread: `tt::runtime::submit` → ttnn ops fire
  `track_function_start` (iterates `processors`)

Without serialisation the runtime thread can iterate `processors` while
the compile thread is mid-mutation.

## Build (TSan)

```bash
cmake -G Ninja -B build \
  -DCMAKE_BUILD_TYPE=TSan \
  -DCMAKE_CXX_COMPILER_LAUNCHER=ccache \
  -DTTMLIR_ENABLE_OPMODEL=ON \
  -DTTMLIR_ENABLE_RUNTIME=ON \
  -DTT_RUNTIME_ENABLE_TTNN=ON \
  -DTTMLIR_ENABLE_RUNTIME_TESTS=ON

cmake --build build --target test_graph_tracker_race
```

The first TSan build of `tt-metal` is from-scratch and slow (~30–60 min).

## Compile a flatbuffer

`optimization-level=2` is required — it enables OpModel constraint queries
(`executeConstraintQuery`) which are the compile-side half of the race.
Without it the optimizer never pushes a GraphProcessor and the race does
not trigger.

```bash
ttrt query --save-artifacts
SD=$(pwd)/ttrt-artifacts/system_desc.ttsys
ttmlir-opt --ttir-to-ttnn-backend-pipeline="system-desc-path=$SD optimization-level=2" \
  -o /tmp/phi.mlir \
  test/ttmlir/models/single_blocks_and_layers/phi_1_decode_layer.mlir
ttmlir-translate --ttnn-to-flatbuffer -o /tmp/phi.ttnn /tmp/phi.mlir
```

## Run

```bash
TTMLIR_RACE_TEST_FB=/tmp/phi.ttnn \
TTMLIR_RACE_DURATION_SECONDS=30 \
TSAN_OPTIONS="suppressions=$(pwd)/runtime/test/ttnn/gtest/test_graph_tracker_race.tsan.supp:halt_on_error=0:history_size=7" \
  ./build/runtime/test/ttnn/gtest/test_graph_tracker_race
```

Look for TSan reports referencing `tt::tt_metal::GraphTracker::push_processor`,
`pop_processor`, or the iteration in `track_function_start`. Reports
unrelated to `processors` (fast-dispatch threads, ARC daemons, etc.)
should be added to the suppressions file iteratively.
