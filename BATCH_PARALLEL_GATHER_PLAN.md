# RewriteBatchParallelGatherPass — implementation notes

Branch: `hshah/batch-gather-fix`

## Problem

`torch.gather(x, dim=1, index)` on a mesh that shards `x` along batch dim 0
produces a 3-D point-gather decomposition with a global batch iota:

```
start_index_map = [0, 1, 2]   ← dim 0 slot is just iota([0..B-1])
```

Shardy's `InsertExplicitReshardsPass` demands the operand be replicated on
dim 0 (because dim 0 appears in `start_index_map`) and inserts
`sdy.reshard → sdy.all_gather`.  This all_gather sits between the index
construction ops and the gather, breaking `ReoutlineCompositePass`.

## Fix

New pass `RewriteBatchParallelGatherPass` (runs right after `FlattenCompositePass`,
before sharding propagation) detects gather ops inside flattened composite bodies
whose index-concatenation slots are iota pass-throughs on sharded operand axes.
It moves those axes into `operand_batching_dims` / `start_indices_batching_dims`,
removing the corresponding concat slot. Shardy treats batching dims as paired
parallel factors, so no reshard is inserted.

## Files changed

| File | Change |
|---|---|
| `include/ttmlir/Dialect/StableHLO/Transforms/Passes.td` | New `RewriteBatchParallelGatherPass` TableGen def. |
| `lib/Dialect/StableHLO/Transforms/RewriteBatchParallelGather.cpp` | New — pass implementation. |
| `lib/Dialect/StableHLO/Transforms/CMakeLists.txt` | Added `RewriteBatchParallelGather.cpp`. |
| `lib/Dialect/StableHLO/Pipelines/StableHLOPipelines.cpp` | Inserted new pass after `createFlattenCompositePass()`. |
| `lib/Dialect/StableHLO/Transforms/UpdateGlobalToLocalShapes.cpp` | Added early `continue` for `operand_batching_dims` in the GatherOp slice-size update loop to prevent batching-axis slice size from being incorrectly shrunk. |

## Build

```bash
cd /localdev/hshah/tt-mlir
source env/activate
cmake -G Ninja -B build -DCMAKE_CXX_COMPILER_LAUNCHER=ccache
cmake --build build
```

## Test

End-to-end from tt-xla repo (after installing the built wheel):
```bash
cd /localdev/hshah/tt-xla
TTXLA_LOGGER_LEVEL=VERBOSE pytest -svv tests/torch/ops/test_gather.py::test_gather_indices[32-4]
```

Success: no `sdy.all_gather` on data tensor inside manual-computation body,
`stablehlo.composite "tenstorrent.gather"` present after ReoutlineCompositePass,
PCC passes.

## Open risks

1. Shardy propagation of `operand_batching_dims`: if the pinned Shardy version
   lacks a rule, sharding may not propagate and a reshard may still be inserted.
   Fallback: extend `RegisterCustomShardingRulePass`.
2. TTIR legalization rejects non-empty batching dims (`gather_to_embedding_negative.mlir`).
   Safe because `ReoutlineCompositePass` re-outlines the body before conversion.
   If reoutlining fails for any reason, legalization will error loudly.
