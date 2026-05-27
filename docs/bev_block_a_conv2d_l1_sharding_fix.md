# BEV Block A — Conv2d Effective Sharding: Issues, Root Causes, and Fixes

## Overview

Block A (`block_A_deformed_backbone.onnx`) of the BEV model is a deep convolutional
backbone consisting of many consecutive Conv2d operations (3×3 and 1×1 kernels),
arranged in a height-sharded L1 (`L1Full`) compute chain.  The Tenstorrent compiler
optimizer (MLA + L1SpillManagement) is responsible for keeping as many of these
operations' results in L1 as possible.  When an output cannot remain in L1, it is
spilled to DRAM and must be re-sharded back to L1 before the next consumer reads it.
Each such round-trip reduces throughput and consumes extra DRAM bandwidth.

This document captures the chain of bugs discovered, their root causes, the fixes
applied, and the measured improvements across the compiler metrics and device FPS.

---

## Baseline State (before fixes)

Config: `opt_level_2 + fp16b + HiFi3 + fp32_dest_acc + trace enabled` on WH N150.

| Metric                     | Value |
|----------------------------|-------|
| total_ops                  | 1198  |
| total_shardable_ops        | 408   |
| effectively_sharded_ops    | 104   |
| effectively_sharded_%      | 25.5% |
| sharded_and_spilled_ops    | 108   |
| dram_spilled_ops           | 100   |
| FPS (10 × timed runs)      | ~2.71 |

Only **25.5%** of shardable ops were effectively sharded — meaning most L1-eligible
Conv2d outputs were being spilled to DRAM, costing significant bandwidth.

---

## Issue 1 — Bounce Spills from ToMemoryConfigOp Chains

### Symptom

The MLA (Memory Layout Analysis) pass and `OperationValidationAndFallback` sometimes
insert layouts for intermediate buffers that create DRAM → L1\_sharded → DRAM
round-trips ("bounce spills").  These arise in patterns such as:

```
%dram1 = to_memory_config(%src,  DRAM_interleaved)
%l1sh  = to_memory_config(%dram1, L1_sharded)       ← transient L1 hop
%dram2 = to_memory_config(%l1sh,  DRAM_interleaved)  ← back to DRAM
```

The transient L1 shard serves no compute purpose — it is only there because
`OperationValidationAndFallback` propagated a sharded layout for an intermediate
value, and the downstream consumer reverted it to DRAM.  At runtime these become
extra PCIe copies with no benefit.

A second pattern appears after `TTNNDecomposeLayouts`:

```
%l1sh = ... → L1_sharded
%dram = to_memory_config(%l1sh, DRAM)
%l1i  = to_memory_config(%dram, L1_interleaved)   ← could read %l1sh directly
```

Here the consumer only needs an L1-interleaved view; it could read the L1-sharded
source directly, bypassing the DRAM hop entirely.

### Root Cause

`ToMemoryConfigOp` had no folder or canonicalization patterns.  Once inserted by
MLA or decomposition passes, these ops were never folded, even when two consecutive
ops were provably redundant.

### Fix

**Files:** `include/ttmlir/Dialect/TTNN/IR/TTNNOps.td`,
           `lib/Dialect/TTNN/IR/TTNNOps.cpp`,
           `lib/Dialect/TTNN/Pipelines/TTNNPipelines.cpp`

**1. Identity folder** (`ToMemoryConfigOp::fold`):
Fold any `to_memory_config` whose input and output types are identical:

```cpp
mlir::OpFoldResult mlir::tt::ttnn::ToMemoryConfigOp::fold(FoldAdaptor) {
  if (getInput().getType() == getResult().getType())
    return getInput();
  return nullptr;
}
```

**2. `FoldConsecutiveToMemoryConfigOps` canonicalization pattern**:
When two consecutive `to_memory_config` ops are chained and the intermediate is
L1-sharded with no other compute users, collapse them into a single op that
converts the original source directly to the final destination:

```
%src   = ... → L1_sharded
%dram  = to_memory_config(%src, DRAM)
%final = to_memory_config(%dram, L1_int)
→  %final = to_memory_config(%src, L1_int)   [%dram erased]
```

**3. `BypassDRAMForL1InterleavedConsumers` canonicalization pattern**:
When a DRAM `to_memory_config` sourced from L1-sharded data has one or more
L1-interleaved consumers, reroute those consumers to read the L1-sharded source
directly, bypassing the DRAM hop:

```
%l1sh  = ... → L1_sharded
%dram  = to_memory_config(%l1sh, DRAM)
%l1i_a = to_memory_config(%dram, L1_int_a)
%l1i_b = to_memory_config(%dram, L1_int_b)
→  %l1i_a = to_memory_config(%l1sh, L1_int_a)
   %l1i_b = to_memory_config(%l1sh, L1_int_b)
   [%dram erased when no other compute users remain]
```

**4. Two new canonicalization passes added to the pipeline** (`TTNNPipelines.cpp`):
- After `OperationValidationAndFallback` inside the optimizer pass manager, to fold
  bounce spills inserted during fallback.
- After `TTNNDecomposeLayouts`, to fold L1\_sharded → DRAM → L1\_interleaved patterns
  emitted during layout decomposition.

---

## Issue 2 — False-Positive CB_ZONE_EVICT Evictions (56 per compile)

### Symptom

`L1SpillManagement` logged 56 `CB_ZONE_EVICT` events per Block A compile.  Each
event evicted a live L1 tensor immediately before a large-CB Conv2d, causing the
predecessor to spill to DRAM and later be resharded back to L1 for the consumer.
This inflated `sharded_and_spilled_ops` from an expected ~24 to 108.

### Root Cause — Inverted CB Zone Direction

The `SumL1MemoryTracker` address simulator allocates addresses **top-down virtual**:
the first tensor allocated receives the highest virtual address
(`l1BudgetPerCore - size`), and subsequent allocations fill down toward 0.

The CB_ZONE_EVICT block was introduced on the assumption that TTNN allocates L1
tensors **bottom-up physically** (from `l1_unreserved_base` upward).  Under this
assumption the virtual-to-physical mapping is inverted:

```
HIGH virtual ↔ LOW physical  (near l1_unreserved_base = near CB zone)  [WRONG MODEL]
```

**However**, the tt-metal L1 allocator actually uses **top-down physical** allocation.
This is proved by `tt_metal/impl/buffers/buffer.cpp` line 289:

```cpp
bottom_up_(bottom_up.value_or(this->is_dram()))
// For L1 buffers: is_dram() == false  →  bottom_up = false  →  TOP-DOWN
```

`FreeListOpt::allocate(size, bottom_up=false)` scans the free list from the highest
address downward and allocates from the **top** of each free block.  The first L1
tensor allocated lands at `worker_l1_size - size` (near the top of L1 physical
address space), not near `l1_unreserved_base`.

The **correct** virtual-to-physical mapping is therefore **same-direction**:

```
physical_address = l1_unreserved_base + virtual_address
```

| Virtual address | Physical address     | Distance from CB zone |
|-----------------|----------------------|-----------------------|
| HIGH (≈1.3 MB)  | HIGH (≈ worker_l1_size - size) | FAR — SAFE  |
| LOW  (≈ 0)      | LOW (≈ l1_unreserved_base)     | NEAR — DANGEROUS |

CBs grow **bottom-up** from `l1_unreserved_base`.  A tensor is in the CB zone if
its physical address < `l1_unreserved_base + cbPeakUsage`, i.e. its **virtual**
address < `cbPeakUsage`.

The CB_ZONE_EVICT block checked the **opposite** condition:

```cpp
// WRONG: evicts HIGH-virtual tensors (= HIGH physical = SAFE)
uint64_t cbVT = l1BudgetPerCore - cbPeakUsage;
for (Value victim : memoryTracker.getValuesAboveVirtualThreshold(cbVT)) {
  evictValue(victim, ...);
}
```

`HIGH virtual > cbVT = l1BudgetPerCore - cbPeakUsage` describes tensors that are
**far from** the CB zone at runtime — not inside it.  The code was systematically
evicting safe tensors.

### Why No Runtime Crashes?

The existing `wouldCBsOverlapTensors` + `evictForCBOverlap` path uses
`getLowestOccupiedAddress()` (minimum virtual address = minimum physical address =
most dangerous tensor) to detect real CB conflicts.  Under the correct mapping this
check is accurate and sufficient.  The CB_ZONE_EVICT loop was a redundant, inverted
pre-emptive check that only produced false positives.

The 56 false-positive evictions caused no runtime crashes because the actual tensors
were at HIGH physical addresses (safely above the CB zone), and `wouldCBsOverlapTensors`
correctly determined no real overlap existed after the unnecessary evictions.

### Diagnostic Evidence

To confirm the false-positive nature, a diagnostic was added inside
`tryReduceConv2dCBForZoneEvict` to track the minimum achievable `cbPeakUsage` for
each trigger op:

| Trigger type   | Count | Victim physical addr | Min achievable cbPeak | Verdict     |
|----------------|-------|----------------------|-----------------------|-------------|
| 1×1 conv2d     | ~18   | 2 KB – 62 KB (virtual HIGH) | Fixed (act_block_h irrelevant) | False positive |
| 3×3 conv2d     | ~38   | 2 KB – 62 KB (virtual HIGH) | 200 KB – 400 KB >> victim | False positive |

Under the **wrong model** the victims at HIGH virtual appeared to be at LOW physical
(inside the CB zone).  Under the **correct model** they are at HIGH physical (safe).

### Fix

**Files:** `lib/Dialect/TTNN/Analysis/L1SpillManagement.cpp`,
           `include/ttmlir/Dialect/TTNN/Analysis/L1SpillManagement.h`

Remove the CB_ZONE_EVICT block from `ensureFitsL1` and the `tryReduceConv2dCBForZoneEvict`
helper that was called only from it.  Real CB conflicts are still caught by the
existing and correct `wouldCBsOverlapTensors` → `handleFragmentation` →
`evictForCBOverlap` path.

```cpp
// REMOVED from ensureFitsL1:
if (cbPeakUsage > l1DeadZone && cbPeakUsage <= l1BudgetPerCore) {
  tryReduceConv2dCBForZoneEvict(op, cbPeakUsage);
  uint64_t cbVT = l1BudgetPerCore - cbPeakUsage;
  for (Value victim : memoryTracker.getValuesAboveVirtualThreshold(cbVT)) {
    ...evictValue(victim, pos, data);
  }
  ...
}
```

The surviving checks (`wouldCBsOverlapTensors`, `DRAM_OP_CB_CHECK`) are correct
under the same-direction mapping and remain untouched.

---

## Issue 3 — OOM Recovery for L1Full Conv2d Under Memory Pressure

### Symptom

When `L1SpillManagement` encounters a Conv2d with a very large CB footprint
(e.g. HiFi3 + fp32\_dest\_acc → 350–600 KB CBs), the `handleOOM` path demotes
the op to DRAM, triggering the costly L1\_sharded → DRAM → L1\_sharded bounce for
the entire chain.

### Fix

**Files:** `lib/Dialect/TTNN/Analysis/L1SpillManagement.cpp`,
           `include/ttmlir/Dialect/TTNN/Analysis/L1SpillManagement.h`

Added `tryReduceConv2dActBlockH`: before falling back to DRAM demotion in the OOM
path, try progressively smaller `act_block_h` values (from 1024 down to 32 in steps
of 32) for L1Full Conv2d ops.  A smaller `act_block_h` reduces the CB peak usage
at the cost of slightly lower throughput — but keeps the op in L1, avoiding the
DRAM round-trip entirely.

```
kActBlockHSearchSpace = {1024, 992, ..., 64, 32}
```

If a candidate `act_block_h` passes validation and `ensureFitsL1`, the op's
`Conv2dConfigAttr` is patched in-place and the function returns `true` (handled).
If no candidate fits, the function returns `false` and the normal eviction path
takes over.

---

## File Changes Summary

### `include/ttmlir/Dialect/TTNN/IR/TTNNOps.td`
- Added `hasFolder = 1` and `hasCanonicalizer = 1` to `ToMemoryConfigOp`

### `lib/Dialect/TTNN/IR/TTNNOps.cpp`
- `ToMemoryConfigOp::fold`: identity fold (same input/output type)
- `FoldConsecutiveToMemoryConfigOps`: collapse two chained `to_memory_config` ops when intermediate is L1-sharded with no other compute users
- `BypassDRAMForL1InterleavedConsumers`: reroute L1-interleaved consumers of a DRAM op to read the L1-sharded source directly
- `ToMemoryConfigOp::getCanonicalizationPatterns`: register both patterns

### `lib/Dialect/TTNN/Pipelines/TTNNPipelines.cpp`
- Canonicalizer pass after `OperationValidationAndFallback` (folds bounce spills from fallback)
- Canonicalizer pass after `TTNNDecomposeLayouts` (folds L1_sharded→DRAM→L1_interleaved patterns)

### `lib/Dialect/TTNN/Transforms/TTNNDecomposeLayouts.cpp`
- Minor whitespace fix (trailing blank line before closing brace)

### `include/ttmlir/Dialect/TTNN/Analysis/L1SpillManagement.h`
- Removed `tryReduceConv2dCBForZoneEvict` declaration (function deleted)
- Added `tryReduceConv2dActBlockH` declaration (new OOM recovery helper)

### `lib/Dialect/TTNN/Analysis/L1SpillManagement.cpp`
- **Removed** CB_ZONE_EVICT block from `ensureFitsL1` (~38 lines)
- **Removed** `tryReduceConv2dCBForZoneEvict` function (~115 lines)
- **Added** `tryReduceConv2dActBlockH` function (~95 lines, OOM path only)
- Added `#include "ttmlir/Dialect/TTNN/Utils/Conv2dConfigParams.h"`

---

## Results After All Fixes

Config: `opt_level_2 + fp16b + HiFi3 + fp32_dest_acc + trace enabled` on WH N150.

| Metric                     | Before | After  | Delta      |
|----------------------------|--------|--------|------------|
| total_ops                  | 1198   | 962    | −236       |
| total_shardable_ops        | 408    | 408    | —          |
| effectively_sharded_ops    | 104    | **208**| **+104 (2×)** |
| effectively_sharded_%      | 25.5%  | **51.0%** | **+25.5 pp** |
| sharded_and_spilled_ops    | 108    | **24** | **−84**    |
| dram_spilled_ops           | 100    | 80     | −20        |
| spilled_%                  | 24.5%  | 19.6%  | −4.9 pp    |
| FPS (10 × timed runs)      | 2.71   | 2.65   | within noise (±0.15) |

### Interpretation

- **`effectively_sharded_ops` doubled (104→208)**: 104 additional Conv2d outputs now
  stay resident in L1 all the way through to their consumer, with no DRAM spill.

- **`sharded_and_spilled_ops` dropped 78% (108→24)**: ops that were L1-sharded by
  MLA but immediately forced to DRAM by spurious CB_ZONE_EVICT evictions are now
  keeping their L1 layout end-to-end.

- **`total_ops` reduced by 236**: each eliminated bounce spill removes at least two
  `to_memory_config` ops from the TTNN IR (spill + reshard).

- **FPS (2.65 vs 2.71)**: within measurement noise (historical range 2.57–2.82 FPS
  across prior fixes on the same benchmark).  The device-side latency is dominated
  by the Conv2d kernels themselves; the eliminated DRAM hops reduce PCIe traffic but
  the improvement is obscured by per-run variance.

---

## Key Takeaways

1. **The SumL1MemoryTracker virtual address maps 1:1 to physical offset** from
   `l1_unreserved_base` (`physical = l1_unreserved_base + virtual`), because both the
   simulator and tt-metal allocate **top-down** (from high addresses toward
   `l1_unreserved_base`).  There is NO inversion.

2. **CB zone tensors have LOW virtual addresses** (< `cbPeakUsage`), not high.
   `wouldCBsOverlapTensors` (which uses `getLowestOccupiedAddress()`) is the correct
   CB guard.  CB_ZONE_EVICT was checking the wrong direction.

3. **Consecutive `to_memory_config` chains are common** after layout analysis.  A
   canonicalization pass that folds them should be inserted at every point in the
   pipeline where they are created: after MLA fallback and after decomposition.

4. **`act_block_h` reduction is a viable alternative to DRAM demotion** for L1Full
   Conv2d ops under memory pressure.  It trades a small amount of throughput for the
   much larger savings of avoiding a DRAM round-trip.
