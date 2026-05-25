# Fix: Height-Sharded Upsample Output Spilled to DRAM Before Width-Trim Slice

## Overview

This document describes a bug in the greedy memory-layout optimizer that caused
bilinear-upsample operations producing `HEIGHT_SHARDED` L1 output to be
immediately spilled to DRAM when their output was consumed by a width-trimming
`slice_static` operation. The fix landed in two files:

- `lib/Dialect/TTNN/Analysis/OpRules/DataMovementRules.cpp` — `SliceRuleBook::getOutputHints`
- `lib/Dialect/TTNN/Analysis/L1SpillManagement.cpp` — DRAM `ToLayoutOp` handling

---

## Symptom

In models that apply a bilinear upsample followed by a slice that trims only the
last dimension (e.g. 96 → 80 columns), the `TTNNCollectPerfMetrics` pass
reported `is_spilled_to_dram: true` for the upsample even though its output
tensor was `HEIGHT_SHARDED` in L1.

Example from an EVO50 FPN block:

```
// BEFORE FIX — spurious DRAM spill
%4 = "ttnn.upsample"(%3)  → tensor<1x6x26x96xbf16, HEIGHT_SHARDED L1>
%5 = "ttnn.to_memory_config"(%4)  → tensor<1x6x26x96xbf16, DRAM>  ← spill
%6 = "ttnn.slice_static"(%5) <{ends = [1,6,26,80]}>               ← DRAM input

// AFTER FIX — no spill
%4 = "ttnn.upsample"(%3)  → tensor<1x6x26x96xbf16, HEIGHT_SHARDED L1>
%5 = "ttnn.slice_static"(%4) <{ends = [1,6,26,80]}>               ← direct HS RM
```

---

## Background: Key Compiler Components

### `UpsampleRuleBook::getOutputHints` (bilinear)

For bilinear upsample, the rule book bypasses `legalConfigs` entirely and
constructs the exact `HEIGHT_SHARDED ROW_MAJOR` layout that tt-metal's
`compute_bilinear_autoshard_memory_config` will produce at runtime. This means
upsample always gets an `HS RM` output layout regardless of the
`rowMajorEnabled` optimizer flag.

### `SliceStaticOp::getOpConstraints` — analytical HS RM path

`SliceStaticOp` has a special analytical path for width-trim slices with
`HEIGHT_SHARDED ROW_MAJOR` input. When the output layout hint is `nullptr`
(i.e. a NULL `OpConfig`), the constraint function produces an `HS RM` output
whose grid exactly matches the input, and the last dimension is trimmed
accordingly. This avoids a full OpModel evaluation.

The trigger condition inside `getOpConstraints` is:

```cpp
bool isHSRM = isHeightSharded(inputLayout) && isRowMajor(inputLayout);
if (isHSRM && (!outputLayout || outputIsHS)) {
    // Analytical path: produce same-grid HS RM output, trimmed last dim.
}
```

### `rowMajorEnabled` flag (default: `false`)

`LegalOpLayoutAnalysis::fillTTNNLayoutAttrs` skips `ROW_MAJOR` layouts for all
ops when `rowMajorEnabled=false`. This is the default in the greedy optimizer
pipeline. As a consequence, `legalConfigs[slice]` contains **no** `HS RM`
entries, even when the preceding upsample produces `HS RM` output.

### NULL `OpConfig` hint

`OpConfig(TTNNLayoutAttr())` represents a hint with a null output layout. When
the optimizer's `evaluateHint` processes this hint for a slice op with an `HS
RM` input beam candidate, it calls `validateOperation` with `outputLayout =
nullptr`. This satisfies the analytical-path trigger condition in
`getOpConstraints`, which then returns the correct `HS RM` output.

---

## Root Cause

### Bug 1 — `SliceRuleBook::getOutputHints` missing the NULL hint

Before the fix, `getOutputHints` for a width-trim slice looked like:

```cpp
if (isWidthTrimSliceStatic(op)) {
    OutputHints result;
    // Search legalConfigs for HS RM entries
    for (const auto &cfg : legalConfigs) {
        if (!cfg.outputLayout) continue;
        auto ml = cfg.outputLayout.getMemLayout();
        if (ml && ml.getValue() == TensorMemoryLayout::HeightSharded &&
            cfg.outputLayout.getLayout() == Layout::RowMajor) {
            result.hints.push_back(cfg);        // ← never populated when rowMajorEnabled=false
        }
    }
    if (!result.hints.empty()) {                // ← guard always false
        // populate fallbacks and return
        return result;
    }
}
// Falls through to nonShardedOutputHints — no HS RM output tried
return layout_filter_utils::nonShardedOutputHints(legalConfigs);
```

When `rowMajorEnabled=false` (the default), `legalConfigs` contains no `HS RM`
entries, so `result.hints` remained empty, the `if (!result.hints.empty())`
guard was never entered, and the function fell through to
`nonShardedOutputHints`. This function returns **only non-sharded DRAM hints**
— it never emits a NULL hint, so the analytical path in `getOpConstraints` was
never triggered.

The greedy optimizer then tried to reconcile the upsample's `HS RM` input
candidate against non-sharded/DRAM output hints. The validation failed
(`QUERY_OP_CONSTRAINTS` errors for a sharded input against an interleaved
output config), so the beam shifted to a DRAM config for the upsample, causing
`L1SpillManagement` to insert a `ToLayoutOp(upsample_output → DRAM)`.

### Bug 2 — `L1SpillManagement` spuriously evicting on DRAM `ToLayoutOp`

When `L1SpillManagement` encountered a pre-decomposition `ToLayoutOp` whose
annotated output was `DRAM`, it would call the OpModel validator with that op.
Because the input to the `ToLayoutOp` was `HEIGHT_SHARDED`, the OpModel
returned `NOT_IMPLEMENTED`. The spill manager interpreted this as a general
failure and called `evictAllFromL1`, clearing every L1 tensor from the tracker
unnecessarily. This amplified the spill effect, degrading metrics for ops well
beyond the upsample itself.

---

## Fix

### Fix 1 — `SliceRuleBook::getOutputHints` (`DataMovementRules.cpp`)

Add the NULL `OpConfig` as the **first primary hint** unconditionally for
width-trim slices. Any `HS RM` entries found in `legalConfigs` are appended
after it. Non-sharded configs become fallbacks.

```cpp
OutputHints
SliceRuleBook::getOutputHints(Operation *op,
                              const std::vector<OpConfig> &legalConfigs) const {
  if (isWidthTrimSliceStatic(op)) {
    OutputHints result;
    // NULL hint: SliceStaticOp::getOpConstraints analytical path triggers on
    // (isHSRM && !outputLayout) and produces HS RM output from the input grid.
    result.hints.push_back(OpConfig(TTNNLayoutAttr()));
    for (const auto &cfg : legalConfigs) {
      if (!cfg.outputLayout) continue;
      auto ml = cfg.outputLayout.getMemLayout();
      if (ml && ml.getValue() == TensorMemoryLayout::HeightSharded &&
          cfg.outputLayout.getLayout() == Layout::RowMajor) {
        result.hints.push_back(cfg);
      }
    }
    // Non-sharded configs are fallbacks.
    for (const auto &cfg : legalConfigs) {
      if (!cfg.outputLayout) { result.fallbackHints.push_back(cfg); continue; }
      auto ml = cfg.outputLayout.getMemLayout();
      if (!ml || !isShardedMemoryLayout(ml.getValue()))
        result.fallbackHints.push_back(cfg);
    }
    return result;
  }
  return layout_filter_utils::nonShardedOutputHints(legalConfigs);
}
```

**Why the NULL hint works**: When `evaluateHint` processes `OpConfig(nullptr)`
for a beam candidate whose input is `HS RM`, it calls `validateOperation` with
`outputLayout = nullptr`. Inside `SliceStaticOp::getOpConstraints` this
satisfies `isHSRM && !outputLayout`, triggering the analytical path that returns
the correct `HS RM` output layout (same grid, last dim trimmed). The optimizer
then scores this `HS RM` output highly and selects it, eliminating the spill.

The removed `if (!result.hints.empty())` guard was the critical mistake: it
prevented any early return when `legalConfigs` had no `HS RM` entries, even
though the NULL hint alone is sufficient.

### Fix 2 — `L1SpillManagement.cpp` DRAM `ToLayoutOp` skip

Add an explicit `continue` for `ToLayoutOp` instances annotated with DRAM
output, bypassing the OpModel validation that would return `NOT_IMPLEMENTED` for
an `HS` input.

```cpp
if (isDRAMToLayoutOp(op)) {
    // Pre-decomposition OpModel is inaccurate for HS→DRAM conversions.
    // The DRAM output has no L1 footprint; no spill accounting is needed.
    TTMLIR_DEBUG(..., "DRAM_TOLAYOUT: skipping validation");
    continue;
}
```

This prevents the false `NOT_IMPLEMENTED` → `evictAllFromL1` cascade that
degraded sharding metrics for ops unrelated to the upsample→slice chain.

---

## Execution Flow: Before and After

### Before (buggy path)

```
UpsampleRuleBook::getOutputHints
  └─ constructs HS RM layout directly (correct)
     → beam[0] = HS RM (39 cores × 4×96 bf16)

SliceRuleBook::getOutputHints
  ├─ searches legalConfigs for HS RM → none found (rowMajorEnabled=false)
  ├─ result.hints.empty() → guard blocks early return
  └─ returns nonShardedOutputHints (DRAM / interleaved only)

evaluateHint(slice, beam[0]=HS_RM_input, hint=DRAM)
  └─ SliceStaticOp::getOpConstraints(HS_RM_input, DRAM_output)
     └─ NOT analytical path → OpModel query → fails / wrong result
        → beam[0] rejected; optimizer selects DRAM output for upsample

applyToIR:
  └─ inserts ToLayoutOp(upsample → DRAM)

L1SpillManagement:
  └─ encounters ToLayoutOp(HS_input → DRAM)
     └─ OpModel returns NOT_IMPLEMENTED
        └─ evictAllFromL1() ← spurious cascade
```

### After (fixed path)

```
UpsampleRuleBook::getOutputHints
  └─ constructs HS RM layout directly
     → beam[0] = HS RM (39 cores × 4×96 bf16)

SliceRuleBook::getOutputHints
  ├─ isWidthTrimSliceStatic → true
  ├─ result.hints[0] = OpConfig(nullptr)    ← NULL hint added unconditionally
  └─ returns immediately with NULL primary + non-sharded fallbacks

evaluateHint(slice, beam[0]=HS_RM_input, hint=nullptr)
  └─ SliceStaticOp::getOpConstraints(HS_RM_input, outputLayout=nullptr)
     └─ isHSRM && !outputLayout → analytical path
        └─ returns HS RM output (same 39-core grid, last dim 96→80)
           → beam[0] accepted; upsample keeps HS RM output

applyToIR:
  └─ no ToLayoutOp inserted between upsample and slice

L1SpillManagement:
  └─ no DRAM ToLayoutOp encountered → no spurious eviction
```

---

## Verification

After applying both fixes and rebuilding `libTTMLIRCompiler.so`, running the
EVO50 FPN benchmark produced the following perf metrics (vs. before):

| Metric | Before | After |
|--------|--------|-------|
| `sharded_and_spilled_ops` | 4 | **0** |
| `effectively_sharded_ops` | 14 | **18** |
| `sharded_percentage` | 24.6% | **31.6%** |

The TTNN IR confirmed no `to_memory_config → DRAM` between any upsample and its
downstream slice:

```
// AFTER FIX — both width-trim upsamples
%4 = "ttnn.upsample"(...)  → tensor<1x6x26x96xbf16, #ttnn_layout24>   HS L1
%5 = "ttnn.slice_static"(%4) <{ends=[1,6,26,80]}>  → tensor<..., #ttnn_layout25>  HS L1

%17 = "ttnn.upsample"(...)  → tensor<1x12x52x96xbf16, #ttnn_layout30>  HS L1
%18 = "ttnn.slice_static"(%17) <{ends=[1,12,52,80]}>  → tensor<..., #ttnn_layout31>  HS L1
```

---

## Affected Files

| File | Change |
|------|--------|
| `lib/Dialect/TTNN/Analysis/OpRules/DataMovementRules.cpp` | Add NULL hint in `SliceRuleBook::getOutputHints`; remove broken `if (!result.hints.empty())` guard |
| `lib/Dialect/TTNN/Analysis/L1SpillManagement.cpp` | Skip OpModel validation for DRAM `ToLayoutOp` to prevent spurious `evictAllFromL1` |

---

## Related Issues

- tt-metal height-sharded `SliceRmShardedWidthTrimProgramFactory`:
  https://github.com/tenstorrent/tt-metal/issues/39074
- tt-metal concat hang fix (referenced in `ConcatRuleBook`):
  https://github.com/tenstorrent/tt-metal/issues/39419
