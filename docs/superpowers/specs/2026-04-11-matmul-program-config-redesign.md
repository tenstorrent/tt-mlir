# Matmul Program Config Redesign

Move matmul program config generation from `LegalOpConfigAnalysis` into
`MatmulRuleBook` (OpRules layer). Add shape-aware output hint filtering
and input-compatibility pruning.

## Problem

Today `generateMatmulProgramConfig()` runs during `LegalOpConfigAnalysis`,
before the optimizer knows input layouts. It produces a single program
config per output sharding type based on output layout + tensor shapes.

This causes three issues:

1. **No shape-aware sharding preference.** The optimizer treats WidthSharded,
   HeightSharded, and BlockSharded equally. For large-M prefill matmuls,
   WidthSharded output causes precision degradation. tt-metal LLM models
   avoid width sharding for prefill, using 2D mcast (BlockSharded) or DRAM
   interleaved instead.

2. **Missing DRAM-sharded config.** Every tt-metal LLM model uses
   `MatmulMultiCoreReuseMultiCastDRAMShardedProgramConfig` for decode
   (M=1 tile, width-sharded L1 activation, width-sharded DRAM weight).
   The compiler never generates this config.

3. **Matmul logic split across files.** Program config generation lives in
   `LegalOpConfigAnalysis` + `MatmulProgramConfig.cpp`, while matmul
   filtering rules live in `MatmulRuleBook`. This makes the matmul
   optimization path harder to reason about and maintain.

## Design

### Architecture

All matmul-specific logic moves into `MatmulRuleBook`. The flow changes from:

```
LegalOpConfigAnalysis::fillOpSpecificAttrs()
  -> generateMatmulProgramConfig(op, outputLayout)
  -> stores MatmulAttrs in OpConfig

MatmulRuleBook::getOutputHints()
  -> dedup (preserves pre-baked attrs)
  -> filter L1-interleaved
```

To:

```
LegalOpConfigAnalysis::fillOpSpecificAttrs()
  -> skip matmul/linear (no matmul-specific code)

MatmulRuleBook::getOutputHints(op, legalConfigs)
  -> dedup by (bufferType, memLayout)
  -> shape-aware filtering
  -> generate program config(s) per surviving hint
  -> attach MatmulAttrs, return hints

MatmulRuleBook::isValidOutputHintForInputs(hint, inputLayouts)  [new]
  -> cheap compatibility check before backend validation
```

### Shape-Aware Output Hint Filtering

`getOutputHints()` extracts Mt (output height in tiles) from the op and
filters output sharding types:

- **Mt > kPrefillMtThreshold (tentatively 4, i.e., M > 128):** Reject
  WidthSharded output hints. Forces BlockSharded (2D mcast) or DRAM
  interleaved for prefill, avoiding width-sharding precision issues.

- **Mt <= 1 (decode, batch=32):** Reject HeightSharded output hints.
  M=1 tile means only 1 core gets work with height sharding.

- **L1 Interleaved:** Reject (existing rule, unchanged).

### Program Config Generation Per Output Sharding

After filtering, `getOutputHints()` generates program configs for each
surviving hint. The program config type is determined by output sharding:

| Output Sharding  | Config Generated                           | Notes                              |
|------------------|--------------------------------------------|------------------------------------|
| WidthSharded L1  | 1D mcast (`mcast_in0=true`)                | Standard                           |
| WidthSharded L1  | DRAM-sharded (additional hint)             | Only when Mt == 1 (decode pattern) |
| HeightSharded L1 | 1D mcast (`mcast_in0=false`, `in0_block_w=Kt`) |                               |
| BlockSharded L1  | 2D mcast                                   | Preferred for large Mt (prefill)   |
| DRAM Interleaved | No program config (nullopt)                | Unchanged                          |

For WidthSharded output with Mt == 1, two hints are generated (same output
layout, different program configs). `isValidOutputHintForInputs()` prunes
the invalid one based on actual input layouts during cross-product evaluation.

Maximum hint count: ~5 (from ~3-4 today). Not an explosion.

### Input Compatibility Pruning

New override `MatmulRuleBook::isValidOutputHintForInputs()` performs a
cheap check before backend validation. It inspects the program config
type in the hint's `MatmulAttrs` and checks input layout compatibility:

| Program Config Type         | in0 Requirement      | in1 Requirement          |
|-----------------------------|----------------------|--------------------------|
| 1D mcast, `mcast_in0=true` | WIDTH_SHARDED        | INTERLEAVED              |
| 1D mcast, `mcast_in0=false`| HEIGHT_SHARDED       | INTERLEAVED              |
| DRAM-sharded                | WIDTH_SHARDED L1     | WIDTH_SHARDED DRAM       |
| 2D mcast                    | BLOCK_SHARDED or INTERLEAVED | INTERLEAVED or sharded |

This prevents wasted backend validation calls for incompatible combinations.

Existing `getInputLayoutFilter()` (rejects width-sharded inputs for
accuracy) and `applyOpSpecificAttrs()` (extracts MatmulAttrs from
winning candidate and applies to IR) are unchanged.

### MatmulProgramConfig.cpp Refactoring

The single entry point `generateMatmulProgramConfig()` is replaced by
four per-config-type generators called directly from `MatmulRuleBook`:

```cpp
std::optional<Attribute>
generateMatmul1DWidthConfig(ctx, Mt, Nt, Kt, outputLayout,
                            fusedActivation, maxSubblockSize, fuseBatch);

std::optional<Attribute>
generateMatmul1DHeightConfig(ctx, Mt, Nt, Kt, outputLayout,
                             fusedActivation, maxSubblockSize, fuseBatch);

std::optional<Attribute>
generateMatmul2DConfig(ctx, Mt, Nt, Kt, outputLayout,
                       fusedActivation, maxSubblockSize, fuseBatch);

std::optional<Attribute>
generateMatmulDRAMShardedConfig(ctx, Mt, Nt, Kt, outputLayout,
                                fusedActivation, maxSubblockSize, fuseBatch);
```

Key improvements per generator:

- **`generateMatmul2DConfig`**: `in0_block_w = 1` for large Mt (stream
  from DRAM, matching tt-metal prefill pattern). `fuseBatch = false` when
  Mt > 64 (matches tt-metal's seq_len > 2048 threshold).

- **`generateMatmul1DHeightConfig`**: `in0_block_w = Kt` (hardware
  requirement when in0 is height-sharded with full K in shard width).

- **`generateMatmulDRAMShardedConfig`** (new): Emits
  `MatmulMultiCoreReuseMultiCastDRAMShardedProgramConfigAttr`. Only
  valid when Mt == 1. `in0_block_w` derived from K and num_cores.

- **`generateMatmul1DWidthConfig`**: Existing width-sharded logic,
  essentially unchanged.

### Cleanup

- **`LegalOpConfigAnalysis.cpp`**: Remove the matmul/linear branch from
  `fillOpSpecificAttrs()`. The `computeKernelConfig` extraction moves
  to `MatmulRuleBook::getOutputHints()`.

- **`OptimizerUtils.cpp`**: Remove `getUniqueTestConfigsForMatmulLinear()`.
  Replace with a simpler `dedupByMemoryLayout()` helper local to
  `MatmulRules.cpp` that deduplicates by (bufferType, memLayout) without
  considering opSpecificAttrs.

## File Changes

| File | Change |
|------|--------|
| `MatmulRules.h` | Add `isValidOutputHintForInputs()` override. Add `dedupByMemoryLayout()` helper declaration if needed. |
| `MatmulRules.cpp` | Rewrite `getOutputHints()` with shape-aware filtering + config generation. Add `isValidOutputHintForInputs()`. Add `dedupByMemoryLayout()`. |
| `MatmulProgramConfig.h` | Replace `generateMatmulProgramConfig()` with four per-type generators. |
| `MatmulProgramConfig.cpp` | Refactor into four generators. Add DRAM-sharded generator. Improve 2D config heuristics. |
| `LegalOpConfigAnalysis.cpp` | Remove matmul/linear branch from `fillOpSpecificAttrs()`. |
| `OptimizerUtils.h` | Remove `getUniqueTestConfigsForMatmulLinear()` declaration. |
| `OptimizerUtils.cpp` | Remove `getUniqueTestConfigsForMatmulLinear()` implementation. |

## Testing

- **Existing lit tests**: Run `check-ttmlir` to verify no regressions in
  compiled MLIR output. Tests that check for specific program config
  attributes may need updating.

- **Perf tests**: Run `check-perf` (resnet, yolo_v8, segformer) to verify
  no performance regressions.

- **PoC script**: `scripts/matmul_transpose_poc.py` provides a standalone
  precision comparison for different matmul configs on hardware.

- **Manual verification**: Compile a prefill model (e.g., llama 70b 2-layer
  prefill) and inspect the emitted MLIR to confirm:
  - Large-M matmuls get BlockSharded output + 2D mcast config
  - Small-M matmuls (decode) get WidthSharded output + 1D mcast or
    DRAM-sharded config
  - No WidthSharded output for matmuls with Mt > threshold

## Risks

- **Threshold tuning**: `kPrefillMtThreshold = 4` is a heuristic. Too low
  may reject valid width-sharded configs for medium-sized matmuls. Too high
  may miss precision improvements. For matmuls with 1 < Mt <= threshold,
  both WidthSharded and BlockSharded hints survive and compete via scoring.
  The threshold can be tuned based on perf test results.

- **DRAM-sharded config correctness**: This is a new config type the compiler
  has never emitted. Backend validation should catch invalid configs, but
  extra care is needed in testing.

- **OptimizerUtils removal**: Verify `getUniqueTestConfigsForMatmulLinear()`
  has no other callers before removing.
