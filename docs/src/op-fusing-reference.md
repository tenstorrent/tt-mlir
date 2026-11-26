# Op Fusing Reference

Every op fusion implemented in tt-mlir today: what it matches, what it produces, what enables it, and where the code and a test live. It also lists the decompositions that undo fused ops and the fusions people expect that do not exist.

Snapshot of commit `5586f6fa7c`. Written from a single read of the source and the lit and golden tests; nothing was executed (there is no built `ttmlir-opt` in this checkout). Check the cited source when an entry matters to you.

**Fusion** means several ops become fewer ops. **Fold** means an op is absorbed into an attribute or operand of a neighbour. **Defusion** means a decomposition or inlining that undoes a fused op. **OpModel** is the tt-metal-backed op constraint validation, available only in a `TTMLIR_ENABLE_OPMODEL` build. **Level** is `optimization-level`.

## How to use this page

Every fusion is one record under a `###` heading named after its C++ pattern class, with the same fields:

- **Level**: the IR it runs on (StableHLO, Conversion, TTIR, TTNN, D2M, Fold, Defusion).
- **Does**: what is matched and what it becomes, with exact op names.
- **Enabled**: `always`, an option with its default, or *needs OpModel*.
- **Undone by**: the decomposition that can expand the fused op again, when there is one.
- **Source**: the pattern implementation. **Test**: a lit test with an input the pattern accepts.

To check or hand-apply a fusion on a new model:

1. Find the op chain in the model IR and look it up here: search this page for the op name (for example `ttir.rms_norm`) or use [Index by op](#index-by-op).
2. Open **Source** for the exact match conditions. The record is a summary; the matcher is authoritative. **Test** shows IR that matches.
3. Check **Enabled**. If it is an option, set it; if it needs OpModel, compile at `optimization-level` 1 or 2.
4. Check **Undone by** and [What undoes fusions](#what-undoes-fusions): a fused op may be decomposed again, mostly at level 0.
5. If nothing matches, check [Not fused today](#not-fused-today) before concluding a pattern is missing, then see [Tests and adding a fusion](#tests-and-adding-a-fusion).

## Contents

- **StableHLO** (9): ConcatenateToBroadcastInDimFusionPattern; ReshapeGatherFusionPattern; CollapseRowScatterPattern; FuseRMSNormWithCCLPattern; ReduceScatterFusion; AllSliceOfAllGatherPattern; AllToAllFusionPattern; StablehloAggressiveSimplificationPass; ReoutlineCompositePass
- **Conversion (StableHLO to TTIR)** (26): StableHLOToTTIRCompositeOpConversionPattern; TenstorrentRMSNormConversionPattern; CustomCallDistributedRMSNormConversionPattern; TenstorrentLayerNormConversionPattern; TenstorrentGroupNormConversionPattern; TenstorrentScaledDotProductAttentionConversionPattern; TenstorrentTopKConversionPattern; TenstorrentArgMaxConversionPattern; TenstorrentGatherConversionPattern; TenstorrentUniformToRandConversionPattern; ShardyAllSliceToTTIRMeshPartitionConversionPattern; TenstorrentAdamWConversionPattern; TenstorrentSDPAForwardConversionPattern, TenstorrentSDPABackwardConversionPattern and the layernorm / cross_entropy fw/bw variants; StableHLOToTTIRReduceOpConversionPattern (argmax); StableHLOToTTIRReduceWindowOpConversionPattern (pooling); StableHLOToTTIRReduceWindowOpConversionPattern (cumsum, cumprod); CacheFillUpdatePattern; StableHLOToTTIREmbeddingBackwardOpConversionPattern; StableHLOToTTIRScatterOpConversionPattern (whole-operand overwrite); StableHLOGatherToEmbeddingPattern; StableHLOToTTIRSortOpConversionPattern; StableHLOFillCacheConversionPattern and the update / paged variants; StableHLOSamplingConversionPattern; StableHLOTopKOpMHLOConversionPattern; StableHLOToTTIRScaledDotProductAttentionOpConversionPattern; StableHLOToTTCoreFlashMlaPrefillOpConversionPattern
- **Frontend (tt-crank)** (7): build_sdpa; build_rms_norm; create_composite (sdpa_fw/bw, rmsnorm_fw/bw, cross_entropy_fw/bw); build_adamw; build_kv_cache_write, build_index_copy; build_linear, build_addmm, build_gemm; build_layer_norm, build_gelu, build_silu, build_softmax, ...
- **TTIR** (31): ConvAddBias; ConvWithMultiply; BatchNormDecomposition (enabler); RepVGGConvSumFusionPattern; PadConv3dFusionPattern; MatmulWithBiasFusionPattern; PermuteMatmulFusionPattern; PermuteSliceAfterMatmulPattern; SharedLHSMatmulFusion; ReluFusionPattern; Relu6FusionPattern; HardsigmoidFusionPattern; SiluFusionPattern; MishFusingPattern; GeluFusionPattern; SoftmaxFusionPattern; NumericStableSoftmaxFusionPattern; RMSNormFusionPattern; NormalizeRMSNormFusionPattern; RoPERotateHalfFusingPattern; RoPEComplexRotationFusingPattern; RoPEBackwardFusingPattern; RoPEInterleavedPairFusingPattern; SDPAFusingPattern; SDPAHeadExpansionFusingPattern; ConcatenateHeadsUpdatePattern; ReductionWithReshapePattern; ScaledSumToMeanPattern; SpatialMeanOptimizationPattern; ReshapeBroadcastReshapeToRepeatPattern; TopKFusingPattern
- **TTNN** (12): TTNNConv2dWithActivation; TTNNMatmulAndLinearWithActivation; TTNNBinaryOpInputsActivation; TTNNBinaryOpOutputActivation; TTNNRMSNormWithActivation; RoPERotateHalfFusing; RoPEExpandedFusing; RoPEDecodeFusing, RoPEDecodeReshapeFusing; SplitQueryKeyValueAndSplitHeadsFusing; NLPCreateQKVHeadsDecodeFusing; NLPConcatHeadsDecodeFusing; SDPAFusing
- **D2M** (3): FuseD2MElementwiseOpsPattern; FuseD2MEltwiseReductionOpsPattern; TTNNCreateD2MSubgraphsPass
- **Fold** (33): reshape of reshape (TTIR); reshape · permute · reshape (TTIR); permute of permute (TTIR); typecast of typecast (TTIR); repeat of repeat (TTIR); slice of slice (TTIR); slice of concat (TTIR); matmul leading-dim reshapes (TTIR); clamp_scalar of clamp_scalar; clamp_tensor with constant bounds; reverse of reverse; to_layout of to_layout (TTIR); ttnn_metal_layout_cast pairs; involution and idempotence folds; PermuteReshapePermuteFusionPattern; typecast of typecast (TTNN); reshape of reshape (TTNN); permute of permute (TTNN); slice of concat (TTNN); to_layout / to_tensor_spec pairs (TTNN); typecast into to_tensor_spec; to_tensor_spec into creation op; repeat on matmul RHS (TTNN); splat ttnn.constant to ttnn.full; ImplicitBroadcastFold; FoldFullToScalar; FoldSplatConstantReshape; FoldSplatConstantBroadcast; MoveReshapeToConstant; ConsolidateStaticCacheUpdates; RewriteDQToRequantize; CommuteDequantizeBelowQuantizableOp; OptimizationBarrierFold
- **Defusion** (18): SDPADecompositionPattern; SDPADecodeDecompositionPattern; ConcatenateHeadsDecompositionRewritePattern; GroupNormDecompositionRewritePattern; TopKDecompositionRewritePattern; DistributedRMSNormDecompositionRewritePattern; DistributedLayerNormDecompositionRewritePattern; GroupedConvChannelSplitRewritePattern; LinearOpRewritePattern; SplitQueryKeyValueAndSplitHeadsOpRewritePattern; TTNNResolveComposites; DecomposeSDPAPattern; DecomposeRMSNormPattern; DecomposeLayerNormPattern; DecomposeSoftmaxPattern; DecomposeRepeatPattern; SplitQueryKeyValueAndSplitHeadsDecompositionPattern; FlattenOrConvertCompositesPass

## How fusing is wired

1. `stablehlo-pipeline`: `stablehlo-fusing` and `fuse-distributed-custom-calls` always run. `stablehlo-aggressive-simplification` sits behind `enable-aggressive-simplification` (off here).
2. `stablehlo-to-ttir-pipeline`: `legalize-stablehlo-composite-to-ttir` turns recognised `tenstorrent.*` composites and custom calls straight into fused TTIR ops, and leftover composites are inlined. Aggressive simplification is on by default here. There is no `ttir-fusing` in this pipeline.
3. TTIR-to-TTNN pipeline (`createTTIRToTTNNCommonPipeline`): `ttir-fusing` runs three times (before `ttir-to-ttir-decomposition`, after it and the implicit-broadcast fold, and after `ttir-erase-inverse-ops`), each gated by `enable-fusing-pass` (default on). After lowering come `ttnn-resolve-composites`, `ttnn-fusing` (once), `ttnn-decomposition` and the workarounds.
4. D2M: `d2m-generic-fusion` runs in the D2M pipeline, reached through `ttir-to-ttmetal-pipeline` or, inside the TTNN pipeline, through `enable-create-d2m-subgraphs`.

What decides whether a fusion fires:

- The `ttir-fusing` patterns are unconditional except for the three options in the table below.
- The TTNN patterns marked *needs OpModel* are registered only when `ttnn-fusing` runs with `enable-op-constraints`. The pipeline sets that only on the optimizer path, which means `optimization-level` ≥ 1 and a build with `TTMLIR_ENABLE_OPMODEL` (`lib/Dialect/TTNN/Pipelines/TTNNPipelines.cpp:245-285`).
- `optimization-level` defaults to 0, in the pipeline and in tt-crank (`include/ttmlir/Dialect/TTNN/Pipelines/TTNNPipelines.h:50-58`). At level 0 there is no OpModel, so `ttnn-decomposition` has no validator and always decomposes fused SDPA, SDPA decode, concat-heads and group-norm (`lib/Dialect/TTNN/Transforms/Decomposition/TTNNDecompositionPass.cpp:42-65`), and `composite-resolution=auto` inlines composites such as the RoPE one (`TTNNPipelines.cpp:186-209`).

**Options that control fusing**

| Option | Default | Effect |
| --- | --- | --- |
| `optimization-level` | 0 | ≥ 1 turns on the optimizer and OpModel validation, and conv-multiply fusing (`TTNNPipelines.h:595-626`) |
| `enable-fusing-pass` | on | gates all three `ttir-fusing` runs and `ttnn-fusing` |
| `enable-fusing-conv2d-with-multiply-pattern` (pass: `ttnn-enable-conv2d-with-multiply-pattern`) | off; on at level ≥ 1 | `ConvWithMultiply` and `BatchNormDecomposition` |
| `enable-permute-matmul-fusion` | off (tt-crank: on) | `PermuteMatmulFusionPattern` |
| `enable-permute-slice-after-matmul-fusion` | on | `PermuteSliceAfterMatmulPattern` |
| `enable-eltwise-activation-fusion` | off | TTNN unary activation into a binary op's activation lists |
| `enable-op-constraints` (`ttnn-fusing` pass) | off; set by the pipeline on the optimizer path | OpModel validation, and registration of the OpModel-only patterns |
| `enable-rope-fusion`, `enable-sdpa-fusion` (`ttnn-fusing` pass only) | off | TTNN-level RoPE and SDPA fusions (the TTIR-level ones are always on) |
| `composite-resolution` | auto | auto becomes validate on the optimizer path with OpModel, otherwise inline |
| `enable-ttnn-decomposition-pass` | on | gates `ttnn-decomposition` |
| `enable-quant-dequant-conversion-pass` | on | `ttir-quant-dequant-conversion` |
| `enable-create-d2m-subgraphs` | off | D2M subgraph fusion (forces the optimizer on) |
| `enable-elementwise-fusion`, `enable-eltwise-reduction-fusion` | off | `d2m-generic-fusion` variants |

## StableHLO fusions

### ConcatenateToBroadcastInDimFusionPattern

- **Level:** StableHLO
- **Does:** Replaces a concatenate of one tensor with itself, optionally with a reshape, by a single `stablehlo.broadcast_in_dim` so Shardy propagates sharding better.
- **Enabled:** always (`stablehlo-fusing`)
- **Source:** `lib/Dialect/StableHLO/Transforms/StableHLOFusing.cpp:16-246`
- **Test:** `test/ttmlir/Dialect/StableHLO/Transforms/StableHLOFusing/broadcast_in_dim.mlir`

### ReshapeGatherFusionPattern

- **Level:** StableHLO
- **Does:** Folds a `stablehlo.reshape` that unsqueezes the gather indices at the trailing index_vector_dim straight into the `stablehlo.gather` start_indices operand.
- **Enabled:** always (`stablehlo-fusing`)
- **Source:** `lib/Dialect/StableHLO/Transforms/StableHLOFusing.cpp:248-308`
- **Test:** `test/ttmlir/Dialect/StableHLO/Transforms/StableHLOFusing/gather.mlir`

### CollapseRowScatterPattern

- **Level:** StableHLO
- **Does:** Rewrites a 2D element-wise scatter indexed by concat(row, column iota) into a single-axis row scatter with the column as update window.
- **Enabled:** always (`stablehlo-fusing`)
- **Source:** `lib/Dialect/StableHLO/Transforms/StableHLOFusing.cpp:310-422`
- **Test:** `test/ttmlir/Dialect/StableHLO/stablehlo_fusing_collapse_scatter.mlir`

### FuseRMSNormWithCCLPattern

- **Level:** StableHLO
- **Does:** Fuses all_gather, `tenstorrent.rms_norm` custom_call and `sdy.all_slice` into one `tenstorrent.distributed_rms_norm` custom_call on per-device tensors.
- **Enabled:** always (`fuse-distributed-custom-calls`)
- **Undone by:** `DistributedRMSNormDecompositionRewritePattern` unless the input is (1, ..., 32, M) with a weight
- **Source:** `lib/Dialect/StableHLO/Transforms/FuseDistributedCustomCalls.cpp:17-258`
- **Test:** `test/ttmlir/Dialect/StableHLO/Transforms/StableHLOFusing/fuse_distributed_custom_calls.mlir`

### ReduceScatterFusion

- **Level:** StableHLO
- **Does:** Fuses `sdy.all_reduce` followed by `sdy.all_slice` into `sdy.reduce_scatter` plus an optional residual `sdy.all_slice`.
- **Enabled:** always (Shardy CCL canonicalization, upstream pattern)
- **Source:** `lib/Dialect/StableHLO/Transforms/ShardyCCLCanonicalization.cpp:15-53`
- **Test:** `test/ttmlir/Dialect/StableHLO/shardy/shardy_ccl_canonicalization.mlir`

### AllSliceOfAllGatherPattern

- **Level:** StableHLO
- **Does:** Cancels an `sdy.all_slice` applied to an `sdy.all_gather` with matching axes, forwarding the original tensor.
- **Enabled:** always (Shardy CCL canonicalization, upstream pattern)
- **Source:** `lib/Dialect/StableHLO/Transforms/ShardyCCLCanonicalization.cpp:27-28`
- **Test:** `test/ttmlir/Dialect/StableHLO/shardy/shardy_ccl_canonicalization.mlir`

### AllToAllFusionPattern

- **Level:** StableHLO
- **Does:** Merges two chained `sdy.all_to_all` ops with non-overlapping params into one `sdy.all_to_all` with combined params.
- **Enabled:** always (Shardy CCL canonicalization, upstream pattern)
- **Source:** `lib/Dialect/StableHLO/Transforms/ShardyCCLCanonicalization.cpp:30`

### StablehloAggressiveSimplificationPass

- **Level:** StableHLO
- **Does:** Upstream StableHLO canonicalizer that, among many folds, merges chained broadcast_in_dim and reshape ops and flattens nested concatenates; optional in the pipelines.
- **Enabled:** `enable-aggressive-simplification` (off in `stablehlo-pipeline`, on in `stablehlo-to-ttir-pipeline`)
- **Source:** `lib/Dialect/StableHLO/Pipelines/StableHLOPipelines.cpp:29-32`
- **Test:** `test/ttmlir/Dialect/StableHLO/Transforms/StableHLOFusing/broadcast_in_dim.mlir`

### ReoutlineCompositePass

- **Level:** StableHLO
- **Does:** Re-groups ops that were inlined from a `stablehlo.composite` back into a `stablehlo.composite` plus private callee when the tagged group is contiguous and closed.
- **Enabled:** always (`reoutline-composite`)
- **Source:** `lib/Dialect/StableHLO/Transforms/ReoutlineComposite.cpp:1-469`
- **Test:** `test/ttmlir/Conversion/StableHLOToTTIR/composite/test_reoutline_composite.mlir`

## Conversion: composites and custom calls into fused ops

All always on when the conversion runs (`legalize-stablehlo-composite-to-ttir`, `convert-stablehlo-to-ttir`). These are the fused ops a frontend can hand over directly.

### StableHLOToTTIRCompositeOpConversionPattern

- **Level:** Conversion (StableHLO to TTIR), always on
- **Does:** Replaces a `stablehlo.composite` named `tenstorrent.gelu` or `tenstorrent.gelu_tanh` with one `ttir.gelu`, discarding the composite's decomposition body.
- **Source:** `lib/Conversion/StableHLOToTTIR/StableHLOLegalizeCompositePass.cpp:43-86`
- **Test:** `test/ttmlir/Conversion/StableHLOToTTIR/composite/test_gelu.mlir`

### TenstorrentRMSNormConversionPattern

- **Level:** Conversion (StableHLO to TTIR), always on
- **Does:** Lowers `tenstorrent.rms_norm`, as a composite or as the sharded custom_call form, to `ttir.rms_norm` with normalized_shape, optional epsilon and operand segments.
- **Source:** `lib/Conversion/StableHLOToTTIR/StableHLOLegalizeCompositePass.cpp:1170-1284`
- **Test:** `test/ttmlir/Conversion/StableHLOToTTIR/composite/test_rms_norm.mlir`

### CustomCallDistributedRMSNormConversionPattern

- **Level:** Conversion (StableHLO to TTIR), always on
- **Does:** Lowers custom_call @`tenstorrent.distributed_rms_norm`, created by FuseDistributedCustomCallsPass, to `ttir.distributed_rms_norm` with cluster_axis, epsilon and segments.
- **Source:** `lib/Conversion/StableHLOToTTIR/StableHLOLegalizeCompositePass.cpp:1286-1366`
- **Test:** `test/ttmlir/Dialect/StableHLO/Transforms/StableHLOFusing/fuse_distributed_custom_calls.mlir`

### TenstorrentLayerNormConversionPattern

- **Level:** Conversion (StableHLO to TTIR), always on
- **Does:** Lowers composite `tenstorrent.layer_norm` to `ttir.layer_norm` with normalized_shape, optional epsilon and operand segments {input, weight, bias}.
- **Source:** `lib/Conversion/StableHLOToTTIR/StableHLOLegalizeCompositePass.cpp:1368-1444`
- **Test:** `test/ttmlir/Conversion/StableHLOToTTIR/composite/test_layer_norm.mlir`

### TenstorrentGroupNormConversionPattern

- **Level:** Conversion (StableHLO to TTIR), always on
- **Does:** Lowers composite `tenstorrent.group_norm` to `ttir.group_norm`, padding rank<4 inputs to 4D with trailing unit dims and reshaping the result back.
- **Source:** `lib/Conversion/StableHLOToTTIR/StableHLOLegalizeCompositePass.cpp:1538-1652`
- **Test:** `test/ttmlir/Conversion/StableHLOToTTIR/composite/test_group_norm.mlir`
- **Undone by:** `GroupNormDecompositionRewritePattern` at level 0, or when OpModel validation fails

### TenstorrentScaledDotProductAttentionConversionPattern

- **Level:** Conversion (StableHLO to TTIR), always on
- **Does:** Lowers `tenstorrent.scaled_dot_product_attention` (composite or sharded custom_call) to `ttir.scaled_dot_product_attention`, promoting rank-3 q/k/v/mask to 4D and forwarding is_causal and scale.
- **Source:** `lib/Conversion/StableHLOToTTIR/StableHLOLegalizeCompositePass.cpp:1654-1865`
- **Test:** `test/ttmlir/Conversion/StableHLOToTTIR/composite/test_sdpa.mlir`
- **Undone by:** `SDPADecompositionPattern` at level 0, or when OpModel validation fails

### TenstorrentTopKConversionPattern

- **Level:** Conversion (StableHLO to TTIR), always on
- **Does:** Lowers `tenstorrent.topk`, topk_values and topk_indices composites or custom_calls to `ttir.topk`, with a distributed local-topk, all_gather and merge sequence when the topk dim is sharded.
- **Source:** `lib/Conversion/StableHLOToTTIR/StableHLOLegalizeCompositePass.cpp:88-222`
- **Test:** `test/ttmlir/Conversion/StableHLOToTTIR/composite/test_topk.mlir`

### TenstorrentArgMaxConversionPattern

- **Level:** Conversion (StableHLO to TTIR), always on
- **Does:** Lowers `tenstorrent.argmax` (composite or sharded custom_call) to `ttir.argmax`; the custom_call form can expand to a distributed max/argmax/all_gather/gather sequence.
- **Source:** `lib/Conversion/StableHLOToTTIR/StableHLOLegalizeCompositePass.cpp:224-623`

### TenstorrentGatherConversionPattern

- **Level:** Conversion (StableHLO to TTIR), always on
- **Does:** Lowers `tenstorrent.gather` and `tenstorrent.gather_dim` (composite or sharded custom_call) to `ttir.gather`, casting the index tensor to ui32 when it is not already ui32 or ui16.
- **Source:** `lib/Conversion/StableHLOToTTIR/StableHLOLegalizeCompositePass.cpp:1867-1998`
- **Test:** `test/ttmlir/Conversion/StableHLOToTTIR/composite/test_gather.mlir`

### TenstorrentUniformToRandConversionPattern

- **Level:** Conversion (StableHLO to TTIR), always on
- **Does:** Lowers composite `tenstorrent.uniform` to `ttir.rand`, moving scalar low/high constant operands into attributes and fixing seed to 0.
- **Source:** `lib/Conversion/StableHLOToTTIR/StableHLOLegalizeCompositePass.cpp:1111-1168`
- **Test:** `test/ttmlir/Conversion/StableHLOToTTIR/composite/test_uniform.mlir`

### ShardyAllSliceToTTIRMeshPartitionConversionPattern

- **Level:** Conversion (StableHLO to TTIR), always on
- **Does:** Lowers the `sdy.all_slice` composite to a chain of `ttir.mesh_partition` ops, one per sharded tensor dimension, derived from its out_sharding attribute.
- **Source:** `lib/Conversion/StableHLOToTTIR/StableHLOLegalizeCompositePass.cpp:1446-1536`
- **Test:** `test/ttmlir/Conversion/StableHLOToTTIR/composite/test_sdy_all_slice_composite.mlir`

### TenstorrentAdamWConversionPattern

- **Level:** Conversion (StableHLO to TTIR), always on
- **Does:** Lowers composite `tenstorrent.adamw` to `ttir.adamw`, normalizing beta1, beta2, epsilon and weight_decay to f32 and defaulting stochastic_rounding to false.
- **Source:** `lib/Conversion/StableHLOToTTIR/StableHLOLegalizeCompositePass.cpp:2000-2064`
- **Test:** `test/ttmlir/Conversion/StableHLOToTTIR/composite/test_adamw.mlir`

### TenstorrentSDPAForwardConversionPattern, TenstorrentSDPABackwardConversionPattern and the layernorm / cross_entropy fw/bw variants

- **Level:** Conversion (StableHLO to TTIR), always on
- **Does:** Renames `tenstorrent.sdpa_fw`/bw, layernorm_fw/bw and cross_entropy_fw/bw composites into `ttcore.composite` ops that keep their decomposition and carry normalized attributes.
- **Source:** `lib/Conversion/StableHLOToTTIR/StableHLOLegalizeCompositePass.cpp:2066-2349`
- **Test:** `test/ttmlir/Conversion/StableHLOToTTIR/composite/test_sdpa_fw.mlir`

### StableHLOToTTIRReduceOpConversionPattern (argmax)

- **Level:** Conversion (StableHLO to TTIR), always on
- **Does:** Matches a two-input `stablehlo.reduce` over (value, iota) with a compare/select reducer body and rewrites it to a single `ttir.argmax`.
- **Source:** `lib/Conversion/StableHLOToTTIR/StableHLOToTTIRPatterns.cpp:264-769`
- **Test:** `test/ttmlir/Conversion/StableHLOToTTIR/reduction/argmax_op.mlir`

### StableHLOToTTIRReduceWindowOpConversionPattern (pooling)

- **Level:** Conversion (StableHLO to TTIR), always on
- **Does:** Rewrites `stablehlo.reduce_window` with max or add bodies to `ttir.max_pool2d` or `ttir.avg_pool2d`, folding a following divide-by-kernel-size and a preceding spatial `stablehlo.pad`.
- **Source:** `lib/Conversion/StableHLOToTTIR/StableHLOToTTIRPatterns.cpp:2693-3666`
- **Test:** `test/ttmlir/Conversion/StableHLOToTTIR/avgpool2d_op.mlir`

### StableHLOToTTIRReduceWindowOpConversionPattern (cumsum, cumprod)

- **Level:** Conversion (StableHLO to TTIR), always on
- **Does:** Recognizes single-input `stablehlo.reduce_window` with add or multiply body and a full-prefix window and rewrites it to `ttir.cumsum` or `ttir.cumprod`.
- **Source:** `lib/Conversion/StableHLOToTTIR/StableHLOToTTIRPatterns.cpp:2786-2812`
- **Test:** `test/ttmlir/Conversion/StableHLOToTTIR/cumsum_op.mlir`

### CacheFillUpdatePattern

- **Level:** Conversion (StableHLO to TTIR), always on
- **Does:** Rewrites a `stablehlo.scatter` of a 4D update into a 4D cache, indexed by a cache-position block argument, into `ttir.fill_cache` (prefill) or `ttir.update_cache` (single-token decode).
- **Source:** `lib/Conversion/StableHLOToTTIR/StableHLOToTTIRPatterns.cpp:6652-6893`
- **Test:** `test/ttmlir/Conversion/StableHLOToTTIR/kv_cache_fusing.mlir`

### StableHLOToTTIREmbeddingBackwardOpConversionPattern

- **Level:** Conversion (StableHLO to TTIR), always on
- **Does:** Recognizes a `stablehlo.scatter` with a sum update computation into the first dim of a 2D table and emits `ttir.embedding_backward` plus an add of the original operand unless it is zero.
- **Source:** `lib/Conversion/StableHLOToTTIR/StableHLOToTTIRPatterns.cpp:6897-7012`
- **Test:** `test/ttmlir/Conversion/StableHLOToTTIR/scatter_op_to_embedding_bw.mlir`

### StableHLOToTTIRScatterOpConversionPattern (whole-operand overwrite)

- **Level:** Conversion (StableHLO to TTIR), always on
- **Does:** Folds a `stablehlo.scatter` that overwrites the whole operand into the update tensor itself, emitting no scatter op.
- **Source:** `lib/Conversion/StableHLOToTTIR/StableHLOToTTIRPatterns.cpp:7016-7100`
- **Test:** `test/ttmlir/Conversion/StableHLOToTTIR/scatter_origin_op.mlir`

### StableHLOGatherToEmbeddingPattern

- **Level:** Conversion (StableHLO to TTIR), always on
- **Does:** Lowers an embedding-style `stablehlo.gather` to `ttir.embedding` on a flattened 2D weight table, wrapped in reshape/permute ops for operand and result layout.
- **Source:** `lib/Conversion/StableHLOToTTIR/StableHLOToTTIRPatterns.cpp:5015-5130`

### StableHLOToTTIRSortOpConversionPattern

- **Level:** Conversion (StableHLO to TTIR), always on
- **Does:** Converts `stablehlo.sort` to a single `ttir.sort` for value-only and value+iota-index sorts, and to `ttir.sort` plus embedding-based reordering for key/value sorts.
- **Source:** `lib/Conversion/StableHLOToTTIR/StableHLOToTTIRPatterns.cpp:7835-8106`
- **Test:** `test/ttmlir/Conversion/StableHLOToTTIR/data_movement/sort_op.mlir`

### StableHLOFillCacheConversionPattern and the update / paged variants

- **Level:** Conversion (StableHLO to TTIR), always on
- **Does:** Maps `stablehlo.custom_call` `tt.fill_cache`, `tt.update_cache`, `tt.paged_update_cache` and `tt.paged_fill_cache` to the in-place ttir cache ops, rewiring the result to the cache operand.
- **Source:** `lib/Conversion/StableHLOToTTIR/StableHLOToTTIRPatterns.cpp:8494-8797`
- **Test:** `test/ttmlir/Conversion/StableHLOToTTIR/cache_ops.mlir`

### StableHLOSamplingConversionPattern

- **Level:** Conversion (StableHLO to TTIR), always on
- **Does:** Maps `stablehlo.custom_call` `tt.sampling` with five operands to `ttir.sampling`, parsing an optional seed from frontend attributes.
- **Source:** `lib/Conversion/StableHLOToTTIR/StableHLOToTTIRPatterns.cpp:8692-8753`
- **Test:** `test/ttmlir/Conversion/StableHLOToTTIR/sampling_op.mlir`

### StableHLOTopKOpMHLOConversionPattern

- **Level:** Conversion (StableHLO to TTIR), always on
- **Does:** Maps `stablehlo.custom_call` `mhlo.topk` to `ttir.topk` using k and largest from the `mhlo.attributes` dictionary, with dim fixed to -1 and sorted false.
- **Source:** `lib/Conversion/StableHLOToTTIR/StableHLOToTTIRPatterns.cpp:10138-10198`
- **Test:** `test/ttmlir/Conversion/StableHLOToTTIR/reduction/topk_op.mlir`

### StableHLOToTTIRScaledDotProductAttentionOpConversionPattern

- **Level:** Conversion (StableHLO to TTIR), always on
- **Does:** Maps tt.* attention custom_calls to `ttir.scaled_dot_product_attention`, its decode, paged-decode, chunked and paged flash-MLA-decode variants, reading flags from string frontend attributes.
- **Source:** `lib/Conversion/StableHLOToTTIR/StableHLOToTTIRPatterns.cpp:8869-9339`
- **Test:** `test/ttmlir/Conversion/StableHLOToTTIR/transformer/scaled_dot_product_attention.mlir`

### StableHLOToTTCoreFlashMlaPrefillOpConversionPattern

- **Level:** Conversion (StableHLO to TTIR), always on
- **Does:** Converts `tt.flash_mla_prefill` and `tt.indexer_score_dsa` custom_calls into `ttcore.composite` ops with a synthesized private primitive-op decomposition as fallback.
- **Source:** `lib/Conversion/StableHLOToTTIR/StableHLOToTTIRPatterns.cpp:9518-9891`
- **Test:** `test/ttmlir/Conversion/StableHLOToTTIR/transformer/flash_mla_prefill.mlir`

Front ends in this repo (tt-crank) that emit fused ops directly:

### build_sdpa

- **Level:** Frontend (tt-crank)
- **Does:** tt-crank emits one `ttir.scaled_dot_product_attention` for inference SDPA (eager and torch.compile), without going through StableHLO.
- **Source:** `tt-crank/src/engine/ttir_module_builder.cpp:1482-1493`
- **Undone by:** `SDPADecompositionPattern` at level 0, or when OpModel validation fails

### build_rms_norm

- **Level:** Frontend (tt-crank)
- **Does:** tt-crank emits `ttir.rms_norm` for inference rms_norm over the last dim, casting the weight to the input dtype.
- **Source:** `tt-crank/src/engine/ttir_module_builder.cpp:2007-2019`

### create_composite (sdpa_fw/bw, rmsnorm_fw/bw, cross_entropy_fw/bw)

- **Level:** Frontend (tt-crank)
- **Does:** tt-crank emits `ttcore.composite` sdpa_fw, sdpa_bw, rmsnorm_fw, rmsnorm_bw, cross_entropy_fw and cross_entropy_bw with a primitive decomposition, so the backend can promote to ttml kernels or inline.
- **Source:** `tt-crank/src/engine/ttir_module_builder.cpp:119-134`

### build_adamw

- **Level:** Frontend (tt-crank)
- **Does:** tt-crank lowers aten::_fused_adamw_ to `ttir.adamw` per parameter, deriving beta^step tensors in graph and rejecting AMP gradient scaling.
- **Source:** `tt-crank/src/engine/ttir_module_builder.cpp:1495-1532`

### build_kv_cache_write, build_index_copy

- **Level:** Frontend (tt-crank)
- **Does:** tt-crank routes a rank-4 aten::index_copy along dim 2 to `ttir.update_cache` (one new token) or `ttir.fill_cache` (prefill) instead of a generic `ttir.scatter`.
- **Source:** `tt-crank/src/engine/ttir_module_builder.cpp:196-241`

### build_linear, build_addmm, build_gemm

- **Level:** Frontend (tt-crank)
- **Does:** tt-crank folds matmul plus bias add into one `ttir.linear` for aten::linear, addmm with alpha = beta = 1, and ONNX Gemm with transB and a 1-D bias.
- **Source:** `tt-crank/src/engine/ttir_module_builder.cpp:1283-1364`

### build_layer_norm, build_gelu, build_silu, build_softmax, ...

- **Level:** Frontend (tt-crank)
- **Does:** tt-crank maps ATen and ONNX nodes directly to fused TTIR ops such as `ttir.layer_norm`, `ttir.gelu`, `ttir.silu`, `ttir.softmax`, `ttir.embedding`, `ttir.argmax`, `ttir.batch_norm_inference`, `ttir.conv2d` with bias, and `ttir.max_pool2d`.
- **Source:** `tt-crank/src/engine/ttir_module_builder.cpp:504-552`

## TTIR fusions (`ttir-fusing`)

Every entry runs in `ttir-fusing` (three times per pipeline) unless **Enabled** says otherwise.

### ConvAddBias

- **Level:** TTIR · Convolution
- **Does:** Folds `ttir.add` of a per-channel bias into the optional bias operand of a single-use conv2d, conv_transpose2d or conv3d, summing biases if one exists.
- **Enabled:** always (`ttir-fusing`)
- **Source:** `lib/Dialect/TTIR/Transforms/TTIRFusing.cpp:36-132`
- **Test:** `test/ttmlir/Dialect/TTIR/fusing/conv2d_bias_fusing.mlir`

### ConvWithMultiply

- **Level:** TTIR · Convolution
- **Does:** Commutes a multiply by a per-output-channel constant through a single-use conv2d or conv_transpose2d by pre-multiplying the weights (and bias).
- **Enabled:** `ttnn-enable-conv2d-with-multiply-pattern` (off in the pass; the pipeline turns it on at level ≥ 1)
- **Source:** `lib/Dialect/TTIR/Transforms/TTIRFusing.cpp:680-961`
- **Test:** `test/ttmlir/Dialect/TTIR/fusing/conv2d_multiply_commute.mlir`

### BatchNormDecomposition (enabler)

- **Level:** TTIR · Convolution
- **Does:** Decomposes `ttir.batch_norm_inference` that directly follows a single-use conv2d or conv_transpose2d into multiply and add so the conv folding patterns can absorb it.
- **Enabled:** `ttnn-enable-conv2d-with-multiply-pattern` (off in the pass; the pipeline turns it on at level ≥ 1)
- **Source:** `lib/Dialect/TTIR/Transforms/TTIRFusing.cpp:1012-1131`
- **Test:** `test/ttmlir/Dialect/TTIR/fusing/batch_norm_decomposition.mlir`

### RepVGGConvSumFusionPattern

- **Level:** TTIR · Convolution
- **Does:** Merges add of a 3x3 conv2d and a 1x1 conv2d on the same input into one 3x3 conv with zero-padded, summed weights and summed biases.
- **Enabled:** always (`ttir-fusing`)
- **Source:** `lib/Dialect/TTIR/Transforms/TTIRFusing.cpp:1133-1298`
- **Test:** `test/ttmlir/Dialect/TTIR/fusing/repvgg_conv_sum_fusing.mlir`

### PadConv3dFusionPattern

- **Level:** TTIR · Convolution
- **Does:** Folds a single-use zero-valued symmetric spatial `ttir.pad` feeding `ttir.conv3d` into the conv3d padding attribute.
- **Enabled:** always (`ttir-fusing`)
- **Source:** `lib/Dialect/TTIR/Transforms/TTIRFusing.cpp:3562-3631`
- **Test:** `test/ttmlir/Dialect/TTIR/fusing/pad_conv3d_fusing.mlir`

### MatmulWithBiasFusionPattern

- **Level:** TTIR · Matmul and linear
- **Does:** Fuses `ttir.add` of a single-use `ttir.matmul` (optionally through one reshape) and a bias into one `ttir.linear`, replaying the reshape.
- **Enabled:** always (`ttir-fusing`)
- **Undone by:** `LinearOpRewritePattern` when tt-metal cannot fuse the bias
- **Source:** `lib/Dialect/TTIR/Transforms/TTIRFusing.cpp:1300-1498`
- **Test:** `test/ttmlir/Dialect/TTIR/fusing/matmul_with_bias_positive.mlir`

### PermuteMatmulFusionPattern

- **Level:** TTIR · Matmul and linear
- **Does:** Folds a last-two-dims `ttir.permute` on matmul/linear operand A and/or B into the toggled transpose_a/transpose_b attribute.
- **Enabled:** `enable-permute-matmul-fusion` (off; tt-crank turns it on)
- **Source:** `lib/Dialect/TTIR/Transforms/TTIRFusing.cpp:3350-3438`
- **Test:** `test/python/golden/ttir_ops/fusing/test_permute_matmul_fusing.py`

### PermuteSliceAfterMatmulPattern

- **Level:** TTIR · Matmul and linear
- **Does:** Pushes a single-axis `ttir.slice_static` of a single-use matmul/linear output into the operand (and bias) producing that dim.
- **Enabled:** `enable-permute-slice-after-matmul-fusion` (on)
- **Source:** `lib/Dialect/TTIR/Transforms/TTIRFusing.cpp:1500-1710`
- **Test:** `test/ttmlir/Dialect/TTIR/fusing/permute_slice_after_matmul.mlir`

### SharedLHSMatmulFusion

- **Level:** TTIR · Matmul and linear
- **Does:** Merges three or more matmuls/linears sharing the same LHS into one op over concatenated weights (and biases), then slices outputs back.
- **Enabled:** always (`ttir-fusing`)
- **Source:** `lib/Dialect/TTIR/Transforms/TTIRFusing.cpp:1712-2097`
- **Test:** `test/ttmlir/Dialect/TTIR/fusing/shared_lhs_linear_bias.mlir`

### ReluFusionPattern

- **Level:** TTIR · Activations and softmax
- **Does:** Replaces `ttir.maximum`(x, 0) with `ttir.relu`(x) when the other operand is a zeros op or a full op filled with 0.
- **Enabled:** always (`ttir-fusing`)
- **Source:** `lib/Dialect/TTIR/Transforms/TTIRFusing.cpp:372-436`
- **Test:** `test/ttmlir/Dialect/TTIR/fusing/zero_maximum_to_relu.mlir`

### Relu6FusionPattern

- **Level:** TTIR · Activations and softmax
- **Does:** Replaces `ttir.minimum`(`ttir.relu`(x), 6) with `ttir.relu6`(x), accepting either operand order and integer or float 6.
- **Enabled:** always (`ttir-fusing`)
- **Source:** `lib/Dialect/TTIR/Transforms/TTIRFusing.cpp:438-478`
- **Test:** `test/ttmlir/Dialect/TTIR/fusing/relu6_fusing.mlir`

### HardsigmoidFusionPattern

- **Level:** TTIR · Activations and softmax
- **Does:** Replaces div(relu6(add(x, 3)), 6) with `ttir.hardsigmoid`(x).
- **Enabled:** always (`ttir-fusing`)
- **Source:** `lib/Dialect/TTIR/Transforms/TTIRFusing.cpp:480-526`
- **Test:** `test/ttmlir/Dialect/TTIR/fusing/hardsigmoid_fusing.mlir`

### SiluFusionPattern

- **Level:** TTIR · Activations and softmax
- **Does:** Replaces multiply(x, sigmoid(x)) with `ttir.silu`(x), looking through typecasts and re-casting the result to the multiply type.
- **Enabled:** always (`ttir-fusing`)
- **Source:** `lib/Dialect/TTIR/Transforms/TTIRFusing.cpp:524-585`
- **Test:** `test/ttmlir/Dialect/TTIR/fusing/silu_fusing.mlir`

### MishFusingPattern

- **Level:** TTIR · Activations and softmax
- **Does:** Replaces x * tanh(where(x > c, x, log1p(exp(x)))) with `ttir.mish`(x), looking through typecasts on the multiply operands.
- **Enabled:** always (`ttir-fusing`)
- **Source:** `lib/Dialect/TTIR/Transforms/TTIRFusing.cpp:587-678`
- **Test:** `test/ttmlir/Dialect/TTIR/fusing/mish_fusing.mlir`

### GeluFusionPattern

- **Level:** TTIR · Activations and softmax
- **Does:** Fuses 0.5 * x * (1 + erf(x * 0.7071)) or its tanh approximation into a single `ttir.gelu`(x).
- **Enabled:** always (`ttir-fusing`)
- **Source:** `lib/Dialect/TTIR/Transforms/TTIRFusing.cpp:2470-2853`
- **Test:** `test/ttmlir/Dialect/TTIR/fusing/gelu_fusing.mlir`

### SoftmaxFusionPattern

- **Level:** TTIR · Activations and softmax
- **Does:** Replaces div(exp(x), [broadcast](sum(exp(x), keep_dim=true))) with `ttir.softmax`(x) over the single reduced dimension, numericStable=false.
- **Enabled:** always (`ttir-fusing`)
- **Source:** `lib/Dialect/TTIR/Transforms/TTIRFusing.cpp:214-290`
- **Test:** `test/ttmlir/Dialect/TTIR/fusing/softmax_fusing.mlir`

### NumericStableSoftmaxFusionPattern

- **Level:** TTIR · Activations and softmax
- **Does:** Folds softmax(x - max(x)) into `ttir.softmax`(x) with numericStable=true, removing the subtract, max and optional broadcast.
- **Enabled:** always (`ttir-fusing`)
- **Source:** `lib/Dialect/TTIR/Transforms/TTIRFusing.cpp:292-370`
- **Test:** `test/ttmlir/Dialect/TTIR/fusing/softmax_fusing.mlir`

### RMSNormFusionPattern

- **Level:** TTIR · Normalization
- **Does:** Fuses decomposed RMSNorm, x * rsqrt(mean(x^2) + eps) * gamma, into `ttir.rms_norm` over the last dimension.
- **Enabled:** always (`ttir-fusing`)
- **Source:** `lib/Dialect/TTIR/Transforms/TTIRFusing.cpp:2857-3036`
- **Test:** `test/ttmlir/Dialect/TTIR/fusing/rms_norm_fusion.mlir`

### NormalizeRMSNormFusionPattern

- **Level:** TTIR · Normalization
- **Does:** Fuses L2-normalize decomposition scaled by sqrt(D), optionally times a per-channel gamma, into `ttir.rms_norm`, permuting if the reduced dim is not last.
- **Enabled:** always (`ttir-fusing`)
- **Source:** `lib/Dialect/TTIR/Transforms/TTIRFusing.cpp:3038-3348`
- **Test:** `test/ttmlir/Dialect/TTIR/fusing/normalize_rms_norm_fusion.mlir`

### RoPERotateHalfFusingPattern

- **Level:** TTIR · RoPE
- **Does:** Fuses x*cos + concat(neg(x[D/2:]), x[:D/2])*sin on a 4D tensor into a `ttcore.composite` "rotary_embedding" with a decomposition function.
- **Enabled:** always (`ttir-fusing`)
- **Undone by:** `TTNNResolveComposites` inlining (level 0, or validation fails)
- **Source:** `lib/Dialect/TTIR/Transforms/Fusing/RoPEFusingPattern.cpp:441-524`
- **Test:** `test/ttmlir/Dialect/TTIR/fusing/rotary_embedding/rope.mlir`

### RoPEComplexRotationFusingPattern

- **Level:** TTIR · RoPE
- **Does:** Fuses concat(x1*cos - x2*sin, x2*cos + x1*sin) with half-D cos/sin into a rotary_embedding composite taking concat(cos_h, cos_h) and concat(sin_h, sin_h).
- **Enabled:** always (`ttir-fusing`)
- **Undone by:** `TTNNResolveComposites` inlining (level 0, or validation fails)
- **Source:** `lib/Dialect/TTIR/Transforms/Fusing/RoPEFusingPattern.cpp:530-677`
- **Test:** `test/ttmlir/Dialect/TTIR/fusing/rotary_embedding/rope.mlir`

### RoPEBackwardFusingPattern

- **Level:** TTIR · RoPE
- **Does:** Fuses the transpose (backward) RoPE x*cos + concat((x*sin)[D/2:], -(x*sin)[:D/2]) built from zero-padded concats into a rotary_embedding composite with a swapped, negated sin.
- **Enabled:** always (`ttir-fusing`)
- **Undone by:** `TTNNResolveComposites` inlining (level 0, or validation fails)
- **Source:** `lib/Dialect/TTIR/Transforms/Fusing/RoPEFusingPattern.cpp:683-883`
- **Test:** `test/ttmlir/Dialect/TTIR/fusing/rotary_embedding/rope.mlir`

### RoPEInterleavedPairFusingPattern

- **Level:** TTIR · RoPE
- **Does:** Rewrites interleaved-pair RoPE using a (..., D/2, 2, 2) freqs_cis tensor into permute-to-rotate-half, a rotary_embedding composite, and permute-back.
- **Enabled:** always (`ttir-fusing`)
- **Undone by:** `TTNNResolveComposites` inlining (level 0, or validation fails)
- **Source:** `lib/Dialect/TTIR/Transforms/Fusing/RoPEFusingPattern.cpp:885-1336`
- **Test:** `test/ttmlir/Dialect/TTIR/fusing/rotary_embedding/rope.mlir`

The RoPE patterns produce a `ttcore.composite` named `rotary_embedding`. `ttnn-resolve-composites` turns it into `ttnn.rotary_embedding` only when `composite-resolution` resolves to validate (level ≥ 1 with OpModel) or force-promote; otherwise it is inlined back to primitives.

### SDPAFusingPattern

- **Level:** TTIR · Attention
- **Does:** Fuses Q·Kᵀ, optional scale, optional additive mask, softmax and the PV matmul into one `ttir.scaled_dot_product_attention`.
- **Enabled:** always (`ttir-fusing`)
- **Undone by:** `SDPADecompositionPattern` at level 0, or when OpModel validation fails
- **Source:** `lib/Dialect/TTIR/Transforms/Fusing/SDPAFusingPattern.cpp:412-712`
- **Test:** `test/ttmlir/Dialect/TTIR/Transforms/sdpa_fusing/sdpa.mlir`

### SDPAHeadExpansionFusingPattern

- **Level:** TTIR · Attention
- **Does:** Removes a matching head-dim `ttir.repeat_interleave` on both K and V ahead of SDPA, relying on the op's native GQA broadcast.
- **Enabled:** always (`ttir-fusing`)
- **Source:** `lib/Dialect/TTIR/Transforms/Fusing/SDPAFusingPattern.cpp:195-289`
- **Test:** `test/ttmlir/Dialect/TTIR/Transforms/sdpa_fusing/sdpa.mlir`

### ConcatenateHeadsUpdatePattern

- **Level:** TTIR · Attention
- **Does:** Fuses permute [0,2,1,3] on [B,H,S,D] followed by a 3D or 2D reshape into `ttir.concatenate_heads`.
- **Enabled:** always (`ttir-fusing`)
- **Undone by:** `ConcatenateHeadsDecompositionRewritePattern` at level 0, or when OpModel validation fails
- **Source:** `lib/Dialect/TTIR/Transforms/TTIRFusing.cpp:2339-2468`

### ReductionWithReshapePattern

- **Level:** TTIR · Reductions, repeat and top-k
- **Does:** Folds a `ttir.reshape` that merely re-inserts the reduced size-1 dims into the preceding reduction by setting keep_dim=true.
- **Enabled:** always (`ttir-fusing`)
- **Source:** `lib/Dialect/TTIR/Transforms/TTIRFusing.cpp:134-212`
- **Test:** `test/ttmlir/Dialect/TTIR/fusing/reduction_fusing.mlir`

### ScaledSumToMeanPattern

- **Level:** TTIR · Reductions, repeat and top-k
- **Does:** Replaces multiply(sum(x, dims), full(1/prod(reduced dim sizes))) with a single `ttir.mean` over the same dims and keep_dim.
- **Enabled:** always (`ttir-fusing`)
- **Source:** `lib/Dialect/TTIR/Transforms/TTIRFusing.cpp:2099-2211`
- **Test:** `test/ttmlir/Dialect/TTIR/fusing/scaled_sum_to_mean_fusing.mlir`

### SpatialMeanOptimizationPattern

- **Level:** TTIR · Reductions, repeat and top-k
- **Does:** Rewrites a rank-4 `ttir.mean` over spatial dims [1,2] into reshape to (N,1,H*W,C) plus a single-dim mean, with an output reshape if keep_dim is false.
- **Enabled:** always (`ttir-fusing`)
- **Source:** `lib/Dialect/TTIR/Transforms/TTIRFusing.cpp:2213-2338`
- **Test:** `test/ttmlir/Dialect/TTIR/fusing/spatial_mean_optimization.mlir`

### ReshapeBroadcastReshapeToRepeatPattern

- **Level:** TTIR · Reductions, repeat and top-k
- **Does:** Rewrites reshape -> broadcast -> reshape (insert size-1 dim, expand it, merge it) into one `ttir.repeat_interleave` or `ttir.repeat`.
- **Enabled:** always (`ttir-fusing`)
- **Source:** `lib/Dialect/TTIR/Transforms/TTIRFusing.cpp:3440-3560`
- **Test:** `test/ttmlir/Dialect/TTIR/fusing/reshape_broadcast_reshape_to_repeat.mlir`

### TopKFusingPattern

- **Level:** TTIR · Reductions, repeat and top-k
- **Does:** Fuses `ttir.sort` whose values and/or indices are only consumed by `ttir.slice_static` taking the first or last k along the sort dim into `ttir.topk`.
- **Enabled:** always (`ttir-fusing`)
- **Undone by:** `TopKDecompositionRewritePattern`, only when OpModel validation fails
- **Source:** `lib/Dialect/TTIR/Transforms/Fusing/TopKFusingPattern.cpp:15-211`
- **Test:** `test/ttmlir/Dialect/TTIR/fusing/topk_fusing.mlir`

## TTNN fusions (`ttnn-fusing`)

### TTNNConv2dWithActivation

- **Level:** TTNN
- **Does:** Folds a relu, relu6, silu or sigmoid that consumes a single-use `ttnn.conv2d` (directly or via one reshape) into conv2d_config.activation.
- **Enabled:** always (`ttnn-fusing`)
- **Source:** `lib/Dialect/TTNN/Transforms/TTNNFusing.cpp:34-127`
- **Test:** `test/ttmlir/Dialect/TTNN/fusing/conv2d_activation_fusing.mlir`

### TTNNMatmulAndLinearWithActivation

- **Level:** TTNN
- **Does:** Folds a single-use sigmoid, silu or gelu following `ttnn.matmul` or `ttnn.linear` into the op's string activation attribute.
- **Enabled:** always (`ttnn-fusing`)
- **Source:** `lib/Dialect/TTNN/Transforms/TTNNFusing.cpp:129-179`
- **Test:** `test/ttmlir/Dialect/TTNN/Transforms/Fusing/matmul_fusing.mlir`

### TTNNBinaryOpInputsActivation

- **Level:** TTNN
- **Does:** Folds a single-use unary op on an operand of a TTNN binary op into that op's input_tensor_a/b_activations attribute.
- **Enabled:** `enable-eltwise-activation-fusion` (off)
- **Source:** `lib/Dialect/TTNN/Transforms/TTNNFusing.cpp:182-286`
- **Test:** `test/ttmlir/Dialect/TTNN/fusing/binary_activation_fusing.mlir`

### TTNNBinaryOpOutputActivation

- **Level:** TTNN
- **Does:** Folds a unary op consuming a single-use TTNN binary op result into that op's output activations list.
- **Enabled:** `enable-eltwise-activation-fusion` (off)
- **Source:** `lib/Dialect/TTNN/Transforms/TTNNFusing.cpp:288-331`
- **Test:** `test/ttmlir/Dialect/TTNN/fusing/binary_activation_fusing.mlir`

### TTNNRMSNormWithActivation

- **Level:** TTNN
- **Does:** Fuses `ttnn.silu`, `ttnn.gelu` or `ttnn.relu` after `ttnn.rms_norm` into `ttnn.dit_rms_norm_unary_fused` with the activation as a string attribute.
- **Enabled:** needs OpModel (`enable-op-constraints`, set on the optimizer path: level ≥ 1)
- **Source:** `lib/Dialect/TTNN/Transforms/TTNNFusing.cpp:581-679`
- **Test:** `test/ttmlir/Dialect/TTNN/optimizer/ttnn_fusing/dit_rms_norm_unary_fused/dit_rms_norm_unary_fused.mlir`

### RoPERotateHalfFusing

- **Level:** TTNN
- **Does:** Fuses (x*cos)+(rotate_half(x)*sin) over `ttnn.slice_static`/neg/concat/multiply/add into `ttnn.rotary_embedding`, validated with OpModel.
- **Enabled:** `enable-rope-fusion` (off) and OpModel
- **Source:** `lib/Dialect/TTNN/Transforms/Fusing/RoPEFusingPattern.cpp:28-35`
- **Test:** `test/ttmlir/Dialect/TTNN/optimizer/ttnn_fusing/rotary_embedding/rope.mlir`

### RoPEExpandedFusing

- **Level:** TTNN
- **Does:** Fuses the expanded trig-identity RoPE (concat of subtract/add of half-dim products) into `ttnn.rotary_embedding` with doubled cos/sin.
- **Enabled:** `enable-rope-fusion` (off) and OpModel
- **Source:** `lib/Dialect/TTNN/Transforms/Fusing/RoPEFusingPattern.cpp:683-824`
- **Test:** `test/ttmlir/Dialect/TTNN/optimizer/ttnn_fusing/rotary_embedding/rope.mlir`

### RoPEDecodeFusing, RoPEDecodeReshapeFusing

- **Level:** TTNN
- **Does:** Commutes the decode layout transform (permute or folded reshape) in front of `ttnn.rotary_embedding` and sets token_index=0, enabling decode QKV-head fusion.
- **Enabled:** needs OpModel (`enable-op-constraints`, set on the optimizer path: level ≥ 1)
- **Source:** `lib/Dialect/TTNN/Transforms/Fusing/RoPEFusingPattern.cpp:1040-1234`
- **Test:** `test/ttmlir/Dialect/TTNN/optimizer/ttnn_fusing/rotary_embedding/rope_decode_folded_reshape.mlir`

### SplitQueryKeyValueAndSplitHeadsFusing

- **Level:** TTNN
- **Does:** Fuses a QKV projection matmul/linear followed by three slice, reshape and optional permute chains into one `ttnn.split_query_key_value_and_split_heads`.
- **Enabled:** needs OpModel (`enable-op-constraints`, set on the optimizer path: level ≥ 1)
- **Undone by:** `SplitQueryKeyValueAndSplitHeadsOpRewritePattern` when head size is not a multiple of 32
- **Source:** `lib/Dialect/TTNN/Transforms/Fusing/SplitQKVFusingPatterns.cpp:20-775`
- **Test:** `test/ttmlir/Dialect/TTNN/optimizer/ttnn_fusing/split_qkv/split_qkv_matmul.mlir`

### NLPCreateQKVHeadsDecodeFusing

- **Level:** TTNN
- **Does:** Fuses a decode-shaped split_query_key_value_and_split_heads plus a [2,0,1,3] permute on each output into one `ttnn.nlp_create_qkv_heads_decode`.
- **Enabled:** needs OpModel (`enable-op-constraints`, set on the optimizer path: level ≥ 1)
- **Source:** `lib/Dialect/TTNN/Transforms/Fusing/SplitQKVFusingPatterns.cpp:777-797`
- **Test:** `test/ttmlir/Dialect/TTNN/optimizer/ttnn_fusing/split_qkv/split_qkv_decode.mlir`

### NLPConcatHeadsDecodeFusing

- **Level:** TTNN
- **Does:** Fuses the decode head-collapse (permute [1,2,0,3] or its reshape form, then reshape to [B,H*D]) into `ttnn.nlp_concat_heads_decode` plus a reshape.
- **Enabled:** needs OpModel (`enable-op-constraints`, set on the optimizer path: level ≥ 1)
- **Source:** `lib/Dialect/TTNN/Transforms/TTNNFusing.cpp:332-579`
- **Test:** `test/ttmlir/Dialect/TTNN/optimizer/ttnn_fusing/nlp_concat_heads_decode/nlp_concat_heads_decode.mlir`

### SDPAFusing

- **Level:** TTNN
- **Does:** Fallback fusion of `ttnn.matmul`/softmax attention chains into `ttnn.scaled_dot_product_attention` or its decode variant; off by default, TTIR-level fusion supersedes it.
- **Enabled:** `enable-sdpa-fusion` (off) and OpModel
- **Undone by:** `SDPADecompositionPattern` when OpModel validation fails
- **Source:** `lib/Dialect/TTNN/Transforms/Fusing/SDPAFusingPattern.cpp:34-780`

## D2M fusions

### FuseD2MElementwiseOpsPattern

- **Level:** D2M
- **Does:** Merges a compute-only, all-parallel producer `d2m.generic` into its single consumer `d2m.generic`, cloning both bodies into one generic region.
- **Enabled:** `enable-elementwise-fusion` (off)
- **Source:** `lib/Dialect/D2M/Transforms/GenericFusion.cpp:98-247`
- **Test:** `test/ttmlir/Dialect/D2M/Transforms/eltwise_fusion_intermediate_buffer.mlir`

### FuseD2MEltwiseReductionOpsPattern

- **Level:** D2M
- **Does:** Fuses an all-parallel elementwise `d2m.generic` producer into a reduction `d2m.generic` consumer that has exactly one reduction iterator.
- **Enabled:** `enable-eltwise-reduction-fusion` (off)
- **Source:** `lib/Dialect/D2M/Transforms/GenericFusion.cpp:590-661`
- **Test:** `test/ttmlir/Dialect/D2M/Transforms/eltwise_reduction_fusion.mlir`

### TTNNCreateD2MSubgraphsPass

- **Level:** D2M
- **Does:** Outlines single-exit trees of allowlisted ttnn eltwise ops (at least 2 ops) into a private function called through `ttnn.d2m_subgraph`.
- **Enabled:** `enable-create-d2m-subgraphs` (off)
- **Source:** `lib/Dialect/TTNN/Transforms/TTNNCreateD2MSubgraphs.cpp:27-327`
- **Test:** `test/ttmlir/Dialect/TTNN/dispatch_d2m/create_d2m_subgraphs.mlir`

## Folds of chained ops

Op folders, canonicalizers and small passes that merge neighbouring ops. The `canonicalize` entries run wherever the generic `canonicalize` pass runs, which the pipelines schedule between most stages.

### reshape of reshape (TTIR)

- **Level:** Fold
- **Does:** Back-to-back `ttir.reshape` ops are merged: the consumer reshape is rewired to read the producer's input.
- **Runs in:** `canonicalize`
- **Source:** `lib/Dialect/TTIR/IR/TTIROps.cpp:2776-2811`

### reshape · permute · reshape (TTIR)

- **Level:** Fold
- **Does:** A trailing reshape that only drops leading unit dims after a permute is absorbed by lowering the permute and its leading reshape to the smaller rank.
- **Runs in:** `canonicalize`
- **Source:** `lib/Dialect/TTIR/IR/TTIROps.cpp:2813-2901`

### permute of permute (TTIR)

- **Level:** Fold
- **Does:** Two directly chained `ttir.permute` ops are composed into a single permute on the original input.
- **Runs in:** `canonicalize`
- **Source:** `lib/Dialect/TTIR/IR/TTIROps.cpp:6455-6471`

### typecast of typecast (TTIR)

- **Level:** Fold
- **Does:** Two chained `ttir.typecast` ops become one typecast from the original input to the final dtype, unless conservative_folding blocks lossy shapes.
- **Runs in:** `canonicalize`
- **Source:** `lib/Dialect/TTIR/IR/TTIROps.cpp:4084-4131`

### repeat of repeat (TTIR)

- **Level:** Fold
- **Does:** Two chained `ttir.repeat` ops are merged into one repeat whose repeat_dimensions are the elementwise product.
- **Runs in:** `canonicalize`
- **Source:** `lib/Dialect/TTIR/IR/TTIROps.cpp:5460-5495`

### slice of slice (TTIR)

- **Level:** Fold
- **Does:** Two chained `ttir.slice_static` ops are merged into one slice with composed begins, ends and steps.
- **Runs in:** `canonicalize`
- **Source:** `lib/Dialect/TTIR/IR/TTIROps.cpp:3309-3369`

### slice of concat (TTIR)

- **Level:** Fold
- **Does:** A `ttir.slice_static` that selects exactly one operand of a `ttir.concat` is replaced by that operand.
- **Runs in:** `canonicalize`
- **Source:** `lib/Dialect/TTIR/IR/TTIROps.cpp:3371-3442`

### matmul leading-dim reshapes (TTIR)

- **Level:** Fold
- **Does:** Drops reshapes that merge leading batch dims on both matmul inputs and split them again on the output, matmul on the unmerged shapes.
- **Runs in:** `canonicalize`
- **Source:** `lib/Dialect/TTIR/IR/TTIROps.cpp:5042-5204`

### clamp_scalar of clamp_scalar

- **Level:** Fold
- **Does:** Two chained `ttir.clamp_scalar` ops merge into one with tighter bounds (max of mins, min of maxes).
- **Runs in:** `canonicalize`
- **Source:** `lib/Dialect/TTIR/IR/TTIROps.cpp:987-1036`

### clamp_tensor with constant bounds

- **Level:** Fold
- **Does:** A `ttir.clamp_tensor` whose min and max operands are constant fill ops becomes `ttir.clamp_scalar` with the values as attributes.
- **Runs in:** `canonicalize`
- **Source:** `lib/Dialect/TTIR/IR/TTIROps.cpp:697-722`

### reverse of reverse

- **Level:** Fold
- **Does:** Two chained `ttir.reverse` ops merge into one reversing the symmetric difference of their dimension sets; empty sets disappear.
- **Runs in:** `canonicalize`
- **Source:** `lib/Dialect/TTIR/IR/TTIROps.cpp:6350-6393`

### to_layout of to_layout (TTIR)

- **Level:** Fold
- **Does:** Back-to-back `ttir.to_layout` ops collapse into one to_layout from the original input to the final output; to_layout of `ttir.empty` becomes an empty.
- **Runs in:** `canonicalize`; D2M option `disable-tolayout-folding`
- **Source:** `lib/Dialect/TTIR/IR/TTIROps.cpp:4357-4376`

### ttnn_metal_layout_cast pairs

- **Level:** Fold
- **Does:** Two chained `ttir.ttnn_metal_layout_cast` ops (ttnn->metal->ttnn or metal->ttnn->metal) are replaced by the first cast's input.
- **Runs in:** `canonicalize`
- **Source:** `lib/Dialect/TTIR/IR/TTIROps.cpp:4597-4647`

### involution and idempotence folds

- **Level:** Fold
- **Does:** Repeated applications of an involutive or idempotent elementwise op, or a binary idempotent op on identical operands, fold into the single inner value.
- **Runs in:** `canonicalize`
- **Source:** `lib/Dialect/TTIR/IR/TTIRTraits.cpp:17-70`

### PermuteReshapePermuteFusionPattern

- **Level:** Fold
- **Does:** Collapses permute -> rank-preserving reshape -> permute into one `ttir.reshape` when the reshape only merges or unit-expands axes and the second permute restores source order.
- **Runs in:** `ttir-erase-inverse-ops` (`enable-erase-inverse-ops-pass`, on)
- **Source:** `lib/Dialect/TTIR/IR/TTIRTMFusionPatterns.cpp:14-159`

### typecast of typecast (TTNN)

- **Level:** Fold
- **Does:** Two chained `ttnn.typecast` ops merge into one typecast from the original value to the final dtype when the first has a single use.
- **Runs in:** `canonicalize`
- **Source:** `lib/Dialect/TTNN/IR/TTNNOps.cpp:48-77`

### reshape of reshape (TTNN)

- **Level:** Fold
- **Does:** Chained `ttnn.reshape` ops merge: the consumer is rewired to the producer's input unless another user writes the intermediate.
- **Runs in:** `canonicalize`
- **Source:** `lib/Dialect/TTNN/IR/TTNNOps.cpp:1749-1797`

### permute of permute (TTNN)

- **Level:** Fold
- **Does:** Two chained `ttnn.permute` ops compose into one when the first has a single use.
- **Runs in:** `canonicalize`
- **Source:** `lib/Dialect/TTNN/IR/TTNNOps.cpp:5763-5801`

### slice of concat (TTNN)

- **Level:** Fold
- **Does:** A `ttnn.slice_static` that selects exactly one operand of a `ttnn.concat` is replaced by that operand.
- **Runs in:** `canonicalize`
- **Source:** `lib/Dialect/TTNN/IR/TTNNOps.cpp:1962-2041`

### to_layout / to_tensor_spec pairs (TTNN)

- **Level:** Fold
- **Does:** Two chained `ttnn.to_layout` (or two chained `ttnn.to_tensor_spec`) ops merge into one reading the first's input, with a DRAM-staging guard.
- **Runs in:** `canonicalize`
- **Source:** `lib/Dialect/TTNN/IR/TTNNOps.cpp:2349-2459`

### typecast into to_tensor_spec

- **Level:** Fold
- **Does:** A `ttnn.to_tensor_spec` reading a `ttnn.typecast` that widened losslessly and returning to the original dtype bypasses the typecast.
- **Runs in:** `canonicalize`
- **Source:** `lib/Dialect/TTNN/IR/TTNNOps.cpp:2506-2572`

### to_tensor_spec into creation op

- **Level:** Fold
- **Does:** A `ttnn.to_tensor_spec` after a single-use creation op is absorbed into the creation op's result type, device operand and layout.
- **Runs in:** `canonicalize`
- **Source:** `lib/Dialect/TTNN/IR/TTNNOps.cpp:2574-2693`

### repeat on matmul RHS (TTNN)

- **Level:** Fold
- **Does:** A `ttnn.repeat` feeding the RHS of `ttnn.matmul` that only expands dim 1 of a rank-4 operand is dropped, relying on TTNN's implicit batch broadcast.
- **Runs in:** `canonicalize`
- **Source:** `lib/Dialect/TTNN/IR/TTNNOps.cpp:2999-3058`

### splat ttnn.constant to ttnn.full

- **Level:** Fold
- **Does:** Rewrites a `ttnn.constant` with a splat value into `ttnn.full` carrying the fill value attribute.
- **Runs in:** `canonicalize`
- **Source:** `lib/Dialect/TTNN/IR/TTNNOps.cpp:310-329`

### ImplicitBroadcastFold

- **Level:** Fold
- **Does:** Removes explicit `ttir.broadcast` feeding Broadcastable ops that broadcast implicitly, re-adding one broadcast on the output if the result shape grew.
- **Runs in:** `ttir-implicit-broadcast-fold` (`enable-implicit-broadcast-folding-pass`, on)
- **Source:** `lib/Dialect/TTIR/Transforms/Broadcast.cpp:22-119`

### FoldFullToScalar

- **Level:** Fold
- **Does:** Shrinks `ttir.full`/zeros/ones to an all-ones-shape tensor when every user is Broadcastable, letting consumers broadcast implicitly.
- **Runs in:** `ttir-fold-full-to-scalar` (always in the TTNN pipeline)
- **Source:** `lib/Dialect/TTIR/Transforms/Broadcast.cpp:125-213`

### FoldSplatConstantReshape

- **Level:** Fold
- **Does:** Replaces a `ttir.reshape` of a splat constant (directly or through a reshape/broadcast chain) by a new splat `ttir.constant` of the target shape.
- **Runs in:** `ttir-fold-constant-reshape-broadcast`
- **Source:** `lib/Dialect/TTIR/Transforms/FoldConstantReshapeBroadcast.cpp:19-122`

### FoldSplatConstantBroadcast

- **Level:** Fold
- **Does:** Replaces a `ttir.broadcast` of a splat constant (directly or through a reshape/broadcast chain) by a splat constant of the broadcast shape.
- **Runs in:** `ttir-fold-constant-reshape-broadcast`
- **Source:** `lib/Dialect/TTIR/Transforms/FoldConstantReshapeBroadcast.cpp:124-161`

### MoveReshapeToConstant

- **Level:** Fold
- **Does:** Moves a reshape from the activation operand of an elementwise binary op to its constant operand, where it folds into the constant; output reshape is kept.
- **Runs in:** `ttir-move-reshape-to-constant`
- **Source:** `lib/Dialect/TTIR/Transforms/MoveReshapeToConstant.cpp:19-145`

### ConsolidateStaticCacheUpdates

- **Level:** Fold
- **Does:** Unifies N per-layer cumulative_length block args that each get the same constant increment and are returned, so CSE can collapse the N adds into one.
- **Runs in:** `ttir-consolidate-static-cache-updates` (always in the TTNN pipeline)
- **Source:** `lib/Dialect/TTIR/Transforms/ConsolidateStaticCacheUpdates.cpp:20-262`

### RewriteDQToRequantize

- **Level:** Fold
- **Does:** Fuses `ttir.dequantize` followed by `ttir.quantize` into one `ttir.requantize`.
- **Runs in:** `ttir-quant-dequant-conversion` (`enable-quant-dequant-conversion-pass`, on)
- **Source:** `lib/Dialect/TTIR/Transforms/QuantDequantConversion.cpp:146-168`

### CommuteDequantizeBelowQuantizableOp

- **Level:** Fold
- **Does:** Rewrites op(dequantize(x)) into dequantize(quantized_op(x)) for ops implementing QuantizableOpInterface, else falls back to a quantize/dequantize sandwich.
- **Runs in:** `ttir-quant-dequant-conversion` (`enable-quant-dequant-conversion-pass`, on)
- **Source:** `lib/Dialect/TTIR/Transforms/QuantDequantConversion.cpp:24-144`

### OptimizationBarrierFold

- **Level:** Fold
- **Does:** Replaces `ttcore.optimization_barrier` by its operands, removing the barrier late in the TTNN pipeline.
- **Runs in:** `ttcore-optimization-barrier-fold`
- **Source:** `lib/Dialect/TTCore/Transforms/OptimizationBarrierFold.cpp:15-49`

## What undoes fusions

Fused ops are often expanded again later. At level 0 there is no OpModel validator, so the SDPA, SDPA-decode, concat-heads and group-norm decompositions fire unconditionally. From level 1 the fused op is kept when OpModel validation accepts it.

### SDPADecompositionPattern

- **Level:** Defusion
- **Does:** Expands `ttnn.scaled_dot_product_attention` into transpose, matmul, scale, mask, softmax, matmul primitives, unconditionally or only when OpModel validation of the fused op fails.
- **When:** `ttnn-decomposition`: level 0 always; level ≥ 1 only if OpModel validation of the fused op fails
- **Source:** `lib/Dialect/TTNN/Transforms/Decomposition/SDPADecompositionPattern.cpp:14-429`

### SDPADecodeDecompositionPattern

- **Level:** Defusion
- **Does:** Rewrites `ttnn.scaled_dot_product_attention_decode` into permute, prefill `ttnn.scaled_dot_product_attention` with a synthesized mask, permute back; the prefill op may decompose further.
- **When:** `ttnn-decomposition`: same as SDPA
- **Source:** `lib/Dialect/TTNN/Transforms/Decomposition/SDPADecodeDecompositionPattern.cpp:15-401`

### ConcatenateHeadsDecompositionRewritePattern

- **Level:** Defusion
- **Does:** Rewrites `ttnn.concatenate_heads` into permute plus reshape, unconditionally or when OpModel validation of the fused op fails.
- **When:** `ttnn-decomposition`: same as SDPA
- **Source:** `lib/Dialect/TTNN/Transforms/Decomposition/ConcatenateHeadsDecompositionRewritePattern.cpp:18-77`

### GroupNormDecompositionRewritePattern

- **Level:** Defusion
- **Does:** Expands `ttnn.group_norm` into reshape, mean, subtract, square, mean, add eps, rsqrt, multiply, reshape plus optional weight multiply and bias add.
- **When:** `ttnn-decomposition`: same as SDPA
- **Source:** `lib/Dialect/TTNN/Transforms/Decomposition/GroupNormDecompositionRewritePattern.cpp:18-164`

### TopKDecompositionRewritePattern

- **Level:** Defusion
- **Does:** Turns `ttnn.topk` back into `ttnn.sort` plus two `ttnn.slice_static` only when OpModel validation of the topk fails; without a validator the topk is always preserved.
- **When:** `ttnn-decomposition`: only if OpModel validation fails (`ttnn.topk` is kept at level 0)
- **Source:** `lib/Dialect/TTNN/Transforms/Decomposition/TopKDecompositionRewritePattern.cpp:15-109`

### DistributedRMSNormDecompositionRewritePattern

- **Level:** Defusion
- **Does:** Keeps `ttnn.distributed_rms_norm` only for (1,..,32,M) inputs with a weight (reshaping to canonical form); every other case is expanded into pre-all-gather, all_gather and primitive post-processing ops.
- **When:** `ttnn-decomposition`: always; the fused op survives only for (1, …, 32, M) inputs with a weight
- **Source:** `lib/Dialect/TTNN/Transforms/Decomposition/DistributedRMSNormDecompositionRewritePattern.cpp:16-403`

### DistributedLayerNormDecompositionRewritePattern

- **Level:** Defusion
- **Does:** Unconditionally expands `ttnn.distributed_layer_norm` into layer_norm_pre_all_gather, all_gather, layer_norm_post_all_gather with an optional residual add first.
- **When:** `ttnn-decomposition`: always
- **Source:** `lib/Dialect/TTNN/Transforms/Decomposition/DistributedLayerNormDecompositionRewritePattern.cpp:14-112`

### GroupedConvChannelSplitRewritePattern

- **Level:** Defusion
- **Does:** Splits a grouped `ttnn.conv1d`/conv2d wider than 4096 channels into per-group-chunk convolutions joined by `ttnn.concat` along channels.
- **When:** `ttnn-decomposition`: always, for grouped convs wider than 4096 channels
- **Source:** `lib/Dialect/TTNN/Transforms/Decomposition/GroupedConvChannelSplitRewritePattern.cpp:23-190`

### LinearOpRewritePattern

- **Level:** Defusion
- **Does:** Rewrites `ttnn.linear` with a bias that tt-metal cannot fuse into `ttnn.matmul` + `ttnn.add` (+ reshape and activation op).
- **When:** `ttnn-workaround`: on unless `disable-workarounds` or the decomposition-workaround option is off
- **Source:** `lib/Dialect/TTNN/Transforms/Workarounds/Decomposition/LinearOpRewritePattern.cpp:15-230`

### SplitQueryKeyValueAndSplitHeadsOpRewritePattern

- **Level:** Defusion
- **Does:** Rewrites `ttnn.split_query_key_value_and_split_heads` into slice_static, reshape and permute chains when head_size is not divisible by 32.
- **When:** `ttnn-workaround`: same gate, when head size is not a multiple of 32
- **Source:** `lib/Dialect/TTNN/Transforms/Workarounds/Decomposition/SplitQueryKeyValueAndSplitHeadsOpRewritePattern.cpp:14-330`

### TTNNResolveComposites

- **Level:** Defusion
- **Does:** Resolves each `ttcore.composite` either to its typed fused TTNN op (when registered, allowed and valid) or by inlining its decomposition function body.
- **When:** `ttnn-resolve-composites`: `composite-resolution` auto = inline unless the optimizer and OpModel are on
- **Source:** `lib/Dialect/TTNN/Transforms/TTNNResolveComposites.cpp:210-872`

### DecomposeSDPAPattern

- **Level:** Defusion
- **Does:** Expands `ttir.scaled_dot_product_attention` into permute, matmul, scale, mask, softmax and matmul for the D2M/TTMetal path, using Q-reshape for GQA.
- **When:** `ttir-decompose-composites`: D2M / TTMetal path
- **Source:** `lib/Dialect/TTIR/Transforms/DecomposeComposites.cpp:222-399`

### DecomposeRMSNormPattern

- **Level:** Defusion
- **Does:** Expands `ttir.rms_norm` into multiply, mean over normalized dims, add eps, rsqrt, multiply and optional weight multiply and bias add.
- **When:** `ttir-decompose-composites`: D2M / TTMetal path
- **Source:** `lib/Dialect/TTIR/Transforms/DecomposeComposites.cpp:21-111`

### DecomposeLayerNormPattern

- **Level:** Defusion
- **Does:** Expands `ttir.layer_norm` into mean, subtract, square, mean, add eps, rsqrt, multiply and optional weight/bias, using one single-dim mean per reduced dimension.
- **When:** `ttir-decompose-composites`: D2M / TTMetal path
- **Source:** `lib/Dialect/TTIR/Transforms/DecomposeComposites.cpp:113-216`

### DecomposeSoftmaxPattern

- **Level:** Defusion
- **Does:** Expands `ttir.softmax` into optional max-subtract, exp, sum and divide for the D2M path.
- **When:** `ttir-decompose-composites`: D2M / TTMetal path
- **Source:** `lib/Dialect/TTIR/Transforms/DecomposeComposites.cpp:405-455`

### DecomposeRepeatPattern

- **Level:** Defusion
- **Does:** Expands `ttir.repeat` into nested `ttir.concat` ops using binary decomposition of each repeat count to keep concat chains O(log n).
- **When:** `ttir-decompose-composites`: D2M / TTMetal path
- **Source:** `lib/Dialect/TTIR/Transforms/DecomposeComposites.cpp:457-563`

### SplitQueryKeyValueAndSplitHeadsDecompositionPattern

- **Level:** Defusion
- **Does:** In cpu-fallback mode only, expands `ttir.split_query_key_value_and_split_heads` into slice_static, reshape and permute ops; in TTNN and TTMetal modes the op stays legal.
- **When:** `ttir-to-ttir-decomposition` in `cpu-fallback` mode only
- **Source:** `lib/Conversion/TTIRToTTIRDecomposition/TTIRToTTIRDecomposition.cpp:2142-2348`

### FlattenOrConvertCompositesPass

- **Level:** Defusion
- **Does:** Inlines `stablehlo.composite` bodies in sharded modules, tagging cloned ops with reoutline.* attributes, but converts composites with custom sharding rules into custom_call ops.
- **When:** `flatten-or-convert-composites` in `stablehlo-pipeline`; no-op for unsharded modules
- **Source:** `lib/Dialect/StableHLO/Transforms/FlattenOrConvertComposites.cpp:49-365`

`ConcatenateHeadsOpRewritePattern` (`lib/Dialect/TTNN/Transforms/Workarounds/Decomposition/ConcatenateHeadsOpRewritePattern.cpp`) is a stale source file: its header does not exist and nothing registers it.

## Not fused today

Expectations that come up often. Status is *implemented*, *partial*, *not implemented*, *gated* (off by default) or *undone* (a later pass reverses it).

| Expectation | Status | What happens instead |
| --- | --- | --- |
| LayerNorm from primitive ops | not implemented | `ttir.layer_norm` is created only from the `tenstorrent.layer_norm` composite (`lib/Conversion/StableHLOToTTIR/StableHLOLegalizeCompositePass.cpp:1440`); primitive chains stay separate. |
| GroupNorm or BatchNorm from primitive ops | not implemented | `ttir.group_norm` comes only from the `tenstorrent.group_norm` composite; batch norm lowers 1:1. The one BN rewrite goes the other way: after a conv it is decomposed so it can fold into the weights. |
| RMSNorm without a scale | partial | The rsqrt form needs the final multiply by the weight; only the weightless L2-normalize form fuses (inferred from the matcher, no test). |
| Residual add plus norm in one op | not implemented | The add stays a separate op; `ttnn.rms_norm` has no residual operand. |
| Dense gate+up (two matmuls sharing an input) | not implemented | `SharedLHSMatmulFusion` needs at least three matmuls on the same input (`lib/Dialect/TTIR/Transforms/TTIRFusing.cpp:1826`), so QKV fuses and a 2-way gate+up does not. |
| QKV with fewer than three sharers | partial | Same three-sharer rule at TTIR; the TTNN `SplitQueryKeyValueAndSplitHeadsFusing` needs exactly three slice chains. |
| SwiGLU or GeGLU as one op | partial | No GLU op exists. `silu` / `gelu` fold into the producing matmul or linear, and with `enable-eltwise-activation-fusion` unary ops fold into a binary op's activation lists. |
| Matmul plus activation inside the kernel program config | partial | The fold is an op-level `activation` string attribute (sigmoid, silu, gelu only), not the program config's fused activation. |
| Conv plus activation beyond relu, relu6, silu, sigmoid | partial | Hardsigmoid is left out pending a tt-metal issue (TODO in the pattern list in `TTNNFusing.cpp`). |
| tanh-GELU stays tanh | partial | `GeluFusionPattern` matches both the erf and tanh forms but always emits `ttir.gelu`, which has no approximate flag (TODO #4486). |
| Causal mask recovered as `is_causal` | not implemented | The SDPA fuser hardcodes `is_causal=false` and keeps the dense additive mask (`lib/Dialect/TTIR/Transforms/Fusing/SDPAFusingPattern.cpp:661`). |
| Sliding-window mask recovered as `sliding_window_size` | not implemented | The attribute is always empty on fused ops. |
| Attention sinks | implemented | Recovered from `concat(scores, sink)`, softmax, then a slice dropping the sink column. |
| GQA head expansion absorbed by SDPA | implemented | `SDPAHeadExpansionFusingPattern` removes the `repeat_interleave` on K and V. |
| Logit soft-capping inside SDPA | not implemented | A `tanh` in the score chain stops the match, and no SDPA op has a soft-cap attribute. |
| Fused SDPA surviving at the default level | undone | At level 0 `ttnn.scaled_dot_product_attention` and the decode op are decomposed back to primitives (see above). |
| Composite-backed fused ops (RoPE, flash-MLA prefill, gated delta rule, ...) | undone at level 0 | `composite-resolution=auto` inlines every composite when the optimizer is off. |
| QKV projection plus RoPE in one op | not implemented | RoPE and the QKV split are separate fusions. |
| KV-cache update fused with attention | not implemented | The cache ops and the SDPA decode ops stay separate. |
| Embedding scale folded away | not implemented | No pattern folds a constant multiply into the table or the lookup. |
| Softmax plus cross-entropy from primitives | not implemented | `ttnn.cross_entropy_fw` exists only through the composite path. |
| All-gather plus matmul, or matmul plus reduce-scatter | not implemented | No such ops exist; the nearest fusion is `FuseRMSNormWithCCLPattern` at StableHLO level. |
| Elementwise chains outside D2M | gated | On the TTNN path only unary activations fold into a conv, matmul or binary op; general chains need `enable-create-d2m-subgraphs`. |
| Top-k plus softmax for MoE routers | partial | Only `sort` plus slice becomes `ttir.topk`. `topk_router_gpt` exists but no fusing pattern produces it. |
| Conv plus batch norm folded at level 0 | gated | Needs `ttnn-enable-conv2d-with-multiply-pattern`, which the pipeline turns on at level ≥ 1. |

## Tests and adding a fusion

- Lit tests: `test/ttmlir/Dialect/TTIR/fusing/` (TTIR patterns), `test/ttmlir/Dialect/TTIR/Transforms/sdpa_fusing/`, `test/ttmlir/Dialect/TTNN/Transforms/Fusing/` and `test/ttmlir/Dialect/TTNN/fusing/` (TTNN patterns), `test/ttmlir/Dialect/TTNN/optimizer/ttnn_fusing/` (OpModel-validated TTNN fusions, `REQUIRES: opmodel`), `test/ttmlir/Dialect/D2M/Transforms/` (D2M fusion), and `test/ttmlir/Dialect/StableHLO/Transforms/StableHLOFusing/`.
- Golden tests: `test/python/golden/ttir_ops/fusing/` (one file per pattern), plus `test/python/golden/test_ttir_fusing.py` and `test_ttnn_fusing.py`.
- A typical TTIR test runs `ttmlir-opt --canonicalize --ttir-fusing -o %t %s` and FileCheck: `CHECK` the fused op, `CHECK-NOT` the original ops. Negative cases are named `*_negative.mlir`, option-off cases `*_disabled.mlir`.
- To add a TTIR fusion: write the pattern in `lib/Dialect/TTIR/Transforms/TTIRFusing.cpp` (or under `Fusing/` for large ones) and register it in `TTIRFusingPass::runOnOperation`. Gate it with a pass option in `include/ttmlir/Dialect/TTIR/Transforms/Passes.td` and thread the option through `TTNNPipelines.h` and `TTNNPipelines.cpp` when it should be pipeline-visible.
- A TTNN fusion that needs hardware validation goes inside the `enableOpConstraints` block of `lib/Dialect/TTNN/Transforms/TTNNFusing.cpp` (around lines 715-753).
- Inside a pattern use `rewriter.notifyMatchFailure()`, never `emitOpError()` (`AGENTS.md`).

## Index by op

Ops that the records name in dotted form, with the records that mention them (as input or as result). Some records name ops without the dialect prefix (for example conv2d or sigmoid), so also search the page for the bare name.

- `d2m.generic`: FuseD2MElementwiseOpsPattern; FuseD2MEltwiseReductionOpsPattern
- `sdy.all_gather`: AllSliceOfAllGatherPattern
- `sdy.all_reduce`: ReduceScatterFusion
- `sdy.all_slice`: FuseRMSNormWithCCLPattern; ReduceScatterFusion; AllSliceOfAllGatherPattern; ShardyAllSliceToTTIRMeshPartitionConversionPattern
- `sdy.all_to_all`: AllToAllFusionPattern
- `sdy.reduce_scatter`: ReduceScatterFusion
- `stablehlo.broadcast_in_dim`: ConcatenateToBroadcastInDimFusionPattern
- `stablehlo.composite`: ReoutlineCompositePass; StableHLOToTTIRCompositeOpConversionPattern; FlattenOrConvertCompositesPass
- `stablehlo.custom_call`: StableHLOFillCacheConversionPattern and the update / paged variants; StableHLOSamplingConversionPattern; StableHLOTopKOpMHLOConversionPattern
- `stablehlo.gather`: ReshapeGatherFusionPattern; StableHLOGatherToEmbeddingPattern
- `stablehlo.pad`: StableHLOToTTIRReduceWindowOpConversionPattern (pooling)
- `stablehlo.reduce`: StableHLOToTTIRReduceOpConversionPattern (argmax)
- `stablehlo.reduce_window`: StableHLOToTTIRReduceWindowOpConversionPattern (pooling); StableHLOToTTIRReduceWindowOpConversionPattern (cumsum, cumprod)
- `stablehlo.reshape`: ReshapeGatherFusionPattern
- `stablehlo.scatter`: CacheFillUpdatePattern; StableHLOToTTIREmbeddingBackwardOpConversionPattern; StableHLOToTTIRScatterOpConversionPattern (whole-operand overwrite)
- `stablehlo.sort`: StableHLOToTTIRSortOpConversionPattern
- `tenstorrent.adamw`: TenstorrentAdamWConversionPattern
- `tenstorrent.argmax`: TenstorrentArgMaxConversionPattern
- `tenstorrent.distributed_rms_norm`: FuseRMSNormWithCCLPattern; CustomCallDistributedRMSNormConversionPattern
- `tenstorrent.gather`: TenstorrentGatherConversionPattern
- `tenstorrent.gather_dim`: TenstorrentGatherConversionPattern
- `tenstorrent.gelu`: StableHLOToTTIRCompositeOpConversionPattern
- `tenstorrent.gelu_tanh`: StableHLOToTTIRCompositeOpConversionPattern
- `tenstorrent.group_norm`: TenstorrentGroupNormConversionPattern
- `tenstorrent.layer_norm`: TenstorrentLayerNormConversionPattern
- `tenstorrent.rms_norm`: FuseRMSNormWithCCLPattern; TenstorrentRMSNormConversionPattern
- `tenstorrent.scaled_dot_product_attention`: TenstorrentScaledDotProductAttentionConversionPattern
- `tenstorrent.sdpa_fw`: TenstorrentSDPAForwardConversionPattern, TenstorrentSDPABackwardConversionPattern and the layernorm / cross_entropy fw/bw variants
- `tenstorrent.topk`: TenstorrentTopKConversionPattern
- `tenstorrent.uniform`: TenstorrentUniformToRandConversionPattern
- `tt.fill_cache`: StableHLOFillCacheConversionPattern and the update / paged variants
- `tt.flash_mla_prefill`: StableHLOToTTCoreFlashMlaPrefillOpConversionPattern
- `tt.indexer_score_dsa`: StableHLOToTTCoreFlashMlaPrefillOpConversionPattern
- `tt.paged_fill_cache`: StableHLOFillCacheConversionPattern and the update / paged variants
- `tt.paged_update_cache`: StableHLOFillCacheConversionPattern and the update / paged variants
- `tt.sampling`: StableHLOSamplingConversionPattern
- `tt.update_cache`: StableHLOFillCacheConversionPattern and the update / paged variants
- `ttcore.composite`: TenstorrentSDPAForwardConversionPattern, TenstorrentSDPABackwardConversionPattern and the layernorm / cross_entropy fw/bw variants; StableHLOToTTCoreFlashMlaPrefillOpConversionPattern; create_composite (sdpa_fw/bw, rmsnorm_fw/bw, cross_entropy_fw/bw); RoPERotateHalfFusingPattern; TTNNResolveComposites
- `ttcore.optimization_barrier`: OptimizationBarrierFold
- `ttir.adamw`: TenstorrentAdamWConversionPattern; build_adamw
- `ttir.add`: ConvAddBias; MatmulWithBiasFusionPattern
- `ttir.argmax`: TenstorrentArgMaxConversionPattern; StableHLOToTTIRReduceOpConversionPattern (argmax); build_layer_norm, build_gelu, build_silu, build_softmax, ...
- `ttir.avg_pool2d`: StableHLOToTTIRReduceWindowOpConversionPattern (pooling)
- `ttir.batch_norm_inference`: build_layer_norm, build_gelu, build_silu, build_softmax, ...; BatchNormDecomposition (enabler)
- `ttir.broadcast`: ImplicitBroadcastFold; FoldSplatConstantBroadcast
- `ttir.clamp_scalar`: clamp_scalar of clamp_scalar; clamp_tensor with constant bounds
- `ttir.clamp_tensor`: clamp_tensor with constant bounds
- `ttir.concat`: slice of concat (TTIR); DecomposeRepeatPattern
- `ttir.concatenate_heads`: ConcatenateHeadsUpdatePattern
- `ttir.constant`: FoldSplatConstantReshape
- `ttir.conv2d`: build_layer_norm, build_gelu, build_silu, build_softmax, ...
- `ttir.conv3d`: PadConv3dFusionPattern
- `ttir.cumprod`: StableHLOToTTIRReduceWindowOpConversionPattern (cumsum, cumprod)
- `ttir.cumsum`: StableHLOToTTIRReduceWindowOpConversionPattern (cumsum, cumprod)
- `ttir.dequantize`: RewriteDQToRequantize
- `ttir.distributed_rms_norm`: CustomCallDistributedRMSNormConversionPattern
- `ttir.embedding`: StableHLOGatherToEmbeddingPattern; build_layer_norm, build_gelu, build_silu, build_softmax, ...
- `ttir.embedding_backward`: StableHLOToTTIREmbeddingBackwardOpConversionPattern
- `ttir.empty`: to_layout of to_layout (TTIR)
- `ttir.fill_cache`: CacheFillUpdatePattern; build_kv_cache_write, build_index_copy
- `ttir.full`: FoldFullToScalar
- `ttir.gather`: TenstorrentGatherConversionPattern
- `ttir.gelu`: StableHLOToTTIRCompositeOpConversionPattern; build_layer_norm, build_gelu, build_silu, build_softmax, ...; GeluFusionPattern
- `ttir.group_norm`: TenstorrentGroupNormConversionPattern
- `ttir.hardsigmoid`: HardsigmoidFusionPattern
- `ttir.layer_norm`: TenstorrentLayerNormConversionPattern; build_layer_norm, build_gelu, build_silu, build_softmax, ...; DecomposeLayerNormPattern
- `ttir.linear`: build_linear, build_addmm, build_gemm; MatmulWithBiasFusionPattern
- `ttir.matmul`: MatmulWithBiasFusionPattern
- `ttir.max_pool2d`: StableHLOToTTIRReduceWindowOpConversionPattern (pooling); build_layer_norm, build_gelu, build_silu, build_softmax, ...
- `ttir.maximum`: ReluFusionPattern
- `ttir.mean`: ScaledSumToMeanPattern; SpatialMeanOptimizationPattern
- `ttir.mesh_partition`: ShardyAllSliceToTTIRMeshPartitionConversionPattern
- `ttir.minimum`: Relu6FusionPattern
- `ttir.mish`: MishFusingPattern
- `ttir.pad`: PadConv3dFusionPattern
- `ttir.permute`: PermuteMatmulFusionPattern; permute of permute (TTIR)
- `ttir.quantize`: RewriteDQToRequantize
- `ttir.rand`: TenstorrentUniformToRandConversionPattern
- `ttir.relu`: ReluFusionPattern; Relu6FusionPattern
- `ttir.relu6`: Relu6FusionPattern
- `ttir.repeat`: ReshapeBroadcastReshapeToRepeatPattern; repeat of repeat (TTIR); DecomposeRepeatPattern
- `ttir.repeat_interleave`: SDPAHeadExpansionFusingPattern; ReshapeBroadcastReshapeToRepeatPattern
- `ttir.requantize`: RewriteDQToRequantize
- `ttir.reshape`: ReductionWithReshapePattern; reshape of reshape (TTIR); PermuteReshapePermuteFusionPattern; FoldSplatConstantReshape
- `ttir.reverse`: reverse of reverse
- `ttir.rms_norm`: TenstorrentRMSNormConversionPattern; build_rms_norm; RMSNormFusionPattern; NormalizeRMSNormFusionPattern; DecomposeRMSNormPattern
- `ttir.sampling`: StableHLOSamplingConversionPattern
- `ttir.scaled_dot_product_attention`: TenstorrentScaledDotProductAttentionConversionPattern; StableHLOToTTIRScaledDotProductAttentionOpConversionPattern; build_sdpa; SDPAFusingPattern; DecomposeSDPAPattern
- `ttir.scatter`: build_kv_cache_write, build_index_copy
- `ttir.silu`: build_layer_norm, build_gelu, build_silu, build_softmax, ...; SiluFusionPattern
- `ttir.slice_static`: PermuteSliceAfterMatmulPattern; TopKFusingPattern; slice of slice (TTIR); slice of concat (TTIR)
- `ttir.softmax`: build_layer_norm, build_gelu, build_silu, build_softmax, ...; SoftmaxFusionPattern; NumericStableSoftmaxFusionPattern; DecomposeSoftmaxPattern
- `ttir.sort`: StableHLOToTTIRSortOpConversionPattern; TopKFusingPattern
- `ttir.split_query_key_value_and_split_heads`: SplitQueryKeyValueAndSplitHeadsDecompositionPattern
- `ttir.to_layout`: to_layout of to_layout (TTIR)
- `ttir.topk`: TenstorrentTopKConversionPattern; StableHLOTopKOpMHLOConversionPattern; TopKFusingPattern
- `ttir.ttnn_metal_layout_cast`: ttnn_metal_layout_cast pairs
- `ttir.typecast`: typecast of typecast (TTIR)
- `ttir.update_cache`: CacheFillUpdatePattern; build_kv_cache_write, build_index_copy
- `ttnn.add`: LinearOpRewritePattern
- `ttnn.concat`: slice of concat (TTNN); GroupedConvChannelSplitRewritePattern
- `ttnn.concatenate_heads`: ConcatenateHeadsDecompositionRewritePattern
- `ttnn.constant`: splat ttnn.constant to ttnn.full
- `ttnn.conv1d`: GroupedConvChannelSplitRewritePattern
- `ttnn.conv2d`: TTNNConv2dWithActivation
- `ttnn.d2m_subgraph`: TTNNCreateD2MSubgraphsPass
- `ttnn.distributed_layer_norm`: DistributedLayerNormDecompositionRewritePattern
- `ttnn.distributed_rms_norm`: DistributedRMSNormDecompositionRewritePattern
- `ttnn.dit_rms_norm_unary_fused`: TTNNRMSNormWithActivation
- `ttnn.full`: splat ttnn.constant to ttnn.full
- `ttnn.gelu`: TTNNRMSNormWithActivation
- `ttnn.group_norm`: GroupNormDecompositionRewritePattern
- `ttnn.linear`: TTNNMatmulAndLinearWithActivation; LinearOpRewritePattern
- `ttnn.matmul`: TTNNMatmulAndLinearWithActivation; SDPAFusing; repeat on matmul RHS (TTNN); LinearOpRewritePattern
- `ttnn.nlp_concat_heads_decode`: NLPConcatHeadsDecodeFusing
- `ttnn.nlp_create_qkv_heads_decode`: NLPCreateQKVHeadsDecodeFusing
- `ttnn.permute`: permute of permute (TTNN)
- `ttnn.relu`: TTNNRMSNormWithActivation
- `ttnn.repeat`: repeat on matmul RHS (TTNN)
- `ttnn.reshape`: reshape of reshape (TTNN)
- `ttnn.rms_norm`: TTNNRMSNormWithActivation
- `ttnn.rotary_embedding`: RoPERotateHalfFusing; RoPEExpandedFusing; RoPEDecodeFusing, RoPEDecodeReshapeFusing
- `ttnn.scaled_dot_product_attention`: SDPAFusing; SDPADecompositionPattern; SDPADecodeDecompositionPattern
- `ttnn.scaled_dot_product_attention_decode`: SDPADecodeDecompositionPattern
- `ttnn.silu`: TTNNRMSNormWithActivation
- `ttnn.slice_static`: RoPERotateHalfFusing; slice of concat (TTNN); TopKDecompositionRewritePattern
- `ttnn.sort`: TopKDecompositionRewritePattern
- `ttnn.split_query_key_value_and_split_heads`: SplitQueryKeyValueAndSplitHeadsFusing; SplitQueryKeyValueAndSplitHeadsOpRewritePattern
- `ttnn.to_layout`: to_layout / to_tensor_spec pairs (TTNN)
- `ttnn.to_tensor_spec`: to_layout / to_tensor_spec pairs (TTNN); typecast into to_tensor_spec; to_tensor_spec into creation op
- `ttnn.topk`: TopKDecompositionRewritePattern
- `ttnn.typecast`: typecast of typecast (TTNN); typecast into to_tensor_spec
