# Decomposition Workaround Test Tracking

Cross-reference of decomposition workarounds in
`lib/Dialect/TTNN/Transforms/Workarounds/Decomposition/` against no-workaround
tests in `test/python/golden/ttir_ops/workarounds/decomposition/test_decomp_workarounds.py`.

## Covered by tests

| Workaround File(s) | Target Op | Test |
|---|---|---|
| `LinearOpRewritePattern.cpp`, `LinearOpOutputShapeRewritePattern.cpp` | `ttnn::LinearOp` | `test_linear_without_workaround`, `test_linear_bias_decomposition_without_workaround` |
| `ScaledDotProductAttentionPadTileDimsRewritePattern.cpp` | `ttnn::ScaledDotProductAttentionOp` | `test_sdpa_with_mask_no_workaround`, `test_sdpa_decode_no_workaround` |
| `ArgMaxOpRewritePattern.cpp` | `ttnn::ArgMaxOp` | `test_argmax_without_workaround` |
| `CumSumOpDimRewritePattern.cpp`, `CumSumOpRankRewritePattern.cpp` | `ttnn::MorehCumSumOp` | `test_cumsum_without_workaround` (**XPASS** - workarounds may be removable!) |
| `EmbeddingOpSqueezeWeightRewritePattern.cpp` | `ttnn::EmbeddingOp` | `test_embedding_without_workaround` |
| `PadHighDimRewritePattern.cpp` | `ttnn::PadOp` | `test_pad_high_dim_without_workaround` |
| `ConcatenateHeadsOpRewritePattern.cpp` | `ttnn::ConcatenateHeadsOp` | `test_concatenate_heads_without_workaround` |
| `Conv3dDepthPaddingRewritePattern.cpp`, `Conv3dPadOutputChannelsRewritePattern.cpp` | `ttnn::Conv3dOp` | `test_conv3d_without_workaround` (covers both patterns) |
| `RMSNormConfigRewritePattern.cpp` | `ttnn::RMSNormOp` | `test_rms_norm_without_workaround` (**XPASS** - workarounds may be removable!) |
| `SplitQueryKeyValueAndSplitHeadsOpRewritePattern.cpp` | `ttnn::SplitQueryKeyValueAndSplitHeadsOp` | `test_split_qkv_without_workaround` |
| `SubtractOpImplicitBroadcastRewritePattern.cpp` | `ttnn::SubtractOp` | `test_subtract_without_workaround` (**XPASS** - workarounds may be removable!) |
| `UpsampleOpRewritePattern.cpp` | `ttnn::UpsampleOp` | `test_upsample_without_workaround` |
| `AllGatherOpRewritePattern.cpp` | `ttnn::AllGatherOp` | `test_all_gather_without_workaround` |

## Missing no-workaround tests

| # | Workaround File | Target Op | Remarks |
|---|---|---|---|
| 8 | `DistributedRMSNormWidthShardInputRewritePattern.cpp` | `ttnn::DistributedRMSNormOp` | Multi-device op, no builder method |
| 10 | `ExplicateOperandBroadcastsRewritePattern.cpp` | Ops with `ExplicateOperandBroadcastsTrait` | Trait-based pattern (applies to multiple ops), needs representative op tests |
| 11 | `MultiplyOpDecompositionRewritePattern.cpp` | `ttnn::MultiplyOp` | Not per-op isolated (requires inputs with defining ops, not function arguments) |
| 12 | `NLPConcatHeadsDecodeInputRewritePattern.cpp` | `ttnn::NLPConcatHeadsDecodeOp` | No builder method |
| 14 | `PagedUpdateCacheOpRewritePattern.cpp` | `ttnn::PagedUpdateCacheOp` | No builder method |
| 15 | `PointToPointOpRewritePattern.cpp` | `ttnn::PointToPointOp` | Multi-device op, no builder method |
| 17 | `ReduceScatterConfigRewritePattern.cpp` | `ttnn::ReduceScatterOp` | Multi-device op, may need special test infrastructure |
| 18 | `ReduceScatterOpRewritePattern.cpp` | `ttnn::ReduceScatterOp` | Multi-device op, may need special test infrastructure |
| 19 | `RotaryEmbeddingOpRewritePattern.cpp` | `ttnn::RotaryEmbeddingOp` | No builder method |
| 20 | `ScatterOpRewritePattern.cpp` | `ttnn::ScatterOp` | No existing golden test; index tensor needs valid integer indices |

**Summary:** 17 out of 26 decomposition workaround patterns are covered (Linear x2, SDPA x1, ArgMax x1, CumSum x2, Embedding x1, Pad x1, ConcatenateHeads x1, Conv3d x2, RMSNorm x1, SplitQKV x1, Subtract x1, Upsample x1, AllGather x1). 10 patterns remain untested.

**Note:** `test_sort_without_workaround` exists in the test file but Sort is not a Decomposition workaround — it lives in a different workaround category.
