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

## Missing no-workaround tests

| # | Workaround File | Target Op | Remarks |
|---|---|---|---|
| 1 | `AllGatherOpRewritePattern.cpp` | `ttnn::AllGatherOp` | Multi-device op, may need special test infrastructure |
| 3 | `ConcatenateHeadsOpRewritePattern.cpp` | `ttnn::ConcatenateHeadsOp` | |
| 4 | `Conv3dDepthPaddingRewritePattern.cpp` | `ttnn::Conv3dOp` | Two workaround patterns target Conv3dOp (#4 and #5), could share a test |
| 5 | `Conv3dPadOutputChannelsRewritePattern.cpp` | `ttnn::Conv3dOp` | Two workaround patterns target Conv3dOp (#4 and #5), could share a test |
| 8 | `DistributedRMSNormWidthShardInputRewritePattern.cpp` | `ttnn::DistributedRMSNormOp` | Multi-device op, may need special test infrastructure |
| 10 | `ExplicateOperandBroadcastsRewritePattern.cpp` | Ops with `ExplicateOperandBroadcastsTrait` | Trait-based pattern (applies to multiple ops), needs representative op tests |
| 11 | `MultiplyOpDecompositionRewritePattern.cpp` | `ttnn::MultiplyOp` | Not per-op isolated (requires inputs with defining ops) |
| 12 | `NLPConcatHeadsDecodeInputRewritePattern.cpp` | `ttnn::NLPConcatHeadsDecodeOp` | No builder method |
| 14 | `PagedUpdateCacheOpRewritePattern.cpp` | `ttnn::PagedUpdateCacheOp` | |
| 15 | `PointToPointOpRewritePattern.cpp` | `ttnn::PointToPointOp` | Multi-device op, may need special test infrastructure |
| 16 | `RMSNormConfigRewritePattern.cpp` | `ttnn::RMSNormOp` | |
| 17 | `ReduceScatterConfigRewritePattern.cpp` | `ttnn::ReduceScatterOp` | Two workaround patterns target ReduceScatterOp (#17 and #18), could share a test. Multi-device op, may need special test infrastructure |
| 18 | `ReduceScatterOpRewritePattern.cpp` | `ttnn::ReduceScatterOp` | Two workaround patterns target ReduceScatterOp (#17 and #18), could share a test. Multi-device op, may need special test infrastructure |
| 19 | `RotaryEmbeddingOpRewritePattern.cpp` | `ttnn::RotaryEmbeddingOp` | |
| 20 | `ScatterOpRewritePattern.cpp` | `ttnn::ScatterOp` | |
| 21 | `SplitQueryKeyValueAndSplitHeadsOpRewritePattern.cpp` | `ttnn::SplitQueryKeyValueAndSplitHeadsOp` | |
| 22 | `SubtractOpImplicitBroadcastRewritePattern.cpp` | `ttnn::SubtractOp` | |
| 23 | `UpsampleOpRewritePattern.cpp` | `ttnn::UpsampleOp` | |

**Summary:** 8 out of 26 decomposition workaround patterns are covered (Linear x2, SDPA x1, ArgMax x1, CumSum x2, Embedding x1, Pad x1). 18 patterns remain untested.

**Note:** `test_sort_without_workaround` exists in the test file but Sort is not a Decomposition workaround — it lives in a different workaround category.
