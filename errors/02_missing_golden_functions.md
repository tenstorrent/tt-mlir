# Missing TTNN Golden Functions (~545 failures, 19 ops)

## Error Signature
```
No golden function found for TTIR operation: <class 'ttmlir.dialects._ttnn_ops_gen.XXXOp'>
```

## Representative Test
`test_reduction_ops[ttnn-sum-dim_arg2-True-f32-32x128x128]`

## How the Registry Works

`tools/golden/mapping.py` line 6827: `GOLDEN_MAPPINGS` is a flat `Dict[type, Callable]`. `get_golden_function(op_class)` at line 7154 does the lookup. The registry has TTIR entries but is **missing corresponding TTNN entries** for many ops.

## All Missing Ops

### Category 1: Direct Reuse (11 ops -- just add mapping entry)
No new code needed, point TTNN op to existing TTIR golden:

| Missing Key | Existing Golden | Line |
|---|---|---|
| `ttnn.SumOp` | `ttir_sum_golden` | 3721 |
| `ttnn.MaxOp` | `ttir_max_golden` | 4249 |
| `ttnn.CumSumOp` | `ttir_cumsum_golden` | 3342 |
| `ttnn.TopKOp` | `ttir_topk_golden` | 4882 |
| `ttnn.ReshapeOp` | `ttir_reshape_golden` | 3733 |
| `ttnn.PermuteOp` | `ttir_permute_golden` | 3760 |
| `ttnn.PadOp` | `ttir_pad_golden` | 3826 |
| `ttnn.SliceStaticOp` | `ttir_slice_golden` | 3683 |
| `ttnn.SortOp` | `ttir_sort_golden` | 4611 |
| `ttnn.HardsigmoidOp` | `ttir_hardsigmoid_golden` | 4538 |
| `ttnn.GlobalAvgPool2dOp` | `global_avg_pool2d_golden` | 962 |

### Category 2: Needs Wrapper (7 ops -- thin adapter ~5-15 lines each)
Existing kwargs-based goldens need adapters for MLIR attribute signatures:

| Missing Key | Base Golden | Complexity |
|---|---|---|
| `ttnn.ProdOp` | `prod_golden` | Low |
| `ttnn.MinOp` | `min_golden` | Low |
| `ttnn.MeanOp` | `mean_golden` | Low |
| `ttnn.SoftmaxOp` | `softmax_golden` | Low |
| `ttnn.GeluBackwardOp` | `ttir_gelu_backward_golden` | Low |
| `ttnn.ScaledDotProductAttentionOp` | `sdpa_golden` | Medium |
| `ttnn.ScaledDotProductAttentionDecodeOp` | `sdpa_decode_golden` | Medium |

### Category 3: Special (1 op)
| Missing Key | Notes |
|---|---|
| `ttcore.LoadCachedOp` | Const-eval caching op. Needs passthrough/identity golden or skip logic. |

## Fix Location

**Single file:** `tools/golden/mapping.py`, add entries to `GOLDEN_MAPPINGS` dict at ~line 7135.
