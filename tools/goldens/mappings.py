# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0


from typing import Dict, Callable
import torch
import torch.nn.functional
from ttmlir.dialects import ttir, stablehlo, d2m, ttnn

from goldens.custom_goldens import *


"""
Dictionary mapping TTIR operation classes to their corresponding golden functions.

This dictionary provides a centralized mapping between TTIR operation types and their
PyTorch-based golden reference implementations. Each key is a TTIR operation class
(e.g., ttir.AbsOp) and each value is the corresponding golden function that computes
the expected output for that operation.

The mapping supports:
    - Elementwise unary operations (abs, ceil, cos, etc.)
    - Elementwise binary operations (add, multiply, subtract, etc.)
    - Elementwise ternary operations (where, select, etc.)
    - Comparison operations (eq, ne, lt, gt, etc.)
    - Bitwise operations (and, or, xor, not)
    - Reduction operations (sum, mean, max, min, etc.)
    - Tensor manipulation (transpose, concat, reshape, etc.)
    - Neural network operations (matmul, embedding, conv2d, etc.)
    - Layout operations (to_layout, view_layout)
    - Quantization operations (quantize, dequantize, requantize)
    - Collective communication operations (all_gather, all_reduce, etc.)

Usage:
    golden_fn = GOLDEN_MAPPINGS.get(ttir.AbsOp)
    if golden_fn:
        result = golden_fn(input_tensor)
"""
GOLDEN_MAPPINGS: Dict[type, Callable] = {
    # ----- TTIR OPS -----
    # Elementwise unary operations
    ttir.GetDimensionSizeOp: get_dimension_size_golden,
    ttir.AbsOp: torch.abs,
    ttir.CeilOp: torch.ceil,
    ttir.CosOp: torch.cos,
    ttir.FloorOp: torch.floor,
    ttir.GeluOp: torch.nn.functional.gelu,
    ttir.IsFiniteOp: torch.isfinite,
    ttir.NegOp: torch.neg,
    ttir.TanOp: torch.tan,
    ttir.AtanOp: torch.atan,
    ttir.TanhOp: torch.tanh,
    ttir.ReciprocalOp: torch.reciprocal,
    ttir.ReluOp: torch.relu,
    ttir.Relu6Op: torch.nn.functional.relu6,
    ttir.RsqrtOp: torch.rsqrt,
    ttir.SigmoidOp: torch.sigmoid,
    ttir.SiluOp: torch.nn.functional.silu,
    ttir.SignOp: torch.sign,
    ttir.SinOp: torch.sin,
    ttir.SqrtOp: torch.sqrt,
    ttir.LogOp: torch.log,
    ttir.Log1pOp: torch.log1p,
    ttir.Expm1Op: torch.expm1,
    ttir.ExpOp: torch.exp,
    # Elementwise binary operations
    ttir.AddOp: torch.add,
    ttir.MultiplyOp: torch.multiply,
    ttir.SubtractOp: torch.subtract,
    ttir.DivOp: torch.div,
    ttir.MaximumOp: torch.maximum,
    ttir.MinimumOp: torch.minimum,
    ttir.RemainderOp: torch.remainder,
    ttir.PowOp: torch.pow,
    # Comparison operations
    ttir.EqualOp: equal_golden,
    ttir.NotEqualOp: not_equal_golden,
    ttir.GreaterEqualOp: greater_equal_golden,
    ttir.GreaterThanOp: greater_than_golden,
    ttir.LessEqualOp: less_equal_golden,
    ttir.LessThanOp: less_than_golden,
    # Logical operations
    ttir.LogicalAndOp: logical_and_golden,
    ttir.LogicalOrOp: logical_or_golden,
    ttir.LogicalXorOp: logical_xor_golden,
    ttir.LogicalNotOp: logical_not_golden,
    # Selection operations
    ttir.WhereOp: torch.where,
    # Bitwise operations
    ttir.BitwiseAndOp: torch.bitwise_and,
    ttir.BitwiseOrOp: torch.bitwise_or,
    ttir.BitwiseXorOp: torch.bitwise_xor,
    ttir.BitwiseNotOp: torch.bitwise_not,
    # Reduction operations
    ttir.SumOp: sum_golden,
    ttir.MeanOp: mean_golden,
    ttir.MaxOp: max_golden,
    ttir.MinOp: min_golden,
    ttir.ProdOp: prod_golden,
    ttir.ReduceAndOp: reduce_and_golden,
    ttir.ReduceOrOp: reduce_or_golden,
    # Tensor manipulation
    ttir.TransposeOp: transpose_golden,
    ttir.ConcatOp: concat_golden,
    ttir.RepeatOp: repeat_golden,
    ttir.RepeatInterleaveOp: repeat_interleave_golden,
    ttir.ReshapeOp: reshape_golden,
    ttir.SqueezeOp: squeeze_golden,
    ttir.UnsqueezeOp: unsqueeze_golden,
    ttir.ReverseOp: reverse_golden,
    ttir.PermuteOp: permute_golden,
    ttir.ClampScalarOp: clamp_scalar_golden,
    ttir.ClampTensorOp: clamp_tensor_golden,
    ttir.CumSumOp: cumsum_golden,
    ttir.BroadcastOp: torch.broadcast_to,
    ttir.PadOp: pad_golden,
    ttir.IndexSelectOp: select_golden,
    ttir.IndexOp: index_golden,
    ttir.SliceStaticOp: slice_golden,
    ttir.GatherOp: gather_golden,
    # Neural network operations
    ttir.SoftmaxOp: softmax_golden,
    ttir.MatmulOp: torch.matmul,
    ttir.EmbeddingOp: embedding_golden,
    ttir.Upsample2dOp: upsample2d_golden,
    ttir.BatchNormInferenceOp: batch_norm_golden,
    ttir.RMSNormOp: rms_norm_golden,
    # Type operations
    ttir.TypecastOp: typecast_golden,
    # Tensor creation
    ttir.ZerosOp: zeros_golden,
    ttir.OnesOp: ones_golden,
    ttir.ArangeOp: arange_golden,
    # Quantization operations
    ttir.QuantizeOp: quantize_golden,
    ttir.DequantizeOp: torch.dequantize,
    ttir.RequantizeOp: requantize_golden,
    # Complex operations
    ttir.CbrtOp: cbrt_golden,
    ttir.Conv2dOp: conv2d_golden,
    ttir.ConvTranspose2dOp: conv_transpose2d_golden,
    ttir.MaxPool2dOp: max_pool2d_golden,
    ttir.AvgPool2dOp: avg_pool2d_golden,
    ttir.ArgMaxOp: argmax_golden,
    ttir.LinearOp: linear_golden,
    ttir.DotGeneralOp: dot_general_golden,
    # Layout operations (identity functions) — accept and ignore extra kwargs like reinterpretLayout
    ttir.ToLayoutOp: (lambda x, **kwargs: x),
    # Cache operations
    ttir.FillCacheOp: fill_cache_golden,
    ttir.UpdateCacheOp: update_cache_golden,
    # CCL (Collective Communication Library) operations
    ttir.MeshShardOp: mesh_shard_golden,
    ttir.AllGatherOp: all_gather_golden,
    ttir.AllReduceOp: all_reduce_golden,
    ttir.ReduceScatterOp: reduce_scatter_golden,
    ttir.CollectivePermuteOp: collective_permute_golden,
    ttir.AllToAllOp: all_to_all_golden,
    ttir.CollectiveBroadcastOp: collective_broadcast_golden,
    # Operations with parameter transformations
    ttir.LeakyReluOp: leaky_relu_golden,
    # ----- StableHLO OPS -----
    # StableHLO elementwise operations
    stablehlo.AddOp: torch.add,
    stablehlo.AbsOp: torch.abs,
    stablehlo.CeilOp: torch.ceil,
    stablehlo.CosineOp: torch.cos,
    stablehlo.ExpOp: torch.exp,
    stablehlo.FloorOp: torch.floor,
    stablehlo.LogOp: torch.log,
    stablehlo.LogisticOp: torch.sigmoid,
    stablehlo.NegOp: torch.neg,
    stablehlo.RsqrtOp: torch.rsqrt,
    stablehlo.SineOp: torch.sin,
    stablehlo.SqrtOp: torch.sqrt,
    stablehlo.TanOp: torch.tan,
    # ----- D2M OPS -----
    # D2M Layout operations (identity functions)
    d2m.ToLayoutOp: (lambda x, **kwargs: x),
    d2m.ViewLayoutOp: (lambda x, **kwargs: x),
    # ----- TTNN OPS -----
    # TTNN Elementwise binary operations
    ttnn.MultiplyOp: torch.multiply,
}
