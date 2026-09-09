// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// RUN: ttmlir-opt --ttcore-register-device --ttir-to-d2m --d2m-materialize-view-returns -o %t %s
// RUN: FileCheck %s --input-file=%t

// argmax lowers to d2m.tile_argmax, which maps onto the
// max_reduce_with_indices LLK. The LLK consumes one value tile plus one index
// tile per call and keeps running value/index maxima in DST, so the op has four
// results: the two reduced tiles and the two accumulators.
//
// The LLK requires both operands in row-major layout, which the tile-based
// d2m.generic machinery cannot express. The lowering works around that by
// physically untilizing each operand and then relabelling the buffer as
// tile-typed via a reinterpret view_layout, so the checks below expect a
// view_layout on each operand rather than a plain to_layout.

module {

  // Reducing the last dim, the orientation the LLK handles natively.
  //
  // The index operand is built by an arange generic sized to the reduction
  // extent (128), so each element carries its own global position; the LLK
  // returns whichever index sits alongside the winning value. Both operands are
  // then relabelled row-major-as-tile: the 4x1 shard shape in the view types
  // below is the reduction axis spanning four tiles.
  // CHECK-LABEL: func @argmax_reduce_last
  // CHECK: d2m.arange_block{{.*}}num_elements = 128
  // CHECK: d2m.view_layout{{.*}}reinterpretLayout = true{{.*}}-> tensor<1x1x4x1x!ttcore.tile<32x32, bf16>
  // CHECK: d2m.view_layout{{.*}}reinterpretLayout = true{{.*}}-> tensor<1x1x4x1x!ttcore.tile<32x32, si32>
  // CHECK: d2m.generic{{.+}}iterator_types = [#reduction, #parallel]
  // CHECK: linalg.generic{{.+}}iterator_types = ["reduction", "parallel"]
  // CHECK: %out_values, %out_indices, %val_acc_out, %idx_acc_out = "d2m.tile_argmax"(%{{[a-zA-Z0-9_]+}}, %{{[a-zA-Z0-9_]+}})
  // CHECK-SAME: (!ttcore.tile<32x32, bf16>, !ttcore.tile<32x32, si32>)
  // CHECK-SAME: -> (!ttcore.tile<32x32, bf16>, !ttcore.tile<32x32, si32>, !ttcore.tile<32x32, bf16>, !ttcore.tile<32x32, si32>)
  func.func @argmax_reduce_last(%arg0: tensor<32x128xbf16>) -> tensor<32x1xsi32> {
    %0 = "ttir.argmax"(%arg0) <{dim_arg = [1 : i32], keep_dim = true}> : (tensor<32x128xbf16>) -> tensor<32x1xsi32>
    return %0 : tensor<32x1xsi32>
  }

  // Reducing the second-to-last dim. The LLK only ever collapses rows, so the
  // lowering emits a ttir.permute on each end and reduces the transposed
  // tensor; by this point in the pipeline those permutes have themselves been
  // lowered into generics, so what remains visible is that the argmax generic
  // is reached with the same reduction/parallel iterator order as the case
  // above, not a mirrored one.
  // CHECK-LABEL: func @argmax_reduce_second_to_last
  // CHECK: d2m.generic{{.+}}iterator_types = [#reduction, #parallel]
  // CHECK: d2m.tile_argmax
  func.func @argmax_reduce_second_to_last(%arg0: tensor<128x32xbf16>) -> tensor<1x32xsi32> {
    %0 = "ttir.argmax"(%arg0) <{dim_arg = [0 : i32], keep_dim = true}> : (tensor<128x32xbf16>) -> tensor<1x32xsi32>
    return %0 : tensor<1x32xsi32>
  }

  // The index domain is si32, not the bf16 value domain. This matters past a
  // reduction extent of 256, where bf16 can no longer represent every index
  // exactly and the arange would start producing duplicates; 512 sits beyond
  // that point. The mixed dtypes are also what force the DST buffer to be typed
  // by its widest access.
  // CHECK-LABEL: func @argmax_index_type_is_si32
  // CHECK: d2m.tile_argmax
  // CHECK-SAME: !ttcore.tile<32x32, si32>, !ttcore.tile<32x32, bf16>, !ttcore.tile<32x32, si32>)
  // CHECK: return %{{.+}} : tensor<32x1xsi32>
  func.func @argmax_index_type_is_si32(%arg0: tensor<32x512xbf16>) -> tensor<32x1xsi32> {
    %0 = "ttir.argmax"(%arg0) <{dim_arg = [1 : i32], keep_dim = true}> : (tensor<32x512xbf16>) -> tensor<32x1xsi32>
    return %0 : tensor<32x1xsi32>
  }
}
