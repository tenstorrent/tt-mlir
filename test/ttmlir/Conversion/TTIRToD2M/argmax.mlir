// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// RUN: ttmlir-opt --ttcore-register-device --ttir-to-d2m --d2m-materialize-view-returns -o %t %s
// RUN: FileCheck %s --input-file=%t

// argmax lowers to d2m.tile_argmax, which maps onto the
// max_reduce_with_indices LLK. The LLK consumes one value tile plus one index
// tile per call and keeps running value/index maxima in DST, so the op has four
// results: the two reduced tiles and the two accumulators.

module {

  // Reducing the last dim. The index operand comes from an arange generic, and
  // both operands are relabelled row-major-as-tile (a pure byte reinterpret,
  // no data movement) because the LLK wants 32x32 chunks in row-major order.
  // CHECK-LABEL: func @argmax_reduce_last
  // CHECK: d2m.arange_block{{.*}}num_elements = 128
  // CHECK: d2m.view_layout{{.*}}reinterpretLayout = true{{.*}}-> tensor<1x1x4x1x!ttcore.tile<32x32, bf16>
  // CHECK: d2m.view_layout{{.*}}reinterpretLayout = true{{.*}}-> tensor<1x1x4x1x!ttcore.tile<32x32, si32>
  // CHECK: d2m.generic{{.+}}iterator_types = [#reduction, #parallel]
  // CHECK: linalg.generic{{.+}}iterator_types = ["reduction", "parallel"]
  // CHECK: %out_values, %out_indices, %val_acc_out, %idx_acc_out = "d2m.tile_argmax"(%{{.+}}, %{{.+}}) <{reduce_dim = #d2m<reduce_dim C>}>
  // CHECK-SAME: (!ttcore.tile<32x32, bf16>, !ttcore.tile<32x32, si32>)
  // CHECK-SAME: -> (!ttcore.tile<32x32, bf16>, !ttcore.tile<32x32, si32>, !ttcore.tile<32x32, bf16>, !ttcore.tile<32x32, si32>)
  func.func @argmax_reduce_last(%arg0: tensor<32x128xbf16>) -> tensor<32x1xsi32> {
    %0 = "ttir.argmax"(%arg0) <{dim_arg = [1 : i32], keep_dim = true}> : (tensor<32x128xbf16>) -> tensor<32x1xsi32>
    return %0 : tensor<32x1xsi32>
  }

  // Reducing the second-to-last dim still reaches the LLK as reduce_dim C: the
  // input is transposed first so the LLK only ever sees one orientation.
  // CHECK-LABEL: func @argmax_reduce_second_to_last
  // CHECK: d2m.generic{{.+}}iterator_types = [#reduction, #parallel]
  // CHECK: d2m.tile_argmax{{.+}}reduce_dim C
  func.func @argmax_reduce_second_to_last(%arg0: tensor<128x32xbf16>) -> tensor<1x32xsi32> {
    %0 = "ttir.argmax"(%arg0) <{dim_arg = [0 : i32], keep_dim = true}> : (tensor<128x32xbf16>) -> tensor<1x32xsi32>
    return %0 : tensor<1x32xsi32>
  }

  // Indices stay si32 end to end; a bf16 index domain would lose exactness past
  // 256. The result of the whole function is si32, not the bf16 value domain.
  // CHECK-LABEL: func @argmax_index_type_is_si32
  // CHECK: d2m.tile_argmax
  // CHECK-SAME: !ttcore.tile<32x32, si32>, !ttcore.tile<32x32, bf16>, !ttcore.tile<32x32, si32>)
  // CHECK: return %{{.+}} : tensor<32x1xsi32>
  func.func @argmax_index_type_is_si32(%arg0: tensor<32x512xbf16>) -> tensor<32x1xsi32> {
    %0 = "ttir.argmax"(%arg0) <{dim_arg = [1 : i32], keep_dim = true}> : (tensor<32x512xbf16>) -> tensor<32x1xsi32>
    return %0 : tensor<32x1xsi32>
  }
}
