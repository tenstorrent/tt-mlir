// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// RUN: ttmlir-opt --convert-ttnn-to-emitc -o %t.mlir %s
// RUN: ttmlir-translate --mlir-to-cpp %t.mlir | FileCheck %s

// Regression test for negative dim/axis emission in TTNN->EmitC.
// These ops declare their dim/dim_arg as a signed SI32Attr/SI64Attr, so the
// generated getter returns a signed value and a dim of -1 is emitted directly
// as -1.

#dram = #ttnn.buffer_type<dram>
#tile_2d_f32 = #ttnn.ttnn_layout<(d0, d1) -> (d0, d1), <1x1>, memref<1x1x!ttcore.tile<32x32, f32>, #dram>, <interleaved>>
#tile_2d_u32 = #ttnn.ttnn_layout<(d0, d1) -> (d0, d1), <1x1>, memref<1x1x!ttcore.tile<32x32, u32>, #dram>, <interleaved>>
#tile_2d_si32 = #ttnn.ttnn_layout<(d0, d1) -> (d0, d1), <1x1>, memref<1x1x!ttcore.tile<32x32, si32>, #dram>, <interleaved>>
#tile_4d_f32 = #ttnn.ttnn_layout<(d0, d1, d2, d3) -> (d0 * 1024 + d1 * 32 + d2, d3), <1x1>, memref<32x1x!ttcore.tile<32x32, f32>, #dram>, <interleaved>>
#tile_4d_f32_x64 = #ttnn.ttnn_layout<(d0, d1, d2, d3) -> (d0 * 1024 + d1 * 32 + d2, d3), <1x1>, memref<32x2x!ttcore.tile<32x32, f32>, #dram>, <interleaved>>
#tile_4d_u32 = #ttnn.ttnn_layout<(d0, d1, d2, d3) -> (d0 * 1024 + d1 * 32 + d2, d3), <1x1>, memref<32x1x!ttcore.tile<32x32, u32>, #dram>, <interleaved>>
#row_major_3d_u32 = #ttnn.ttnn_layout<(d0, d1, d2) -> (d0, d1, d2), <1x1>, memref<1x1xui32, #dram>, <interleaved>>
#row_major_3d_f32 = #ttnn.ttnn_layout<(d0, d1, d2) -> (d0, d1, d2), <1x1>, memref<1x1xf32, #dram>, <interleaved>>

// CHECK-LABEL: gather_neg
func.func @gather_neg(%a: tensor<32x32xf32, #tile_2d_f32>, %i: tensor<32x32xui32, #tile_2d_u32>) -> tensor<32x32xf32, #tile_2d_f32> {
  // CHECK: ttnn::gather(
  // CHECK-SAME: -1
  %0 = "ttnn.gather"(%a, %i) <{dim = -1 : si32}> : (tensor<32x32xf32, #tile_2d_f32>, tensor<32x32xui32, #tile_2d_u32>) -> tensor<32x32xf32, #tile_2d_f32>
  return %0 : tensor<32x32xf32, #tile_2d_f32>
}

// CHECK-LABEL: scatter_neg
func.func @scatter_neg(%a: tensor<32x32xf32, #tile_2d_f32>, %i: tensor<32x32xi32, #tile_2d_si32>, %s: tensor<32x32xf32, #tile_2d_f32>) -> tensor<32x32xf32, #tile_2d_f32> {
  // CHECK: ttnn::scatter(
  // CHECK-SAME: -1
  %0 = "ttnn.scatter"(%a, %i, %s) <{dim = -1 : si32, scatter_reduce_type = #ttcore.reduce_type<invalid>}> : (tensor<32x32xf32, #tile_2d_f32>, tensor<32x32xi32, #tile_2d_si32>, tensor<32x32xf32, #tile_2d_f32>) -> tensor<32x32xf32, #tile_2d_f32>
  return %0 : tensor<32x32xf32, #tile_2d_f32>
}

// CHECK-LABEL: cumsum_neg
func.func @cumsum_neg(%a: tensor<32x32xf32, #tile_2d_f32>) -> tensor<32x32xf32, #tile_2d_f32> {
  // CHECK: ttnn::cumsum(
  // CHECK-SAME: -1
  %0 = "ttnn.cumsum"(%a) <{dim = -1 : si32}> : (tensor<32x32xf32, #tile_2d_f32>) -> tensor<32x32xf32, #tile_2d_f32>
  return %0 : tensor<32x32xf32, #tile_2d_f32>
}

// CHECK-LABEL: cumprod_neg
func.func @cumprod_neg(%a: tensor<32x32xf32, #tile_2d_f32>) -> tensor<32x32xf32, #tile_2d_f32> {
  // CHECK: ttnn::cumprod(
  // CHECK-SAME: -1
  %0 = "ttnn.cumprod"(%a) <{dim = -1 : si32}> : (tensor<32x32xf32, #tile_2d_f32>) -> tensor<32x32xf32, #tile_2d_f32>
  return %0 : tensor<32x32xf32, #tile_2d_f32>
}

// CHECK-LABEL: argmax_neg
func.func @argmax_neg(%a: tensor<1x1x32x32xf32, #tile_4d_f32>) -> tensor<1x1x32xui32, #row_major_3d_u32> {
  // CHECK: ttnn::argmax(
  // CHECK-SAME: -1
  %0 = "ttnn.argmax"(%a) <{dim = -1 : si32, keep_dim = false, use_multicore = false}> : (tensor<1x1x32x32xf32, #tile_4d_f32>) -> tensor<1x1x32xui32, #row_major_3d_u32>
  return %0 : tensor<1x1x32xui32, #row_major_3d_u32>
}

// CHECK-LABEL: prod_neg
func.func @prod_neg(%a: tensor<1x1x32x32xf32, #tile_4d_f32>) -> tensor<1x1x32xf32, #row_major_3d_f32> {
  // CHECK: ttnn::prod(
  // CHECK-SAME: -1
  %0 = "ttnn.prod"(%a) <{dim_arg = -1 : si64, keep_dim = false}> : (tensor<1x1x32x32xf32, #tile_4d_f32>) -> tensor<1x1x32xf32, #row_major_3d_f32>
  return %0 : tensor<1x1x32xf32, #row_major_3d_f32>
}

// CHECK-LABEL: topk_neg
func.func @topk_neg(%a: tensor<1x1x32x64xf32, #tile_4d_f32_x64>) -> (tensor<1x1x32x4xf32, #tile_4d_f32>, tensor<1x1x32x4xui32, #tile_4d_u32>) {
  // CHECK: ttnn::topk(
  // CHECK-SAME: -1
  %v, %i = "ttnn.topk"(%a) <{k = 4 : i32, dim = -1 : si32, largest = true, sorted = true}> : (tensor<1x1x32x64xf32, #tile_4d_f32_x64>) -> (tensor<1x1x32x4xf32, #tile_4d_f32>, tensor<1x1x32x4xui32, #tile_4d_u32>)
  return %v, %i : tensor<1x1x32x4xf32, #tile_4d_f32>, tensor<1x1x32x4xui32, #tile_4d_u32>
}
