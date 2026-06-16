// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// RUN: ttmlir-opt --convert-ttnn-to-emitpy -o %t.mlir %s
// RUN: ttmlir-translate --mlir-to-python -o %t.py %t.mlir
// RUN: FileCheck %s --input-file=%t.py

// Regression test for negative dim/axis emission in TTNN->EmitPy.
// These ops declare their dim as a signless I32Attr/I64Attr, so the generated
// getter returns an unsigned value; without a signed cast a dim of -1 would be
// emitted as 4294967295 (or 18446744073709551615 for the i64 dim_arg). The
// conversion casts to a signed type so the negative dim is preserved as -1.

#dram = #ttnn.buffer_type<dram>
#tile_2d_f32 = #ttnn.ttnn_layout<(d0, d1) -> (d0, d1), <1x1>, memref<1x1x!ttcore.tile<32x32, f32>, #dram>, <interleaved>>
#tile_2d_u32 = #ttnn.ttnn_layout<(d0, d1) -> (d0, d1), <1x1>, memref<1x1x!ttcore.tile<32x32, u32>, #dram>, <interleaved>>
#tile_2d_si32 = #ttnn.ttnn_layout<(d0, d1) -> (d0, d1), <1x1>, memref<1x1x!ttcore.tile<32x32, si32>, #dram>, <interleaved>>
#tile_4d_f32 = #ttnn.ttnn_layout<(d0, d1, d2, d3) -> (d0 * 1024 + d1 * 32 + d2, d3), <1x1>, memref<32x1x!ttcore.tile<32x32, f32>, #dram>, <interleaved>>
#tile_4d_f32_x64 = #ttnn.ttnn_layout<(d0, d1, d2, d3) -> (d0 * 1024 + d1 * 32 + d2, d3), <1x1>, memref<32x2x!ttcore.tile<32x32, f32>, #dram>, <interleaved>>
#tile_4d_u32 = #ttnn.ttnn_layout<(d0, d1, d2, d3) -> (d0 * 1024 + d1 * 32 + d2, d3), <1x1>, memref<32x1x!ttcore.tile<32x32, u32>, #dram>, <interleaved>>
#row_major_3d_u32 = #ttnn.ttnn_layout<(d0, d1, d2) -> (d0, d1, d2), <1x1>, memref<1x1xui32, #dram>, <interleaved>>
#row_major_3d_f32 = #ttnn.ttnn_layout<(d0, d1, d2) -> (d0, d1, d2), <1x1>, memref<1x1xf32, #dram>, <interleaved>>
#tile_1d_bf16 = #ttnn.ttnn_layout<(d0) -> (0, d0), <1x1>, memref<1x1x!ttcore.tile<32x32, bf16>, #dram>, <interleaved>>

// CHECK-LABEL: def scatter_neg
func.func @scatter_neg(%a: tensor<32x32xf32, #tile_2d_f32>, %i: tensor<32x32xi32, #tile_2d_si32>, %s: tensor<32x32xf32, #tile_2d_f32>) -> tensor<32x32xf32, #tile_2d_f32> {
  // CHECK: ttnn.scatter(
  // CHECK-SAME: dim=-1
  %0 = "ttnn.scatter"(%a, %i, %s) <{dim = -1 : i32, scatter_reduce_type = #ttcore.reduce_type<invalid>}> : (tensor<32x32xf32, #tile_2d_f32>, tensor<32x32xi32, #tile_2d_si32>, tensor<32x32xf32, #tile_2d_f32>) -> tensor<32x32xf32, #tile_2d_f32>
  return %0 : tensor<32x32xf32, #tile_2d_f32>
}

// CHECK-LABEL: def cumsum_neg
func.func @cumsum_neg(%a: tensor<32x32xf32, #tile_2d_f32>) -> tensor<32x32xf32, #tile_2d_f32> {
  // CHECK: ttnn.cumsum(
  // CHECK-SAME: -1
  %0 = "ttnn.cumsum"(%a) <{dim = -1 : i32}> : (tensor<32x32xf32, #tile_2d_f32>) -> tensor<32x32xf32, #tile_2d_f32>
  return %0 : tensor<32x32xf32, #tile_2d_f32>
}

// CHECK-LABEL: def cumprod_neg
func.func @cumprod_neg(%a: tensor<32x32xf32, #tile_2d_f32>) -> tensor<32x32xf32, #tile_2d_f32> {
  // CHECK: ttnn.cumprod(
  // CHECK-SAME: -1
  %0 = "ttnn.cumprod"(%a) <{dim = -1 : i32}> : (tensor<32x32xf32, #tile_2d_f32>) -> tensor<32x32xf32, #tile_2d_f32>
  return %0 : tensor<32x32xf32, #tile_2d_f32>
}

// CHECK-LABEL: def argmax_neg
func.func @argmax_neg(%a: tensor<1x1x32x32xf32, #tile_4d_f32>) -> tensor<1x1x32xui32, #row_major_3d_u32> {
  // CHECK: ttnn.argmax(
  // CHECK-SAME: -1
  %0 = "ttnn.argmax"(%a) <{dim = -1 : i32, keep_dim = false, use_multicore = false}> : (tensor<1x1x32x32xf32, #tile_4d_f32>) -> tensor<1x1x32xui32, #row_major_3d_u32>
  return %0 : tensor<1x1x32xui32, #row_major_3d_u32>
}

// CHECK-LABEL: def prod_neg
func.func @prod_neg(%a: tensor<1x1x32x32xf32, #tile_4d_f32>) -> tensor<1x1x32xf32, #row_major_3d_f32> {
  // CHECK: ttnn.prod(
  // CHECK-SAME: -1
  %0 = "ttnn.prod"(%a) <{dim_arg = -1 : i64, keep_dim = false}> : (tensor<1x1x32x32xf32, #tile_4d_f32>) -> tensor<1x1x32xf32, #row_major_3d_f32>
  return %0 : tensor<1x1x32xf32, #row_major_3d_f32>
}

// CHECK-LABEL: def topk_neg
func.func @topk_neg(%a: tensor<1x1x32x64xf32, #tile_4d_f32_x64>) -> (tensor<1x1x32x4xf32, #tile_4d_f32>, tensor<1x1x32x4xui32, #tile_4d_u32>) {
  // CHECK: ttnn.topk(
  // CHECK-SAME: -1
  %v, %i = "ttnn.topk"(%a) <{k = 4 : i32, dim = -1 : i32, largest = true, sorted = true}> : (tensor<1x1x32x64xf32, #tile_4d_f32_x64>) -> (tensor<1x1x32x4xf32, #tile_4d_f32>, tensor<1x1x32x4xui32, #tile_4d_u32>)
  return %v, %i : tensor<1x1x32x4xf32, #tile_4d_f32>, tensor<1x1x32x4xui32, #tile_4d_u32>
}

// arange start/step are signless I64Attr (getter returns uint64_t); a negative
// start would otherwise be emitted as 18446744073709551611.
// CHECK-LABEL: def arange_neg
func.func @arange_neg() -> tensor<32xbf16, #tile_1d_bf16> {
  // CHECK: ttnn.arange(
  // CHECK-SAME: -5
  %0 = "ttnn.arange"() <{start = -5 : si64, end = 27 : si64, step = 1 : si64, layout = #ttnn.layout<tile>}> : () -> tensor<32xbf16, #tile_1d_bf16>
  return %0 : tensor<32xbf16, #tile_1d_bf16>
}
