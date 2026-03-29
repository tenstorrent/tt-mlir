// REQUIRES: stablehlo
// RUN: ttmlir-opt --reoutline-composite -o %t %s
// RUN: FileCheck %s --input-file=%t

// Verify that ReoutlineCompositePass produces a private decomposition function
// with results in the correct order, as specified by reoutline.output_pos
// annotations from FlattenCompositePass. The topk decomposition has a
// cross-order return: in block order the i64 convert appears before the f32
// slice, but the composite returns (f32, i64). The output_pos attributes
// ensure the reoutlined function preserves the original result order.

// CHECK-LABEL: func.func @main
// CHECK: stablehlo.composite "tenstorrent.topk"
// CHECK-SAME: -> (tensor<128x4xf32>, tensor<128x4xi64>)
module @ReoutlineCompositeResultOrder attributes {} {
  func.func @main(%arg0: tensor<128x32xf32>) -> (tensor<128x4xf32>, tensor<128x4xi64>) {
    %0 = stablehlo.iota dim = 0 {reoutline.comp_attrs = {dim = -1 : i64, k = 4 : i64, largest = true, sorted = true}, reoutline.group = "composite_tenstorrent.topk.impl", reoutline.orig_name = "tenstorrent.topk", reoutline.seed} : tensor<32xi32>
    %1 = stablehlo.broadcast_in_dim %0, dims = [1] {reoutline.group = "composite_tenstorrent.topk.impl"} : (tensor<32xi32>) -> tensor<128x32xi32>
    %2:2 = "stablehlo.sort"(%arg0, %1) <{dimension = 1 : i64}> ({
    ^bb0(%arg1: tensor<f32>, %arg2: tensor<f32>, %arg3: tensor<i32>, %arg4: tensor<i32>):
      %6 = stablehlo.compare  GT, %arg1, %arg2,  TOTALORDER : (tensor<f32>, tensor<f32>) -> tensor<i1>
      stablehlo.return %6 : tensor<i1>
    }) {reoutline.arg_operand_indices = array<i64: 0, -1>, reoutline.group = "composite_tenstorrent.topk.impl"} : (tensor<128x32xf32>, tensor<128x32xi32>) -> (tensor<128x32xf32>, tensor<128x32xi32>)
    %3 = stablehlo.slice %2#1 [0:128, 0:4] {reoutline.group = "composite_tenstorrent.topk.impl"} : (tensor<128x32xi32>) -> tensor<128x4xi32>
    %4 = stablehlo.convert %3 {reoutline.group = "composite_tenstorrent.topk.impl", reoutline.output_pos = 1 : i64} : (tensor<128x4xi32>) -> tensor<128x4xi64>
    %5 = stablehlo.slice %2#0 [0:128, 0:4] {reoutline.group = "composite_tenstorrent.topk.impl", reoutline.output_pos = 0 : i64} : (tensor<128x32xf32>) -> tensor<128x4xf32>
    return %5, %4 : tensor<128x4xf32>, tensor<128x4xi64>
  }
}
// The private function signature should have (f32, i64) result types in order.
// CHECK: func.func private @outlined_composite_tenstorrent.topk.impl
// CHECK-SAME: -> (tensor<128x4xf32>, tensor<128x4xi64>)
// The return should place f32 values before i64 indices, matching the original
// composite result order (not the block order of the defining ops).
// CHECK: %[[INDICES:.*]] = stablehlo.convert
// CHECK-SAME: (tensor<128x4xi32>) -> tensor<128x4xi64>
// CHECK: %[[VALUES:.*]] = stablehlo.slice
// CHECK-SAME: (tensor<128x32xf32>) -> tensor<128x4xf32>
// CHECK: return %[[VALUES]], %[[INDICES]] : tensor<128x4xf32>, tensor<128x4xi64>
