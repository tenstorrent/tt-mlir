// REQUIRES: stablehlo
// RUN: ttmlir-opt --reoutline-composite -o %t %s
// RUN: FileCheck %s --input-file=%t

// Verify that ReoutlineCompositePass correctly reads reoutline.result_pos from
// multi-result ops and produces the outlined function with results in the
// original composite order.

// CHECK-LABEL: func.func @test_normal_order
// CHECK: stablehlo.composite "tenstorrent.sort_both_normal"
// CHECK-SAME: -> (tensor<128x32xf32>, tensor<128x32xi32>)

// CHECK-LABEL: func.func @test_reverse_order
// CHECK: stablehlo.composite "tenstorrent.sort_both_reversed"
// CHECK-SAME: -> (tensor<128x32xi32>, tensor<128x32xf32>)

// CHECK-LABEL: func.func @test_multiple
// CHECK: stablehlo.composite "tenstorrent.sort_multiple"
// CHECK-SAME: -> (tensor<128x32xi32>, tensor<128x32xf32>, tensor<128x32xf32>, tensor<128x32xi32>, tensor<32xi32>)

module @ReoutlineCompositeMultiResultPos {
  func.func @test_normal_order(%arg0: tensor<128x32xf32>) -> (tensor<128x32xf32>, tensor<128x32xi32>) {
    %0 = stablehlo.iota dim = 0 {reoutline.comp_attrs = {dimension = 1 : i64}, reoutline.group = "composite_tenstorrent.sort_both_normal.impl", reoutline.orig_name = "tenstorrent.sort_both_normal", reoutline.seed} : tensor<32xi32>
    %1 = stablehlo.broadcast_in_dim %0, dims = [1] {reoutline.group = "composite_tenstorrent.sort_both_normal.impl"} : (tensor<32xi32>) -> tensor<128x32xi32>
    %2:2 = "stablehlo.sort"(%arg0, %1) <{dimension = 1 : i64}> ({
    ^bb0(%arg1: tensor<f32>, %arg2: tensor<f32>, %arg3: tensor<i32>, %arg4: tensor<i32>):
      %3 = stablehlo.compare  GT, %arg1, %arg2,  TOTALORDER : (tensor<f32>, tensor<f32>) -> tensor<i1>
      stablehlo.return %3 : tensor<i1>
    }) {reoutline.arg_operand_indices = array<i64: 0, -1>, reoutline.group = "composite_tenstorrent.sort_both_normal.impl", reoutline.result_pos = array<i64: 0, 1>} : (tensor<128x32xf32>, tensor<128x32xi32>) -> (tensor<128x32xf32>, tensor<128x32xi32>)
    return %2#0, %2#1 : tensor<128x32xf32>, tensor<128x32xi32>
  }
  // test_normal_order: Both results in normal order (result_pos = [0, 1]).
  // The outlined function should return (f32, i32) matching the composite.
  // CHECK: func.func private @outlined_composite_tenstorrent.sort_both_normal.impl
  // CHECK-SAME: -> (tensor<128x32xf32>, tensor<128x32xi32>)
  // CHECK: %[[SORT_NORMAL:.*]]:2 = "stablehlo.sort"
  // CHECK: return %[[SORT_NORMAL]]#0, %[[SORT_NORMAL]]#1 : tensor<128x32xf32>, tensor<128x32xi32>

  func.func @test_reverse_order(%arg0: tensor<128x32xf32>) -> (tensor<128x32xi32>, tensor<128x32xf32>) {
    %0 = stablehlo.iota dim = 0 {reoutline.comp_attrs = {dimension = 1 : i64}, reoutline.group = "composite_tenstorrent.sort_both_reversed.impl", reoutline.orig_name = "tenstorrent.sort_both_reversed", reoutline.seed} : tensor<32xi32>
    %1 = stablehlo.broadcast_in_dim %0, dims = [1] {reoutline.group = "composite_tenstorrent.sort_both_reversed.impl"} : (tensor<32xi32>) -> tensor<128x32xi32>
    %2:2 = "stablehlo.sort"(%arg0, %1) <{dimension = 1 : i64}> ({
    ^bb0(%arg1: tensor<f32>, %arg2: tensor<f32>, %arg3: tensor<i32>, %arg4: tensor<i32>):
      %3 = stablehlo.compare  GT, %arg1, %arg2,  TOTALORDER : (tensor<f32>, tensor<f32>) -> tensor<i1>
      stablehlo.return %3 : tensor<i1>
    }) {reoutline.arg_operand_indices = array<i64: 0, -1>, reoutline.group = "composite_tenstorrent.sort_both_reversed.impl", reoutline.result_pos = array<i64: 1, 0>} : (tensor<128x32xf32>, tensor<128x32xi32>) -> (tensor<128x32xf32>, tensor<128x32xi32>)
    return %2#1, %2#0 : tensor<128x32xi32>, tensor<128x32xf32>
  }
  // test_reverse_order: Results are swapped (result_pos = [1, 0]).
  // The outlined function should return (i32, f32) — sort#1 before sort#0.
  // CHECK: func.func private @outlined_composite_tenstorrent.sort_both_reversed.impl
  // CHECK-SAME: -> (tensor<128x32xi32>, tensor<128x32xf32>)
  // CHECK: %[[SORT_REV:.*]]:2 = "stablehlo.sort"
  // CHECK: return %[[SORT_REV]]#1, %[[SORT_REV]]#0 : tensor<128x32xi32>, tensor<128x32xf32>
  func.func @test_multiple(%arg0: tensor<128x32xf32>, %arg1: tensor<128x32xf32>) -> (tensor<128x32xi32>, tensor<128x32xf32>, tensor<128x32xf32>, tensor<128x32xi32>, tensor<32xi32>) {
    %0 = stablehlo.iota dim = 0 {reoutline.comp_attrs = {dimension = 1 : i64}, reoutline.group = "composite_tenstorrent.sort_multiple.impl", reoutline.orig_name = "tenstorrent.sort_multiple", reoutline.result_pos = array<i64: 4>, reoutline.seed} : tensor<32xi32>
    %1 = stablehlo.broadcast_in_dim %0, dims = [1] {reoutline.group = "composite_tenstorrent.sort_multiple.impl"} : (tensor<32xi32>) -> tensor<128x32xi32>
    %2:2 = "stablehlo.sort"(%arg0, %1) <{dimension = 1 : i64}> ({
    ^bb0(%arg2: tensor<f32>, %arg3: tensor<f32>, %arg4: tensor<i32>, %arg5: tensor<i32>):
      %4 = stablehlo.compare  GT, %arg2, %arg3,  TOTALORDER : (tensor<f32>, tensor<f32>) -> tensor<i1>
      stablehlo.return %4 : tensor<i1>
    }) {reoutline.arg_operand_indices = array<i64: 0, -1>, reoutline.group = "composite_tenstorrent.sort_multiple.impl", reoutline.result_pos = array<i64: 1, 0>} : (tensor<128x32xf32>, tensor<128x32xi32>) -> (tensor<128x32xf32>, tensor<128x32xi32>)
    %3:2 = "stablehlo.sort"(%arg1, %1) <{dimension = 1 : i64}> ({
    ^bb0(%arg2: tensor<f32>, %arg3: tensor<f32>, %arg4: tensor<i32>, %arg5: tensor<i32>):
      %4 = stablehlo.compare  GT, %arg2, %arg3,  TOTALORDER : (tensor<f32>, tensor<f32>) -> tensor<i1>
      stablehlo.return %4 : tensor<i1>
    }) {reoutline.arg_operand_indices = array<i64: 1, -1>, reoutline.group = "composite_tenstorrent.sort_multiple.impl", reoutline.result_pos = array<i64: 2, 3>} : (tensor<128x32xf32>, tensor<128x32xi32>) -> (tensor<128x32xf32>, tensor<128x32xi32>)
    return %2#1, %2#0, %3#0, %3#1, %0 : tensor<128x32xi32>, tensor<128x32xf32>, tensor<128x32xf32>, tensor<128x32xi32>, tensor<32xi32>
  }
  // test_multiple: 5 results from 3 ops (iota, sort, sort) with mixed ordering.
  // The outlined function should return all 5 results in composite order.
  // CHECK: func.func private @outlined_composite_tenstorrent.sort_multiple.impl
  // CHECK-SAME: -> (tensor<128x32xi32>, tensor<128x32xf32>, tensor<128x32xf32>, tensor<128x32xi32>, tensor<32xi32>)
  // CHECK: %[[IOTA:.*]] = stablehlo.iota
  // CHECK: %[[SORT1:.*]]:2 = "stablehlo.sort"
  // CHECK: %[[SORT2:.*]]:2 = "stablehlo.sort"
  // CHECK: return %[[SORT1]]#1, %[[SORT1]]#0, %[[SORT2]]#0, %[[SORT2]]#1, %[[IOTA]] : tensor<128x32xi32>, tensor<128x32xf32>, tensor<128x32xf32>, tensor<128x32xi32>, tensor<32xi32>
}
