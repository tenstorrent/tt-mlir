// REQUIRES: stablehlo
// RUN: ttmlir-opt --flatten-composite -o %t %s
// RUN: FileCheck %s --input-file=%t

// Verify that FlattenCompositePass correctly annotates a multi-result op
// (stablehlo.sort) when both of its results are returned by the composite.
// The reoutline.result_pos array should carry a position for each result.

module @FlattenCompositeMultiResultPos attributes {} {
  sdy.mesh @mesh = <["_axis_0"=2]>
  func.func @test_normal_order(%arg0: tensor<128x32xf32> {sdy.sharding = #sdy.sharding<@mesh, [{}, {"_axis_0"}]>}) -> (tensor<128x32xf32>, tensor<128x32xi32>) {
    %0:2 = stablehlo.composite "tenstorrent.sort_both_normal" %arg0 {composite_attributes = {dimension = 1 : i64}, decomposition = @tenstorrent.sort_both_normal.impl} : (tensor<128x32xf32>) -> (tensor<128x32xf32>, tensor<128x32xi32>)
    return %0#0, %0#1 : tensor<128x32xf32>, tensor<128x32xi32>
  }
  func.func private @tenstorrent.sort_both_normal.impl(%arg0: tensor<128x32xf32>) -> (tensor<128x32xf32>, tensor<128x32xi32>) {
    %0 = stablehlo.iota dim = 0 : tensor<32xi32>
    %1 = stablehlo.broadcast_in_dim %0, dims = [1] : (tensor<32xi32>) -> tensor<128x32xi32>
    %2:2 = "stablehlo.sort"(%arg0, %1) <{dimension = 1 : i64}> ({
    ^bb0(%arg1: tensor<f32>, %arg2: tensor<f32>, %arg3: tensor<i32>, %arg4: tensor<i32>):
      %3 = stablehlo.compare  GT, %arg1, %arg2,  TOTALORDER : (tensor<f32>, tensor<f32>) -> tensor<i1>
      stablehlo.return %3 : tensor<i1>
    }) : (tensor<128x32xf32>, tensor<128x32xi32>) -> (tensor<128x32xf32>, tensor<128x32xi32>)
    return %2#0, %2#1 : tensor<128x32xf32>, tensor<128x32xi32>
  }
  // CHECK-LABEL: func.func @test_normal_order
  // Both results come from the same stablehlo.sort op, so the attribute should
  // be a 2-element array: result #0 -> composite pos 0, result #1 -> composite pos 1.
  // CHECK: stablehlo.sort
  // CHECK: reoutline.result_pos = array<i64: 0, 1>


  func.func @test_reverse_order(%arg0: tensor<128x32xf32> {sdy.sharding = #sdy.sharding<@mesh, [{}, {"_axis_0"}]>}) -> (tensor<128x32xi32>, tensor<128x32xf32>) {
    %0:2 = stablehlo.composite "tenstorrent.sort_both_reversed" %arg0 {composite_attributes = {dimension = 1 : i64}, decomposition = @tenstorrent.sort_both_reversed.impl} : (tensor<128x32xf32>) -> (tensor<128x32xi32>, tensor<128x32xf32>)
    return %0#0, %0#1 : tensor<128x32xi32>, tensor<128x32xf32>
  }
  func.func private @tenstorrent.sort_both_reversed.impl(%arg0: tensor<128x32xf32>) -> (tensor<128x32xi32>, tensor<128x32xf32>) {
    %0 = stablehlo.iota dim = 0 : tensor<32xi32>
    %1 = stablehlo.broadcast_in_dim %0, dims = [1] : (tensor<32xi32>) -> tensor<128x32xi32>
    %2:2 = "stablehlo.sort"(%arg0, %1) <{dimension = 1 : i64}> ({
    ^bb0(%arg1: tensor<f32>, %arg2: tensor<f32>, %arg3: tensor<i32>, %arg4: tensor<i32>):
      %3 = stablehlo.compare  GT, %arg1, %arg2,  TOTALORDER : (tensor<f32>, tensor<f32>) -> tensor<i1>
      stablehlo.return %3 : tensor<i1>
    }) : (tensor<128x32xf32>, tensor<128x32xi32>) -> (tensor<128x32xf32>, tensor<128x32xi32>)
    return %2#1, %2#0 : tensor<128x32xi32>, tensor<128x32xf32>
  }
  // CHECK-LABEL: func.func @test_reverse_order
  // Both results come from the same stablehlo.sort op, so the attribute should
  // be a 2-element array: result #1 -> composite pos 1, result #0 -> composite pos 0.
  // The sort op spans multiple lines (region body), so we match the closing line.
  // CHECK: stablehlo.sort
  // CHECK: reoutline.result_pos = array<i64: 1, 0>


  func.func @test_multiple(%arg0: tensor<128x32xf32> {sdy.sharding = #sdy.sharding<@mesh, [{}, {"_axis_0"}]>}, %arg1: tensor<128x32xf32> {sdy.sharding = #sdy.sharding<@mesh, [{}, {"_axis_0"}]>}) -> (tensor<128x32xi32>, tensor<128x32xf32>, tensor<128x32xf32>, tensor<128x32xi32>, tensor<32xi32>) {
    %0:5 = stablehlo.composite "tenstorrent.sort_multiple" %arg0, %arg1 {composite_attributes = {dimension = 1 : i64}, decomposition = @tenstorrent.sort_multiple.impl} : (tensor<128x32xf32>, tensor<128x32xf32>) -> (tensor<128x32xi32>, tensor<128x32xf32>, tensor<128x32xf32>, tensor<128x32xi32>, tensor<32xi32>)
    return %0#0, %0#1, %0#2, %0#3, %0#4 : tensor<128x32xi32>, tensor<128x32xf32>, tensor<128x32xf32>, tensor<128x32xi32>, tensor<32xi32>
  }
  func.func private @tenstorrent.sort_multiple.impl(%arg0: tensor<128x32xf32>, %arg1: tensor<128x32xf32>) -> (tensor<128x32xi32>, tensor<128x32xf32>, tensor<128x32xf32>, tensor<128x32xi32>, tensor<32xi32>) {
    %0 = stablehlo.iota dim = 0 : tensor<32xi32>
    %1 = stablehlo.broadcast_in_dim %0, dims = [1] : (tensor<32xi32>) -> tensor<128x32xi32>
    %2:2 = "stablehlo.sort"(%arg0, %1) <{dimension = 1 : i64}> ({
    ^bb0(%arg2: tensor<f32>, %arg3: tensor<f32>, %arg4: tensor<i32>, %arg5: tensor<i32>):
      %3 = stablehlo.compare  GT, %arg2, %arg3,  TOTALORDER : (tensor<f32>, tensor<f32>) -> tensor<i1>
      stablehlo.return %3 : tensor<i1>
    }) : (tensor<128x32xf32>, tensor<128x32xi32>) -> (tensor<128x32xf32>, tensor<128x32xi32>)
    %4:2 = "stablehlo.sort"(%arg1, %1) <{dimension = 1 : i64}> ({
    ^bb0(%arg6: tensor<f32>, %arg7: tensor<f32>, %arg8: tensor<i32>, %arg9: tensor<i32>):
      %5 = stablehlo.compare  GT, %arg6, %arg7,  TOTALORDER : (tensor<f32>, tensor<f32>) -> tensor<i1>
      stablehlo.return %5 : tensor<i1>
    }) : (tensor<128x32xf32>, tensor<128x32xi32>) -> (tensor<128x32xf32>, tensor<128x32xi32>)
    return %2#1, %2#0, %4#0, %4#1, %0 : tensor<128x32xi32>, tensor<128x32xf32>, tensor<128x32xf32>, tensor<128x32xi32>, tensor<32xi32>
  }
  // CHECK-LABEL: func.func @test_multiple
  // Now there are 5 results (one from %0 and two each from each sort op)
  // CHECK: %0 = stablehlo.iota
  // CHECK-SAME: reoutline.result_pos = array<i64: 4>
  // CHECK: stablehlo.sort
  // CHECK: reoutline.result_pos = array<i64: 1, 0>
  // CHECK: stablehlo.sort
  // CHECK: reoutline.result_pos = array<i64: 2, 3>
}
