// REQUIRES: stablehlo
// RUN: ttmlir-opt --flatten-composite -o %t %s
// RUN: FileCheck %s --input-file=%t

// Verify that FlattenCompositePass annotates result-producing ops with
// reoutline.result_pos to preserve the original composite result order.

module @FlattenCompositeOutputPos attributes {} {
  sdy.mesh @mesh = <["_axis_0"=2]>
  func.func @main(%arg0: tensor<128x32xf32> {sdy.sharding = #sdy.sharding<@mesh, [{}, {"_axis_0"}]>}) -> (tensor<128x4xf32>, tensor<128x4xi64>) {
    %0:2 = stablehlo.composite "tenstorrent.topk" %arg0 {composite_attributes = {dim = -1 : i64, k = 4 : i64, largest = true, sorted = true}, decomposition = @tenstorrent.topk.impl} : (tensor<128x32xf32>) -> (tensor<128x4xf32>, tensor<128x4xi64>)
    return %0#0, %0#1 : tensor<128x4xf32>, tensor<128x4xi64>
  }
  func.func private @tenstorrent.topk.impl(%arg0: tensor<128x32xf32>) -> (tensor<128x4xf32>, tensor<128x4xi64>) {
    %0 = stablehlo.iota dim = 0 : tensor<32xi32>
    %1 = stablehlo.broadcast_in_dim %0, dims = [1] : (tensor<32xi32>) -> tensor<128x32xi32>
    %2:2 = "stablehlo.sort"(%arg0, %1) <{dimension = 1 : i64}> ({
    ^bb0(%arg1: tensor<f32>, %arg2: tensor<f32>, %arg3: tensor<i32>, %arg4: tensor<i32>):
      %6 = stablehlo.compare  GT, %arg1, %arg2,  TOTALORDER : (tensor<f32>, tensor<f32>) -> tensor<i1>
      stablehlo.return %6 : tensor<i1>
    }) : (tensor<128x32xf32>, tensor<128x32xi32>) -> (tensor<128x32xf32>, tensor<128x32xi32>)
    %3 = stablehlo.slice %2#1 [0:128, 0:4] : (tensor<128x32xi32>) -> tensor<128x4xi32>
    %4 = stablehlo.convert %3 : (tensor<128x4xi32>) -> tensor<128x4xi64>
    %5 = stablehlo.slice %2#0 [0:128, 0:4] : (tensor<128x32xf32>) -> tensor<128x4xf32>
    return %5, %4 : tensor<128x4xf32>, tensor<128x4xi64>
  }
}
// CHECK-LABEL: func.func @main
// CHECK: %[[INDICES:.*]] = stablehlo.convert
// The INDICES result (%4) appears second in @tenstorrent.topk.impl's result list
// so it should have result_pos = 1
// CHECK-SAME: reoutline.result_pos = 1 : i64

// The VALUES result (%5) appears first in @tenstorrent.topk.impl's result list
// so it should have result_pos = 0
// CHECK: %[[VALUES:.*]] = stablehlo.slice
// CHECK-SAME: reoutline.result_pos = 0 : i64
// CHECK: return %[[VALUES]], %[[INDICES]] : tensor<128x4xf32>, tensor<128x4xi64>
