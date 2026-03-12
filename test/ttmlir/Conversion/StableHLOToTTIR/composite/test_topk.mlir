// REQUIRES: stablehlo
// RUN: ttmlir-opt --stablehlo-to-ttir-pipeline -o %t %s
// RUN: FileCheck %s --input-file=%t

module @topk_composite_tests attributes {mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} {
  // CHECK-LABEL: func.func @topk_both
  func.func @topk_both(%arg0: tensor<1x40xf32>) -> (tensor<1x5xf32>, tensor<1x5xi64>) {
    // CHECK: %[[VALUES:.*]], %[[INDICES:.*]] = "ttir.topk"(%arg0) <{dim = -1 : i32, k = 5 : i32, largest = true, sorted = true}> : (tensor<1x40xf32>) -> (tensor<1x5xf32>, tensor<1x5xi64>)
    // CHECK: return %[[VALUES]], %[[INDICES]]
    %0:2 = stablehlo.composite "tenstorrent.topk" %arg0 {composite_attributes = {dim = -1 : i64, k = 5 : i64, largest = true, sorted = true}, decomposition = @tenstorrent.topk.impl} : (tensor<1x40xf32>) -> (tensor<1x5xf32>, tensor<1x5xi64>)
    return %0#0, %0#1 : tensor<1x5xf32>, tensor<1x5xi64>
  }
  func.func private @tenstorrent.topk.impl(%arg0: tensor<1x40xf32>) -> (tensor<1x5xf32>, tensor<1x5xi64>) {
    %0 = stablehlo.iota dim = 0 : tensor<40xi32>
    %1 = stablehlo.reshape %0 : (tensor<40xi32>) -> tensor<1x40xi32>
    %2:2 = "stablehlo.sort"(%arg0, %1) <{dimension = 1 : i64}> ({
    ^bb0(%arg1: tensor<f32>, %arg2: tensor<f32>, %arg3: tensor<i32>, %arg4: tensor<i32>):
      %6 = stablehlo.compare  GT, %arg1, %arg2,  TOTALORDER : (tensor<f32>, tensor<f32>) -> tensor<i1>
      stablehlo.return %6 : tensor<i1>
    }) : (tensor<1x40xf32>, tensor<1x40xi32>) -> (tensor<1x40xf32>, tensor<1x40xi32>)
    %3 = stablehlo.slice %2#0 [0:1, 0:5] : (tensor<1x40xf32>) -> tensor<1x5xf32>
    %4 = stablehlo.slice %2#1 [0:1, 0:5] : (tensor<1x40xi32>) -> tensor<1x5xi32>
    %5 = stablehlo.convert %4 : (tensor<1x5xi32>) -> tensor<1x5xi64>
    return %3, %5 : tensor<1x5xf32>, tensor<1x5xi64>
  }

  // CHECK-LABEL: func.func @topk_values_only
  func.func @topk_values_only(%arg0: tensor<1x40xf32>) -> tensor<1x5xf32> {
    // CHECK: %[[VALUES:.*]], %{{.*}} = "ttir.topk"(%arg0) <{dim = -1 : i32, k = 5 : i32, largest = true, sorted = true}> : (tensor<1x40xf32>) -> (tensor<1x5xf32>, tensor<1x5xi32>)
    // CHECK: return %[[VALUES]]
    %0 = stablehlo.composite "tenstorrent.topk_values" %arg0 {composite_attributes = {dim = -1 : i64, k = 5 : i64, largest = true, sorted = true}, decomposition = @tenstorrent.topk_values.impl} : (tensor<1x40xf32>) -> tensor<1x5xf32>
    return %0 : tensor<1x5xf32>
  }
  func.func private @tenstorrent.topk_values.impl(%arg0: tensor<1x40xf32>) -> tensor<1x5xf32> {
    %0 = "stablehlo.sort"(%arg0) <{dimension = 1 : i64}> ({
    ^bb0(%arg1: tensor<f32>, %arg2: tensor<f32>):
      %2 = stablehlo.compare  GT, %arg1, %arg2,  TOTALORDER : (tensor<f32>, tensor<f32>) -> tensor<i1>
      stablehlo.return %2 : tensor<i1>
    }) : (tensor<1x40xf32>) -> tensor<1x40xf32>
    %1 = stablehlo.slice %0 [0:1, 0:5] : (tensor<1x40xf32>) -> tensor<1x5xf32>
    return %1 : tensor<1x5xf32>
  }

  // CHECK-LABEL: func.func @topk_indices_only
  func.func @topk_indices_only(%arg0: tensor<1x40xf32>) -> tensor<1x5xi64> {
    // CHECK: %{{.*}}, %[[INDICES:.*]] = "ttir.topk"(%arg0) <{dim = -1 : i32, k = 5 : i32, largest = true, sorted = true}> : (tensor<1x40xf32>) -> (tensor<1x5xf32>, tensor<1x5xi64>)
    // CHECK: return %[[INDICES]]
    %0 = stablehlo.composite "tenstorrent.topk_indices" %arg0 {composite_attributes = {dim = -1 : i64, k = 5 : i64, largest = true, sorted = true}, decomposition = @tenstorrent.topk_indices.impl} : (tensor<1x40xf32>) -> tensor<1x5xi64>
    return %0 : tensor<1x5xi64>
  }
  func.func private @tenstorrent.topk_indices.impl(%arg0: tensor<1x40xf32>) -> tensor<1x5xi64> {
    %0 = stablehlo.iota dim = 0 : tensor<40xi32>
    %1 = stablehlo.reshape %0 : (tensor<40xi32>) -> tensor<1x40xi32>
    %2:2 = "stablehlo.sort"(%arg0, %1) <{dimension = 1 : i64}> ({
    ^bb0(%arg1: tensor<f32>, %arg2: tensor<f32>, %arg3: tensor<i32>, %arg4: tensor<i32>):
      %5 = stablehlo.compare  GT, %arg1, %arg2,  TOTALORDER : (tensor<f32>, tensor<f32>) -> tensor<i1>
      stablehlo.return %5 : tensor<i1>
    }) : (tensor<1x40xf32>, tensor<1x40xi32>) -> (tensor<1x40xf32>, tensor<1x40xi32>)
    %3 = stablehlo.slice %2#1 [0:1, 0:5] : (tensor<1x40xi32>) -> tensor<1x5xi32>
    %4 = stablehlo.convert %3 : (tensor<1x5xi32>) -> tensor<1x5xi64>
    return %4 : tensor<1x5xi64>
  }
}
