// RUN: ttmlir-opt --split-input-file %s | FileCheck %s
// Unit tests for ttir reduce_scatter op

// -----

// CHECK-LABEL: @reduce_scatter_sum_positive
module attributes {} {
  func.func @reduce_scatter_sum_positive(%arg0: tensor<8192x2048xf32>) -> tensor<8192x1024xf32> {
    %0 = ttir.empty() : tensor<8192x1024xf32>
    // CHECK: ttir.reduce_scatter
    // CHECK-SAME: cluster_axis = 1 : ui32
    // CHECK-SAME: reduce_type = #ttcore.reduce_type<sum>
    // CHECK-SAME: scatter_dim = 1 : si32
    %1 = "ttir.reduce_scatter"(%arg0, %0) <{cluster_axis = 1 : ui32, reduce_type = #ttcore.reduce_type<sum>, scatter_dim = 1 : si32}> : (tensor<8192x2048xf32>, tensor<8192x1024xf32>) -> tensor<8192x1024xf32>
    return %1 : tensor<8192x1024xf32>
  }
}

// -----

// CHECK-LABEL: @reduce_scatter_sum_4d_positive
module attributes {} {
  func.func @reduce_scatter_sum_4d_positive(%arg0: tensor<1x1x8192x2048xf32>) -> tensor<1x1x8192x1024xf32> {
    %0 = ttir.empty() : tensor<1x1x8192x1024xf32>
    // CHECK: ttir.reduce_scatter
    // CHECK-SAME: cluster_axis = 1 : ui32
    // CHECK-SAME: reduce_type = #ttcore.reduce_type<sum>
    // CHECK-SAME: scatter_dim = 3 : si32
    %1 = "ttir.reduce_scatter"(%arg0, %0) <{cluster_axis = 1 : ui32, reduce_type = #ttcore.reduce_type<sum>, scatter_dim = 3 : si32}> : (tensor<1x1x8192x2048xf32>, tensor<1x1x8192x1024xf32>) -> tensor<1x1x8192x1024xf32>
    return %1 : tensor<1x1x8192x1024xf32>
  }
}
