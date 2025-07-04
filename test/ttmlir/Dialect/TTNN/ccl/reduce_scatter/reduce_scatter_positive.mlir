// RUN: ttmlir-opt --split-input-file --ttir-to-ttnn-backend-pipeline="mesh-shape=1,2" %s | FileCheck %s
// Unit tests for ttnn reduce_scatter op

// Verify lowering of ttir reduce_scatter to ttnn ops

module attributes {} {
  // CHECK-LABEL: reduce_scatter_positive
  func.func @reduce_scatter_positive(%arg0: tensor<1x1x8192x256xf32>) -> tensor<1x1x8192x256xf32> {
    %0 = ttir.empty() : tensor<1x1x8192x256xf32>
    %1 = "ttir.reduce_scatter"(%arg0, %0) <{cluster_axis = 1 : ui32, reduce_type = #ttcore.reduce_type<sum>, scatter_dim = 3 : si32}> : (tensor<1x1x8192x256xf32>, tensor<1x1x8192x256xf32>) -> tensor<1x1x8192x256xf32>
    // CHECK: "ttnn.reduce_scatter"
    return %1 : tensor<1x1x8192x256xf32>
  }
}

// -----

// Verify op folding for single mesh device communication

module attributes {} {
  // CHECK-LABEL: reduce_scatter_positive_folding
  func.func @reduce_scatter_positive_folding(%arg0: tensor<1x1x8192x256xf32>) -> tensor<1x1x8192x256xf32> {
    %0 = ttir.empty() : tensor<1x1x8192x256xf32>
    %1 = "ttir.reduce_scatter"(%arg0, %0) <{cluster_axis = 0 : ui32, reduce_type = #ttcore.reduce_type<sum>, scatter_dim = 3 : si32}> : (tensor<1x1x8192x256xf32>, tensor<1x1x8192x256xf32>) -> tensor<1x1x8192x256xf32>
    // CHECK-NOT: "ttnn.reduce_scatter"
    return %1 : tensor<1x1x8192x256xf32>
  }
}
