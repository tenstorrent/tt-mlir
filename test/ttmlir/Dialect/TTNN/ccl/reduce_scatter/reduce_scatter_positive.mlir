// RUN: ttmlir-opt --split-input-file --ttir-to-ttnn-backend-pipeline="mesh-shape=1,1" %s | FileCheck %s
// Unit tests for ttnn reduce_scatter op

// -----

// Verify lowering of ttir reduce_scatter to ttnn ops

module attributes {} {
  func.func @reduce_scatter_positive(%arg0: tensor<1x1x8192x256xf32>) -> tensor<1x1x8192x256xf32> {
    %0 = tensor.empty() : tensor<1x1x8192x256xf32>
    %1 = "ttir.reduce_scatter"(%arg0, %0) <{cluster_axis = 1 : ui32, reduce_type = #tt.reduce_type<sum>, scatter_dim = 3 : si32}> : (tensor<1x1x8192x256xf32>, tensor<1x1x8192x256xf32>) -> tensor<1x1x8192x256xf32>
    return %1 : tensor<1x1x8192x256xf32>
  }
}
// CHECK: "ttnn.reduce_scatter"
