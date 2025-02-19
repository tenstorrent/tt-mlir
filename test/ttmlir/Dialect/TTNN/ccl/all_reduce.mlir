// RUN: ttmlir-opt --split-input-file --ttir-to-ttnn-backend-pipeline="mesh-shape=1,1" %s | FileCheck %s
// Unit tests for ttnn all_reduce op

// Verify lowering of ttir all_reduce to ttnn ops
module attributes {} {
  func.func @all_reduce(%arg0: tensor<4096x16384xf32>) -> tensor<4096x16384xf32> {
    %0 = tensor.empty() : tensor<4096x16384xf32>
    %1 = "ttir.all_reduce"(%arg0, %0) <{all_gather_dim = 0 : si32, cluster_axis = 1 : ui32, reduce_type = #tt.reduce_type<sum>, scatter_dim = 0 : si32}> : (tensor<4096x16384xf32>, tensor<4096x16384xf32>) -> tensor<4096x16384xf32>
    return %1 : tensor<4096x16384xf32>
  }
}
// CHECK: %[[C:.*]] = "ttnn.reshape"[[C:.*]]
// CHECK: "ttnn.reduce_scatter"
// CHECK: "ttnn.all_gather"
// CHECK: %[[C:.*]] = "ttnn.reshape"[[C:.*]]

// -----

// Verify lowering of ttir all_reduce to ttnn ops
module attributes {} {
  func.func @all_reduce(%arg0: tensor<1x1x4096x16384xf32>) -> tensor<1x1x4096x16384xf32> {
    %0 = tensor.empty() : tensor<1x1x4096x16384xf32>
    %1 = "ttir.all_reduce"(%arg0, %0) <{all_gather_dim = 0 : si32, cluster_axis = 1 : ui32, reduce_type = #tt.reduce_type<sum>, scatter_dim = 0 : si32}> : (tensor<1x1x4096x16384xf32>, tensor<1x1x4096x16384xf32>) -> tensor<1x1x4096x16384xf32>
    return %1 : tensor<1x1x4096x16384xf32>
  }
}
// CHECK-NOT: %[[C:.*]] = "ttnn.reshape"[[C:.*]]
// CHECK: "ttnn.reduce_scatter"
// CHECK: "ttnn.all_gather"
// CHECK-NOT: %[[C:.*]] = "ttnn.reshape"[[C:.*]]
