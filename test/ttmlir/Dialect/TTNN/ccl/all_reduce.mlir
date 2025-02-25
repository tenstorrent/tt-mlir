// RUN: ttmlir-opt --split-input-file --ttir-to-ttnn-backend-pipeline %s | FileCheck %s
// Unit tests for ttnn all_reduce op

// Verify lowering of ttir all_reduce to ttnn ops
module attributes {} {
  func.func @all_reduce(%arg0: tensor<4096x16384xf32>) -> tensor<4096x16384xf32> {
    %0 = tensor.empty() : tensor<4096x16384xf32>
    %1 = "ttir.all_reduce"(%arg0, %0) <{channel_handle = 1 : si32, dim = 0 : si32, reduce_type = #tt.reduce_type<sum>, replica_groups = dense<[[0, 1, 2, 3], [4, 5, 6, 7]]> : tensor<2x4xi64>, use_global_device_ids}> : (tensor<4096x16384xf32>, tensor<4096x16384xf32>) -> tensor<4096x16384xf32>
    return %1 : tensor<4096x16384xf32>
  }
}
// CHECK: = "ttnn.reshape"
// CHECK: "ttnn.reduce_scatter"
// CHECK: "ttnn.all_gather"
// CHECK: = "ttnn.reshape"

// -----

// Verify lowering of ttir all_reduce to ttnn ops
module attributes {} {
  func.func @all_reduce(%arg0: tensor<1x1x4096x16384xf32>) -> tensor<1x1x4096x16384xf32> {
    %0 = tensor.empty() : tensor<1x1x4096x16384xf32>
    %1 = "ttir.all_reduce"(%arg0, %0) <{channel_handle = 1 : si32, dim = 0 : si32, reduce_type = #tt.reduce_type<sum>, replica_groups = dense<[[0, 1, 2, 3], [4, 5, 6, 7]]> : tensor<2x4xi64>, use_global_device_ids}> : (tensor<1x1x4096x16384xf32>, tensor<1x1x4096x16384xf32>) -> tensor<1x1x4096x16384xf32>
    return %1 : tensor<1x1x4096x16384xf32>
  }
}
// CHECK-NOT: = "ttnn.reshape"
// CHECK: "ttnn.reduce_scatter"
// CHECK: "ttnn.all_gather"
// CHECK-NOT: = "ttnn.reshape"
