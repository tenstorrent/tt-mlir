// RUN: ttmlir-opt --ttir-to-ttnn-backend-pipeline %s | FileCheck %s
module attributes {} {
  func.func @forward(%arg0: tensor<8192x784xf32>) -> tensor<4096x196xf32> {
    %0 = tensor.empty() : tensor<4096x196xf32>
    %1 = "ttir.mesh_shard"(%arg0, %0) <{shard_direction = #tt.shard_direction<full_to_shard>, shard_shape = #tt.grid<2x4>, shard_type = #tt.shard_type<devices>}> : (tensor<8192x784xf32>, tensor<4096x196xf32>) -> tensor<4096x196xf32>
    return %1 : tensor<4096x196xf32>
  }
}

// CHECK: %[[C:.*]] = "ttnn.get_device"[[C:.*]]
// CHECK-NEXT: %[[C:.*]] = "ttnn.mesh_shard"[[C:.*]]
