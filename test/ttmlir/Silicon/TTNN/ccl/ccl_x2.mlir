// RUN: ttmlir-opt --ttir-to-ttnn-backend-pipeline="system-desc-path=%system_desc_path% mesh-shape=1,2" %s > %t.mlir
// RUN: FileCheck %s --input-file=%t.mlir
// RUN: ttmlir-translate --ttnn-to-flatbuffer %t.mlir > %t.ttnn
// REQUIRES: multi-chip-x2

func.func @forward(%arg0: tensor<1x1x32x128xf32>) -> tensor<1x1x32x128xf32> {
  %0 = tensor.empty() : tensor<1x1x32x64xf32>
  %1 = "ttir.mesh_shard"(%arg0, %0) <{shard_direction = #tt.shard_direction<full_to_shard>, shard_shape = #tt.grid<1x2>, shard_type = #tt.shard_type<devices>}> : (tensor<1x1x32x128xf32>, tensor<1x1x32x64xf32>) -> tensor<1x1x32x64xf32>
  // CHECK: %[[C:.*]] = "ttnn.mesh_shard"[[C:.*]]
  %2 = tensor.empty() : tensor<1x1x32x128xf32>
  %3 = "ttir.all_gather"(%1, %2) <{dim = 3 : si32}> : (tensor<1x1x32x64xf32>, tensor<1x1x32x128xf32>) -> tensor<1x1x32x128xf32>
  // CHECK: %[[C:.*]] = "ttnn.all_gather"[[C:.*]]
  %4 = tensor.empty() : tensor<1x1x32x128xf32>
  %5 = "ttir.mesh_shard"(%3, %4) <{shard_direction = #tt.shard_direction<shard_to_full>, shard_shape = #tt.grid<1>, shard_type = #tt.shard_type<replicate>}> : (tensor<1x1x32x128xf32>, tensor<1x1x32x128xf32>) -> tensor<1x1x32x128xf32>
  // CHECK: %[[C:.*]] = "ttnn.mesh_shard"[[C:.*]]
  return %5 : tensor<1x1x32x128xf32>
}

func.func @forward2(%arg0: tensor<1x1x256x512xf32>) -> tensor<1x1x256x256xf32> {
  %0 = tensor.empty() : tensor<1x1x256x256xf32>
  %1 = "ttir.mesh_shard"(%arg0, %0) <{shard_direction = #tt.shard_direction<full_to_shard>, shard_shape = #tt.grid<1x2>, shard_type = #tt.shard_type<devices>}> : (tensor<1x1x256x512xf32>, tensor<1x1x256x256xf32>) -> tensor<1x1x256x256xf32>
  // CHECK: %[[C:.*]] = "ttnn.mesh_shard"[[C:.*]]
  %2 = tensor.empty() : tensor<1x1x256x256xf32>
  %3 = "ttir.all_reduce"(%1, %2) <{channel_handle = 1 : si32, dim = 3 : si32, reduce_type = #tt.reduce_type<sum>, replica_groups = dense<[[0, 1]]> : tensor<1x2xi64>, use_global_device_ids}> : (tensor<1x1x256x256xf32>, tensor<1x1x256x256xf32>) -> tensor<1x1x256x256xf32>
  // CHECK: %[[C:.*]] = "ttnn.reduce_scatter"[[C:.*]]
  // CHECK: %[[C:.*]] = "ttnn.all_gather"[[C:.*]]
  %4 = tensor.empty() : tensor<1x1x256x256xf32>
  %5 = "ttir.mesh_shard"(%3, %4) <{shard_direction = #tt.shard_direction<shard_to_full>, shard_shape = #tt.grid<1>, shard_type = #tt.shard_type<replicate>}> : (tensor<1x1x256x256xf32>, tensor<1x1x256x256xf32>) -> tensor<1x1x256x256xf32>
  // CHECK: %[[C:.*]] = "ttnn.mesh_shard"[[C:.*]]
  return %5 : tensor<1x1x256x256xf32>
}
