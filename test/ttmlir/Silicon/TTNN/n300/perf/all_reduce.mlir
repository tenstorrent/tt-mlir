// RUN: ttmlir-opt --ttir-to-ttnn-backend-pipeline="system-desc-path=%system_desc_path% mesh-shape=1,2" %s > %t.mlir
// RUN: FileCheck %s --input-file=%t.mlir
// RUN: ttmlir-translate --ttnn-to-flatbuffer %t.mlir > %t.ttnn

func.func @forward2(%arg0: tensor<1x1x256x512xf32>) -> tensor<1x1x256x256xf32> {
  %0 = ttir.empty() : tensor<1x1x256x256xf32>
  %1 = "ttir.mesh_shard"(%arg0, %0) <{shard_dims = array<i64: -1, 3>, shard_direction = #tt.shard_direction<full_to_shard>, shard_shape = array<i64: 1, 1, 1, 2>, shard_type = #tt.shard_type<devices>}> : (tensor<1x1x256x512xf32>, tensor<1x1x256x256xf32>) -> tensor<1x1x256x256xf32>
  // CHECK: "ttnn.mesh_shard"
  %2 = ttir.empty() : tensor<1x1x256x256xf32>
  %3 = "ttir.all_reduce"(%1, %2) <{cluster_axis = 1 : ui32, reduce_type = #tt.reduce_type<sum>}> : (tensor<1x1x256x256xf32>, tensor<1x1x256x256xf32>) -> tensor<1x1x256x256xf32>
  // CHECK: "ttnn.reduce_scatter"
  // CHECK: "ttnn.all_gather"
  %4 = ttir.empty() : tensor<1x1x256x256xf32>
  %5 = "ttir.mesh_shard"(%3, %4) <{shard_dims = array<i64: -1>, shard_direction = #tt.shard_direction<shard_to_full>, shard_shape = array<i64: 1>, shard_type = #tt.shard_type<replicate>}> : (tensor<1x1x256x256xf32>, tensor<1x1x256x256xf32>) -> tensor<1x1x256x256xf32>
  // CHECK: "ttnn.mesh_shard"
  return %5 : tensor<1x1x256x256xf32>
}
