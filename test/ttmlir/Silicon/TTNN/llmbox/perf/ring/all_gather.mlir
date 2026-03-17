// RUN: ttmlir-opt --ttir-to-ttnn-backend-pipeline="system-desc-path=%system_desc_path% mesh-shape=2,4 mesh-topology=ring,ring" -o %t.mlir %s
// RUN: FileCheck %s --input-file=%t.mlir
// RUN: ttmlir-translate --ttnn-to-flatbuffer -o %t.ttnn %t.mlir

// CHECK-LABEL: @all_gather_cluster_axis0
func.func @all_gather_cluster_axis0(%arg0: tensor<1x1x256x512xf32>) -> tensor<1x1x256x512xf32> {
  %1 = "ttir.mesh_shard"(%arg0) <{shard_dims = array<i64: 2, 3>, shard_direction = #ttcore.shard_direction<full_to_shard>, shard_shape = array<i64: 1, 1, 2, 4>, shard_type = #ttcore.shard_type<devices>}> : (tensor<1x1x256x512xf32>) -> tensor<1x1x128x128xf32>
  // CHECK: "ttnn.distribute_tensor"
  %3 = "ttir.all_gather"(%1) <{all_gather_dim = 2 : si32, cluster_axis = 0 : ui32}> : (tensor<1x1x128x128xf32>) -> tensor<1x1x256x128xf32>
  // CHECK: "ttnn.all_gather"
  // CHECK-SAME: topology = #ttcore.topology<ring>
  %5 = "ttir.mesh_shard"(%3) <{shard_dims = array<i64: -1, 3>, shard_direction = #ttcore.shard_direction<shard_to_full>, shard_shape = array<i64: 1, 1, 1, 4>, shard_type = #ttcore.shard_type<devices>}> : (tensor<1x1x256x128xf32>) -> tensor<1x1x256x512xf32>
  // CHECK: "ttnn.aggregate_tensor"
  return %5 : tensor<1x1x256x512xf32>
}

// CHECK-LABEL: @all_gather_cluster_axis1
func.func @all_gather_cluster_axis1(%arg0: tensor<1x1x256x512xf32>) -> tensor<1x1x256x512xf32> {
  %1 = "ttir.mesh_shard"(%arg0) <{shard_dims = array<i64: 2, 3>, shard_direction = #ttcore.shard_direction<full_to_shard>, shard_shape = array<i64: 1, 1, 2, 4>, shard_type = #ttcore.shard_type<devices>}> : (tensor<1x1x256x512xf32>) -> tensor<1x1x128x128xf32>
  // CHECK: "ttnn.distribute_tensor"
  %3 = "ttir.all_gather"(%1) <{all_gather_dim = 3 : si32, cluster_axis = 1 : ui32}> : (tensor<1x1x128x128xf32>) -> tensor<1x1x128x512xf32>
  // CHECK: "ttnn.all_gather"
  // CHECK-SAME: topology = #ttcore.topology<ring>
  %5 = "ttir.mesh_shard"(%3) <{shard_dims = array<i64: 2, -1>, shard_direction = #ttcore.shard_direction<shard_to_full>, shard_shape = array<i64: 1, 1, 2, 1>, shard_type = #ttcore.shard_type<devices>}> : (tensor<1x1x128x512xf32>) -> tensor<1x1x256x512xf32>
  // CHECK: "ttnn.aggregate_tensor"
  return %5 : tensor<1x1x256x512xf32>
}
