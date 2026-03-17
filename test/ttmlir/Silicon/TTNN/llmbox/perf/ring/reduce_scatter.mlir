// RUN: ttmlir-opt --ttir-to-ttnn-backend-pipeline="system-desc-path=%system_desc_path% mesh-shape=2,4 mesh-topology=ring,ring" -o %t.mlir %s
// RUN: FileCheck %s --input-file=%t.mlir
// RUN: ttmlir-translate --ttnn-to-flatbuffer -o %t.ttnn %t.mlir

// CHECK-LABEL: @reduce_scatter_cluster_axis0
func.func @reduce_scatter_cluster_axis0(%arg0: tensor<1x1x8192x512xf32>) -> tensor<1x1x8192x256xf32> {
  %1 = "ttir.mesh_shard"(%arg0) <{shard_dims = array<i64: 2, 3>, shard_direction = #ttcore.shard_direction<full_to_shard>, shard_shape = array<i64: 1, 1, 2, 4>, shard_type = #ttcore.shard_type<devices>}> : (tensor<1x1x8192x512xf32>) -> tensor<1x1x4096x128xf32>
  // CHECK: "ttnn.distribute_tensor"
  %3 = "ttir.reduce_scatter"(%1) <{cluster_axis = 0 : ui32, reduce_type = #ttcore.reduce_type<sum>, scatter_dim = 3 : si32}> : (tensor<1x1x4096x128xf32>) -> tensor<1x1x4096x64xf32>
  // CHECK: "ttnn.reduce_scatter"
  // CHECK-SAME: topology = #ttcore.topology<ring>
  %5 = "ttir.mesh_shard"(%3) <{shard_dims = array<i64: 2, 3>, shard_direction = #ttcore.shard_direction<shard_to_full>, shard_shape = array<i64: 1, 1, 2, 4>, shard_type = #ttcore.shard_type<devices>}> : (tensor<1x1x4096x64xf32>) -> tensor<1x1x8192x256xf32>
  // CHECK: "ttnn.aggregate_tensor"
  return %5 : tensor<1x1x8192x256xf32>
}

// CHECK-LABEL: @reduce_scatter_cluster_axis1
func.func @reduce_scatter_cluster_axis1(%arg0: tensor<1x1x8192x512xf32>) -> tensor<1x1x8192x128xf32> {
  %1 = "ttir.mesh_shard"(%arg0) <{shard_dims = array<i64: 2, 3>, shard_direction = #ttcore.shard_direction<full_to_shard>, shard_shape = array<i64: 1, 1, 2, 4>, shard_type = #ttcore.shard_type<devices>}> : (tensor<1x1x8192x512xf32>) -> tensor<1x1x4096x128xf32>
  // CHECK: "ttnn.distribute_tensor"
  %3 = "ttir.reduce_scatter"(%1) <{cluster_axis = 1 : ui32, reduce_type = #ttcore.reduce_type<sum>, scatter_dim = 3 : si32}> : (tensor<1x1x4096x128xf32>) -> tensor<1x1x4096x32xf32>
  // CHECK: "ttnn.reduce_scatter"
  // CHECK-SAME: topology = #ttcore.topology<ring>
  %5 = "ttir.mesh_shard"(%3) <{shard_dims = array<i64: 2, 3>, shard_direction = #ttcore.shard_direction<shard_to_full>, shard_shape = array<i64: 1, 1, 2, 4>, shard_type = #ttcore.shard_type<devices>}> : (tensor<1x1x4096x32xf32>) -> tensor<1x1x8192x128xf32>
  // CHECK: "ttnn.aggregate_tensor"
  return %5 : tensor<1x1x8192x128xf32>
}
