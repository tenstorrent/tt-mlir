// RUN: ttmlir-opt --ttir-to-ttnn-backend-pipeline="system-desc-path=%system_desc_path% mesh-shape=1,2" -o %t.mlir %s
// RUN: FileCheck %s --input-file=%t.mlir
// RUN: ttmlir-translate --ttnn-to-flatbuffer -o %t.ttnn %t.mlir

func.func @reduce_scatter_cluster1(%arg0: tensor<1x1x8192x512xf32>) -> (tensor<1x1x8192x256xf32> {jax.result_info = ""}) {
  %0 = ttir.empty() : tensor<1x1x8192x256xf32>
  %1 = "ttir.mesh_shard"(%arg0, %0) <{shard_dims = array<i64: -1, 3>, shard_direction = #ttcore.shard_direction<full_to_shard>, shard_shape = array<i64: 1, 1, 1, 2>, shard_type = #ttcore.shard_type<devices>}> : (tensor<1x1x8192x512xf32>, tensor<1x1x8192x256xf32>) -> tensor<1x1x8192x256xf32>
  // CHECK: "ttnn.mesh_shard"
  %2 = ttir.empty() : tensor<1x1x8192x128xf32>
  %3 = "ttir.reduce_scatter"(%1, %2) <{cluster_axis = 1 : ui32, reduce_type = #ttcore.reduce_type<sum>, scatter_dim = 3 : si32}> : (tensor<1x1x8192x256xf32>, tensor<1x1x8192x128xf32>) -> tensor<1x1x8192x128xf32>
  // CHECK: "ttnn.reduce_scatter"
  %4 = ttir.empty() : tensor<1x1x8192x256xf32>
  %5 = "ttir.mesh_shard"(%3, %4) <{shard_dims = array<i64: -1, 3>, shard_direction = #ttcore.shard_direction<shard_to_full>, shard_shape = array<i64: 1, 1, 1, 2>, shard_type = #ttcore.shard_type<devices>}> : (tensor<1x1x8192x128xf32>, tensor<1x1x8192x256xf32>) -> tensor<1x1x8192x256xf32>
  // CHECK: "ttnn.mesh_shard"
  return %5 : tensor<1x1x8192x256xf32>
}
