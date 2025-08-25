// RUN: ttmlir-opt --ttir-to-ttnn-backend-pipeline="system-desc-path=%system_desc_path% mesh-shape=1,32" -o %t.mlir %s
// RUN: FileCheck %s --input-file=%t.mlir
// RUN: ttmlir-translate --ttnn-to-flatbuffer -o %t.ttnn %t.mlir

func.func public @all_gather_cluster1(%arg0: tensor<1x1x8192x2048xf32>) -> (tensor<1x1x8192x65536xf32> {jax.result_info = ""}) {
  %1 = "ttir.mesh_shard"(%arg0) <{shard_dims = array<i64: -1, 3>, shard_direction = #ttcore.shard_direction<full_to_shard>, shard_shape = array<i64: 1, 1, 1, 32>, shard_type = #ttcore.shard_type<devices>}> : (tensor<1x1x8192x2048xf32>) -> tensor<1x1x8192x64xf32>
  %2 = ttir.empty() : tensor<1x1x8192x2048xf32>
  %3 = "ttir.all_gather"(%1, %2) <{all_gather_dim = 3 : si32, cluster_axis = 1 : ui32}> : (tensor<1x1x8192x64xf32>, tensor<1x1x8192x2048xf32>) -> tensor<1x1x8192x2048xf32>
  %5 = "ttir.mesh_shard"(%3) <{shard_dims = array<i64: -1, 3>, shard_direction = #ttcore.shard_direction<shard_to_full>, shard_shape = array<i64: 1, 1, 1, 32>, shard_type = #ttcore.shard_type<devices>}> : (tensor<1x1x8192x2048xf32>) -> tensor<1x1x8192x65536xf32>
  return %5 : tensor<1x1x8192x65536xf32>
}

func.func @all_reduce_cluster1(%arg0: tensor<1x1x256x512xf32>) -> tensor<1x1x256x16xf32> {
  %1 = "ttir.mesh_shard"(%arg0) <{shard_dims = array<i64: -1, 3>, shard_direction = #ttcore.shard_direction<full_to_shard>, shard_shape = array<i64: 1, 1, 1, 32>, shard_type = #ttcore.shard_type<devices>}> : (tensor<1x1x256x512xf32>) -> tensor<1x1x256x16xf32>
  // CHECK: "ttnn.mesh_shard"
  %2 = ttir.empty() : tensor<1x1x256x16xf32>
  %3 = "ttir.all_reduce"(%1, %2) <{cluster_axis = 1 : ui32, reduce_type = #ttcore.reduce_type<sum>}> : (tensor<1x1x256x16xf32>, tensor<1x1x256x16xf32>) -> tensor<1x1x256x16xf32>
  // CHECK: "ttnn.reduce_scatter"
  // CHECK: "ttnn.all_gather"
  %5 = "ttir.mesh_shard"(%3) <{shard_dims = array<i64: -1>, shard_direction = #ttcore.shard_direction<shard_to_full>, shard_shape = array<i64: 1>, shard_type = #ttcore.shard_type<replicate>}> : (tensor<1x1x256x16xf32>) -> tensor<1x1x256x16xf32>
  // CHECK: "ttnn.mesh_shard"
  return %5 : tensor<1x1x256x16xf32>
}

func.func @reduce_scatter_cluster1(%arg0: tensor<1x1x512x8192xf32>) -> (tensor<1x1x512x256xf32> {jax.result_info = ""}) {
  %1 = "ttir.mesh_shard"(%arg0) <{shard_dims = array<i64: -1, 3>, shard_direction = #ttcore.shard_direction<full_to_shard>, shard_shape = array<i64: 1, 1, 1, 32>, shard_type = #ttcore.shard_type<devices>}> : (tensor<1x1x512x8192xf32>) -> tensor<1x1x512x256xf32>
  // CHECK: "ttnn.mesh_shard"
  %2 = ttir.empty() : tensor<1x1x512x8xf32>
  %3 = "ttir.reduce_scatter"(%1, %2) <{cluster_axis = 1 : ui32, reduce_type = #ttcore.reduce_type<sum>, scatter_dim = 3 : si32}> : (tensor<1x1x512x256xf32>, tensor<1x1x512x8xf32>) -> tensor<1x1x512x8xf32>
  // CHECK: "ttnn.reduce_scatter"
  %5 = "ttir.mesh_shard"(%3) <{shard_dims = array<i64: -1, 3>, shard_direction = #ttcore.shard_direction<shard_to_full>, shard_shape = array<i64: 1, 1, 1, 32>, shard_type = #ttcore.shard_type<devices>}> : (tensor<1x1x512x8xf32>) -> tensor<1x1x512x256xf32>
  // CHECK: "ttnn.mesh_shard"
  return %5 : tensor<1x1x512x256xf32>
}

func.func public @collective_permute_cluster_1(%arg0: tensor<1x1x8192x512xf32>) -> (tensor<1x1x8192x512xf32> {jax.result_info = ""}) {
  %1 = "ttir.mesh_shard"(%arg0) <{shard_dims = array<i64: -1, 3>, shard_direction = #ttcore.shard_direction<full_to_shard>, shard_shape = array<i64: 1, 1, 1, 32>, shard_type = #ttcore.shard_type<devices>}> : (tensor<1x1x8192x512xf32>) -> tensor<1x1x8192x16xf32>
  // CHECK: "ttnn.mesh_shard"
  %2 = ttir.empty() : tensor<1x1x8192x16xf32>
  %3 = "ttir.collective_permute"(%1, %2) <{source_target_pairs = dense<[[0, 1], [1, 2], [2, 3], [3, 4], [4, 5], [5, 6], [6, 7], [7, 8], [8, 9], [9, 10], [10, 11], [11, 12], [12, 13], [13, 14], [14, 15], [15, 16], [16, 17], [17, 18], [18, 19], [19, 20], [20, 21], [21, 22], [22, 23], [23, 24], [24, 25], [25, 26], [26, 27], [27, 28], [28, 29], [29, 30], [30, 31], [31, 0]]> : tensor<32x2xi64>}> : (tensor<1x1x8192x16xf32>, tensor<1x1x8192x16xf32>) -> tensor<1x1x8192x16xf32>
  // CHECK: "ttnn.collective_permute"
  %5 = "ttir.mesh_shard"(%3) <{shard_dims = array<i64: -1, 3>, shard_direction = #ttcore.shard_direction<shard_to_full>, shard_shape = array<i64: 1, 1, 1, 32>, shard_type = #ttcore.shard_type<devices>}> : (tensor<1x1x8192x16xf32>) -> tensor<1x1x8192x512xf32>
  // CHECK: "ttnn.mesh_shard"
  return %5 : tensor<1x1x8192x512xf32>
}

func.func public @collective_permute_cluster_1_partial_target_pairs(%arg0: tensor<1x1x8192x512xf32>) -> (tensor<1x1x8192x512xf32> {jax.result_info = ""}) {
  %1 = "ttir.mesh_shard"(%arg0) <{shard_dims = array<i64: -1, 3>, shard_direction = #ttcore.shard_direction<full_to_shard>, shard_shape = array<i64: 1, 1, 1, 32>, shard_type = #ttcore.shard_type<devices>}> : (tensor<1x1x8192x512xf32>) -> tensor<1x1x8192x16xf32>
  // CHECK: "ttnn.mesh_shard"
  %2 = ttir.empty() : tensor<1x1x8192x16xf32>
  %3 = "ttir.collective_permute"(%1, %2) <{source_target_pairs = dense<[[0, 1], [2, 3], [4, 5], [6, 7], [8, 9], [10, 11], [12, 13], [14, 15], [16, 17], [18, 19], [20, 21], [22, 23], [24, 25], [26, 27], [28, 29], [30, 31]]> : tensor<16x2xi64>}> : (tensor<1x1x8192x16xf32>, tensor<1x1x8192x16xf32>) -> tensor<1x1x8192x16xf32>
  // CHECK: "ttnn.collective_permute"
  %5 = "ttir.mesh_shard"(%3) <{shard_dims = array<i64: -1, 3>, shard_direction = #ttcore.shard_direction<shard_to_full>, shard_shape = array<i64: 1, 1, 1, 32>, shard_type = #ttcore.shard_type<devices>}> : (tensor<1x1x8192x16xf32>) -> tensor<1x1x8192x512xf32>
  // CHECK: "ttnn.mesh_shard"
  return %5 : tensor<1x1x8192x512xf32>
}

func.func public @all_to_all_same_dim(%arg0: tensor<1x1x8192x512xf32>) -> (tensor<1x1x8192x512xf32> {jax.result_info = ""}) {
  %1 = "ttir.mesh_shard"(%arg0) <{shard_dims = array<i64: 2, 3>, shard_direction = #ttcore.shard_direction<full_to_shard>, shard_shape = array<i64: 1, 1, 1, 32>, shard_type = #ttcore.shard_type<devices>}> : (tensor<1x1x8192x512xf32>) -> tensor<1x1x8192x16xf32>
  // CHECK: "ttnn.mesh_shard"
  %2 = ttir.empty() : tensor<1x1x8192x16xf32>
  %3 = "ttir.all_to_all"(%1, %2) <{replica_groups = dense<[[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31]]> : tensor<1x32xi64>, concat_dim = 2 : si32, split_count = 32 : si32, split_dim = 2 : si32}> : (tensor<1x1x8192x16xf32>, tensor<1x1x8192x16xf32>) -> tensor<1x1x8192x16xf32>
  // CHECK: "ttnn.point_to_point"
  %5 = "ttir.mesh_shard"(%3) <{shard_dims = array<i64: 2, 3>, shard_direction = #ttcore.shard_direction<shard_to_full>, shard_shape = array<i64: 1, 1, 1, 32>, shard_type = #ttcore.shard_type<devices>}> : (tensor<1x1x8192x16xf32>) -> tensor<1x1x8192x512xf32>
  // CHECK: "ttnn.mesh_shard"
  return %5 : tensor<1x1x8192x512xf32>
}

func.func public @all_to_all_different_dim(%arg0: tensor<1x1x8192x512xf32>) -> (tensor<1x1x256x16384xf32> {jax.result_info = ""}) {
  %1 = "ttir.mesh_shard"(%arg0) <{shard_dims = array<i64: 2, 3>, shard_direction = #ttcore.shard_direction<full_to_shard>, shard_shape = array<i64: 1, 1, 1, 32>, shard_type = #ttcore.shard_type<devices>}> : (tensor<1x1x8192x512xf32>) -> tensor<1x1x8192x16xf32>
  // CHECK: "ttnn.mesh_shard"
  %2 = ttir.empty() : tensor<1x1x256x512xf32>
  %3 = "ttir.all_to_all"(%1, %2) <{replica_groups = dense<[[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31]]> : tensor<1x32xi64>, concat_dim = 3 : si32, split_count = 32 : si32, split_dim = 2 : si32}> : (tensor<1x1x8192x16xf32>, tensor<1x1x256x512xf32>) -> tensor<1x1x256x512xf32>
  // CHECK: "ttnn.point_to_point"
  %5 = "ttir.mesh_shard"(%3) <{shard_dims = array<i64: 2, 3>, shard_direction = #ttcore.shard_direction<shard_to_full>, shard_shape = array<i64: 1, 1, 1, 32>, shard_type = #ttcore.shard_type<devices>}> : (tensor<1x1x256x512xf32>) -> tensor<1x1x256x16384xf32>
  // CHECK: "ttnn.mesh_shard"
  return %5 : tensor<1x1x256x16384xf32>
}

func.func public @collective_permute_broadcast_cluster_1(%arg0: tensor<1x1x256x2048xf32>) -> (tensor<1x1x256x2048xf32> {jax.result_info = ""}) {
  %1 = "ttir.mesh_shard"(%arg0) <{shard_dims = array<i64: -1, 3>, shard_direction = #ttcore.shard_direction<full_to_shard>, shard_shape = array<i64: 1, 1, 1, 32>, shard_type = #ttcore.shard_type<devices>}> : (tensor<1x1x256x2048xf32>) -> tensor<1x1x256x32xf32>
  // CHECK: "ttnn.mesh_shard"
  %2 = ttir.empty() : tensor<1x1x256x32xf32>
  %3 = "ttir.collective_broadcast"(%1, %2) <{replica_groups = dense<[[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31]]> : tensor<1x32xi64>}> : (tensor<1x1x256x32xf32>, tensor<1x1x256x32xf32>) -> tensor<1x1x256x32xf32>
  // CHECK: "ttnn.point_to_point"
  %5 = "ttir.mesh_shard"(%3) <{shard_dims = array<i64: -1, 3>, shard_direction = #ttcore.shard_direction<shard_to_full>, shard_shape = array<i64: 1, 1, 1, 32>, shard_type = #ttcore.shard_type<devices>}> : (tensor<1x1x256x32xf32>) -> tensor<1x1x256x2048xf32>
  // CHECK: "ttnn.mesh_shard"
  return %5 : tensor<1x1x256x2048xf32>
}
