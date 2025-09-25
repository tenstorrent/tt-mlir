// RUN: ttmlir-opt --ttir-to-ttnn-backend-pipeline="system-desc-path=%system_desc_path% mesh-shape=2,4" -o %t.mlir %s
// RUN: FileCheck %s --input-file=%t.mlir
// RUN: ttmlir-translate --ttnn-to-flatbuffer -o %t.ttnn %t.mlir

func.func public @all_gather_cluster0(%arg0: tensor<1x1x8192x512xf32>) -> (tensor<1x1x16384x512xf32> {jax.result_info = ""}) {
  %1 = "ttir.mesh_shard"(%arg0) <{shard_dims = array<i64: 2, 3>, shard_direction = #ttcore.shard_direction<full_to_shard>, shard_shape = array<i64: 1, 1, 2, 4>, shard_type = #ttcore.shard_type<devices>}> : (tensor<1x1x8192x512xf32>) -> tensor<1x1x4096x128xf32>
  // CHECK: "ttnn.mesh_shard"
  %2 = ttir.empty() : tensor<1x1x8192x128xf32>
  %3 = "ttir.all_gather"(%1, %2) <{all_gather_dim = 2 : si32, cluster_axis = 0 : ui32}> : (tensor<1x1x4096x128xf32>, tensor<1x1x8192x128xf32>) -> tensor<1x1x8192x128xf32>
  // CHECK: "ttnn.all_gather"
  %5 = "ttir.mesh_shard"(%3) <{shard_dims = array<i64: 2, 3>, shard_direction = #ttcore.shard_direction<shard_to_full>, shard_shape = array<i64: 1, 1, 2, 4>, shard_type = #ttcore.shard_type<devices>}> : (tensor<1x1x8192x128xf32>) -> tensor<1x1x16384x512xf32>
  // CHECK: "ttnn.mesh_shard"
  return %5 : tensor<1x1x16384x512xf32>
}

func.func @all_gather_cluster1(%arg0: tensor<1x1x256x512xf32>) -> tensor<1x1x256x512xf32> {
  %1 = "ttir.mesh_shard"(%arg0) <{shard_dims = array<i64: 2, 3>, shard_direction = #ttcore.shard_direction<full_to_shard>, shard_shape = array<i64: 1, 1, 2, 4>, shard_type = #ttcore.shard_type<devices>}> : (tensor<1x1x256x512xf32>) -> tensor<1x1x128x128xf32>
  // CHECK: "ttnn.mesh_shard"
  %2 = ttir.empty() : tensor<1x1x128x512xf32>
  %3 = "ttir.all_gather"(%1, %2) <{all_gather_dim = 3 : si32, cluster_axis = 1 : ui32}> : (tensor<1x1x128x128xf32>, tensor<1x1x128x512xf32>) -> tensor<1x1x128x512xf32>
  // CHECK: "ttnn.all_gather"
  %5 = "ttir.mesh_shard"(%3) <{shard_dims = array<i64: 2, -1>, shard_direction = #ttcore.shard_direction<shard_to_full>, shard_shape = array<i64: 1, 1, 2, 1>, shard_type = #ttcore.shard_type<devices>}> : (tensor<1x1x128x512xf32>) -> tensor<1x1x256x512xf32>
  // CHECK: "ttnn.mesh_shard"
  return %5 : tensor<1x1x256x512xf32>
}

func.func @all_reduce_cluster0(%arg0: tensor<1x1x8192x512xf32>) -> (tensor<1x1x8192x256xf32> {jax.result_info = ""}) {
  %1 = "ttir.mesh_shard"(%arg0) <{shard_dims = array<i64: 3, 2>, shard_direction = #ttcore.shard_direction<full_to_shard>, shard_shape = array<i64: 1, 1, 4, 2>, shard_type = #ttcore.shard_type<devices>}> : (tensor<1x1x8192x512xf32>) -> tensor<1x1x2048x256xf32>
  // CHECK: "ttnn.mesh_shard"
  %2 = ttir.empty() : tensor<1x1x2048x256xf32>
  %3 = "ttir.all_reduce"(%1, %2) <{cluster_axis = 0 : ui32, reduce_type = #ttcore.reduce_type<sum>}> : (tensor<1x1x2048x256xf32>, tensor<1x1x2048x256xf32>) -> tensor<1x1x2048x256xf32>
  // CHECK: "ttnn.reduce_scatter"
  // CHECK: "ttnn.all_gather"
  %5 = "ttir.mesh_shard"(%3) <{shard_dims = array<i64: -1, 2>, shard_direction = #ttcore.shard_direction<shard_to_full>, shard_shape = array<i64: 1, 1, 4, 1>, shard_type = #ttcore.shard_type<devices>}> : (tensor<1x1x2048x256xf32>) -> tensor<1x1x8192x256xf32>
  // CHECK: "ttnn.mesh_shard"
  return %5 : tensor<1x1x8192x256xf32>
}

func.func @all_reduce_cluster1(%arg0: tensor<1x1x256x512xf32>) -> tensor<1x1x256x128xf32> {
  %1 = "ttir.mesh_shard"(%arg0) <{shard_dims = array<i64: 2, 3>, shard_direction = #ttcore.shard_direction<full_to_shard>, shard_shape = array<i64: 1, 1, 2, 4>, shard_type = #ttcore.shard_type<devices>}> : (tensor<1x1x256x512xf32>) -> tensor<1x1x128x128xf32>
  // CHECK: "ttnn.mesh_shard"
  %2 = ttir.empty() : tensor<1x1x128x128xf32>
  %3 = "ttir.all_reduce"(%1, %2) <{cluster_axis = 1 : ui32, reduce_type = #ttcore.reduce_type<sum>}> : (tensor<1x1x128x128xf32>, tensor<1x1x128x128xf32>) -> tensor<1x1x128x128xf32>
  // CHECK: "ttnn.reduce_scatter"
  // CHECK: "ttnn.all_gather"
  %5 = "ttir.mesh_shard"(%3) <{shard_dims = array<i64: 2, -1>, shard_direction = #ttcore.shard_direction<shard_to_full>, shard_shape = array<i64: 1, 1, 2, 1>, shard_type = #ttcore.shard_type<devices>}> : (tensor<1x1x128x128xf32>) -> tensor<1x1x256x128xf32>
  // CHECK: "ttnn.mesh_shard"
  return %5 : tensor<1x1x256x128xf32>
}

func.func @reduce_scatter_cluster0(%arg0: tensor<1x1x8192x512xf32>) -> (tensor<1x1x8192x256xf32> {jax.result_info = ""}) {
  %1 = "ttir.mesh_shard"(%arg0) <{shard_dims = array<i64: 3, 2>, shard_direction = #ttcore.shard_direction<full_to_shard>, shard_shape = array<i64: 1, 1, 4, 2>, shard_type = #ttcore.shard_type<devices>}> : (tensor<1x1x8192x512xf32>) -> tensor<1x1x2048x256xf32>
  // CHECK: "ttnn.mesh_shard"
  %2 = ttir.empty() : tensor<1x1x2048x128xf32>
  %3 = "ttir.reduce_scatter"(%1, %2) <{cluster_axis = 0 : ui32, reduce_type = #ttcore.reduce_type<sum>, scatter_dim = 3 : si32}> : (tensor<1x1x2048x256xf32>, tensor<1x1x2048x128xf32>) -> tensor<1x1x2048x128xf32>
  // CHECK: "ttnn.reduce_scatter"
  %5 = "ttir.mesh_shard"(%3) <{shard_dims = array<i64: 3, 2>, shard_direction = #ttcore.shard_direction<shard_to_full>, shard_shape = array<i64: 1, 1, 4, 2>, shard_type = #ttcore.shard_type<devices>}> : (tensor<1x1x2048x128xf32>) -> tensor<1x1x8192x256xf32>
  // CHECK: "ttnn.mesh_shard"
  return %5 : tensor<1x1x8192x256xf32>
}

func.func @reduce_scatter_cluster1(%arg0: tensor<1x1x8192x512xf32>) -> (tensor<1x1x8192x128xf32> {jax.result_info = ""}) {
  %1 = "ttir.mesh_shard"(%arg0) <{shard_dims = array<i64: 2, 3>, shard_direction = #ttcore.shard_direction<full_to_shard>, shard_shape = array<i64: 1, 1, 2, 4>, shard_type = #ttcore.shard_type<devices>}> : (tensor<1x1x8192x512xf32>) -> tensor<1x1x4096x128xf32>
  // CHECK: "ttnn.mesh_shard"
  %2 = ttir.empty() : tensor<1x1x4096x32xf32>
  %3 = "ttir.reduce_scatter"(%1, %2) <{cluster_axis = 1 : ui32, reduce_type = #ttcore.reduce_type<sum>, scatter_dim = 3 : si32}> : (tensor<1x1x4096x128xf32>, tensor<1x1x4096x32xf32>) -> tensor<1x1x4096x32xf32>
  // CHECK: "ttnn.reduce_scatter"
  %5 = "ttir.mesh_shard"(%3) <{shard_dims = array<i64: 2, 3>, shard_direction = #ttcore.shard_direction<shard_to_full>, shard_shape = array<i64: 1, 1, 2, 4>, shard_type = #ttcore.shard_type<devices>}> : (tensor<1x1x4096x32xf32>) -> tensor<1x1x8192x128xf32>
  // CHECK: "ttnn.mesh_shard"
  return %5 : tensor<1x1x8192x128xf32>
}

func.func public @collective_permute_cluster_1(%arg0: tensor<1x1x8192x512xf32>) -> (tensor<1x1x8192x512xf32> {jax.result_info = ""}) {
  %1 = "ttir.mesh_shard"(%arg0) <{shard_dims = array<i64: 2, 3>, shard_direction = #ttcore.shard_direction<full_to_shard>, shard_shape = array<i64: 1, 1, 2, 4>, shard_type = #ttcore.shard_type<devices>}> : (tensor<1x1x8192x512xf32>) -> tensor<1x1x4096x128xf32>
  // CHECK: "ttnn.mesh_shard"
  %2 = ttir.empty() : tensor<1x1x4096x128xf32>
  %3 = "ttir.collective_permute"(%1, %2) <{source_target_pairs = dense<[[0, 1], [1, 2], [2, 3], [3, 0], [4, 5], [5, 6], [6, 7], [7, 4]]> : tensor<8x2xi64>}> : (tensor<1x1x4096x128xf32>, tensor<1x1x4096x128xf32>) -> tensor<1x1x4096x128xf32>
  // CHECK: "ttnn.collective_permute"
  %5 = "ttir.mesh_shard"(%3) <{shard_dims = array<i64: 2, 3>, shard_direction = #ttcore.shard_direction<shard_to_full>, shard_shape = array<i64: 1, 1, 2, 4>, shard_type = #ttcore.shard_type<devices>}> : (tensor<1x1x4096x128xf32>) -> tensor<1x1x8192x512xf32>
  // CHECK: "ttnn.mesh_shard"
  return %5 : tensor<1x1x8192x512xf32>
}

func.func public @collective_permute_cluster_1_partial_target_pairs(%arg0: tensor<1x1x8192x512xf32>) -> (tensor<1x1x8192x512xf32> {jax.result_info = ""}) {
  %1 = "ttir.mesh_shard"(%arg0) <{shard_dims = array<i64: 2, 3>, shard_direction = #ttcore.shard_direction<full_to_shard>, shard_shape = array<i64: 1, 1, 2, 4>, shard_type = #ttcore.shard_type<devices>}> : (tensor<1x1x8192x512xf32>) -> tensor<1x1x4096x128xf32>
  // CHECK: "ttnn.mesh_shard"
  %2 = ttir.empty() : tensor<1x1x4096x128xf32>
  %3 = "ttir.collective_permute"(%1, %2) <{source_target_pairs = dense<[[0, 1], [2, 3], [4, 5], [6, 7]]> : tensor<4x2xi64>}> : (tensor<1x1x4096x128xf32>, tensor<1x1x4096x128xf32>) -> tensor<1x1x4096x128xf32>
  // CHECK: "ttnn.collective_permute"
  %5 = "ttir.mesh_shard"(%3) <{shard_dims = array<i64: 2, 3>, shard_direction = #ttcore.shard_direction<shard_to_full>, shard_shape = array<i64: 1, 1, 2, 4>, shard_type = #ttcore.shard_type<devices>}> : (tensor<1x1x4096x128xf32>) -> tensor<1x1x8192x512xf32>
  // CHECK: "ttnn.mesh_shard"
  return %5 : tensor<1x1x8192x512xf32>
}

func.func public @collective_permute_cluster_0(%arg0: tensor<1x1x8192x512xf32>) -> (tensor<1x1x8192x512xf32> {jax.result_info = ""}) {
  %1 = "ttir.mesh_shard"(%arg0) <{shard_dims = array<i64: 3, 2>, shard_direction = #ttcore.shard_direction<full_to_shard>, shard_shape = array<i64: 1, 1, 4, 2>, shard_type = #ttcore.shard_type<devices>}> : (tensor<1x1x8192x512xf32>) -> tensor<1x1x2048x256xf32>
  // CHECK: "ttnn.mesh_shard"
  %2 = ttir.empty() : tensor<1x1x2048x256xf32>
  %3 = "ttir.collective_permute"(%1, %2) <{source_target_pairs = dense<[[0, 4], [4, 0], [1, 5], [5, 1], [2, 6], [6, 2], [3, 7], [7, 3]]> : tensor<8x2xi64>}> : (tensor<1x1x2048x256xf32>, tensor<1x1x2048x256xf32>) -> tensor<1x1x2048x256xf32>
  // CHECK: "ttnn.collective_permute"
  %5 = "ttir.mesh_shard"(%3) <{shard_dims = array<i64: 3, 2>, shard_direction = #ttcore.shard_direction<shard_to_full>, shard_shape = array<i64: 1, 1, 4, 2>, shard_type = #ttcore.shard_type<devices>}> : (tensor<1x1x2048x256xf32>) -> tensor<1x1x8192x512xf32>
  // CHECK: "ttnn.mesh_shard"
  return %5 : tensor<1x1x8192x512xf32>
}

func.func public @collective_permute_cluster_0_partial_target_pairs(%arg0: tensor<1x1x8192x512xf32>) -> (tensor<1x1x8192x512xf32> {jax.result_info = ""}) {
  %1 = "ttir.mesh_shard"(%arg0) <{shard_dims = array<i64: 3, 2>, shard_direction = #ttcore.shard_direction<full_to_shard>, shard_shape = array<i64: 1, 1, 4, 2>, shard_type = #ttcore.shard_type<devices>}> : (tensor<1x1x8192x512xf32>) -> tensor<1x1x2048x256xf32>
  // CHECK: "ttnn.mesh_shard"
  %2 = ttir.empty() : tensor<1x1x2048x256xf32>
  %3 = "ttir.collective_permute"(%1, %2) <{source_target_pairs = dense<[[0, 4], [1, 5], [2, 6], [3, 7]]> : tensor<4x2xi64>}> : (tensor<1x1x2048x256xf32>, tensor<1x1x2048x256xf32>) -> tensor<1x1x2048x256xf32>
  // CHECK: "ttnn.collective_permute"
  %5 = "ttir.mesh_shard"(%3) <{shard_dims = array<i64: 3, 2>, shard_direction = #ttcore.shard_direction<shard_to_full>, shard_shape = array<i64: 1, 1, 4, 2>, shard_type = #ttcore.shard_type<devices>}> : (tensor<1x1x2048x256xf32>) -> tensor<1x1x8192x512xf32>
  // CHECK: "ttnn.mesh_shard"
  return %5 : tensor<1x1x8192x512xf32>
}

func.func public @all_to_all_same_dim_cluster_0(%arg0: tensor<1x1x8192x512xf32>) -> (tensor<1x1x8192x512xf32> {jax.result_info = ""}) {
  %1 = "ttir.mesh_shard"(%arg0) <{shard_dims = array<i64: 2, 3>, shard_direction = #ttcore.shard_direction<full_to_shard>, shard_shape = array<i64: 1, 1, 2, 4>, shard_type = #ttcore.shard_type<devices>}> : (tensor<1x1x8192x512xf32>) -> tensor<1x1x4096x128xf32>
  // CHECK: "ttnn.mesh_shard"
  %2 = ttir.empty() : tensor<1x1x4096x128xf32>
  %3 = "ttir.all_to_all"(%1, %2) <{replica_groups = dense<[[0, 1], [2, 3], [4, 5], [6, 7]]> : tensor<4x2xi64>, concat_dim = 2 : si32, split_count = 2 : si32, split_dim = 2 : si32}> : (tensor<1x1x4096x128xf32>, tensor<1x1x4096x128xf32>) -> tensor<1x1x4096x128xf32>
  // CHECK: "ttnn.point_to_point"
  %5 = "ttir.mesh_shard"(%3) <{shard_dims = array<i64: 2, 3>, shard_direction = #ttcore.shard_direction<shard_to_full>, shard_shape = array<i64: 1, 1, 2, 4>, shard_type = #ttcore.shard_type<devices>}> : (tensor<1x1x4096x128xf32>) -> tensor<1x1x8192x512xf32>
  // CHECK: "ttnn.mesh_shard"
  return %5 : tensor<1x1x8192x512xf32>
}

func.func public @all_to_all_same_dim_cluster_1(%arg0: tensor<1x1x8192x512xf32>) -> (tensor<1x1x8192x512xf32> {jax.result_info = ""}) {
  %1 = "ttir.mesh_shard"(%arg0) <{shard_dims = array<i64: 2, 3>, shard_direction = #ttcore.shard_direction<full_to_shard>, shard_shape = array<i64: 1, 1, 2, 4>, shard_type = #ttcore.shard_type<devices>}> : (tensor<1x1x8192x512xf32>) -> tensor<1x1x4096x128xf32>
  // CHECK: "ttnn.mesh_shard"
  %2 = ttir.empty() : tensor<1x1x4096x128xf32>
  %3 = "ttir.all_to_all"(%1, %2) <{replica_groups = dense<[[0, 1, 2, 3], [4, 5, 6, 7]]> : tensor<2x4xi64>, concat_dim = 2 : si32, split_count = 4 : si32, split_dim = 2 : si32}> : (tensor<1x1x4096x128xf32>, tensor<1x1x4096x128xf32>) -> tensor<1x1x4096x128xf32>
  // CHECK: "ttnn.point_to_point"
  %5 = "ttir.mesh_shard"(%3) <{shard_dims = array<i64: 2, 3>, shard_direction = #ttcore.shard_direction<shard_to_full>, shard_shape = array<i64: 1, 1, 2, 4>, shard_type = #ttcore.shard_type<devices>}> : (tensor<1x1x4096x128xf32>) -> tensor<1x1x8192x512xf32>
  // CHECK: "ttnn.mesh_shard"
  return %5 : tensor<1x1x8192x512xf32>
}

func.func public @all_to_all_different_dim_cluster_0(%arg0: tensor<1x1x8192x512xf32>) -> (tensor<1x1x4096x1024xf32> {jax.result_info = ""}) {
  %1 = "ttir.mesh_shard"(%arg0) <{shard_dims = array<i64: 2, 3>, shard_direction = #ttcore.shard_direction<full_to_shard>, shard_shape = array<i64: 1, 1, 2, 4>, shard_type = #ttcore.shard_type<devices>}> : (tensor<1x1x8192x512xf32>) -> tensor<1x1x4096x128xf32>
  // CHECK: "ttnn.mesh_shard"
  %2 = ttir.empty() : tensor<1x1x2048x256xf32>
  %3 = "ttir.all_to_all"(%1, %2) <{replica_groups = dense<[[0, 1], [2, 3], [4, 5], [6, 7]]> : tensor<4x2xi64>, concat_dim = 3 : si32, split_count = 2 : si32, split_dim = 2 : si32}> : (tensor<1x1x4096x128xf32>, tensor<1x1x2048x256xf32>) -> tensor<1x1x2048x256xf32>
  // CHECK: "ttnn.point_to_point"
  %5 = "ttir.mesh_shard"(%3) <{shard_dims = array<i64: 2, 3>, shard_direction = #ttcore.shard_direction<shard_to_full>, shard_shape = array<i64: 1, 1, 2, 4>, shard_type = #ttcore.shard_type<devices>}> : (tensor<1x1x2048x256xf32>) -> tensor<1x1x4096x1024xf32>
  // CHECK: "ttnn.mesh_shard"
  return %5 : tensor<1x1x4096x1024xf32>
}

func.func public @all_to_all_different_dim_cluster_1(%arg0: tensor<1x1x8192x512xf32>) -> (tensor<1x1x2048x2048xf32> {jax.result_info = ""}) {
  %1 = "ttir.mesh_shard"(%arg0) <{shard_dims = array<i64: 2, 3>, shard_direction = #ttcore.shard_direction<full_to_shard>, shard_shape = array<i64: 1, 1, 2, 4>, shard_type = #ttcore.shard_type<devices>}> : (tensor<1x1x8192x512xf32>) -> tensor<1x1x4096x128xf32>
  // CHECK: "ttnn.mesh_shard"
  %2 = ttir.empty() : tensor<1x1x1024x512xf32>
  %3 = "ttir.all_to_all"(%1, %2) <{replica_groups = dense<[[0, 1, 2, 3], [4, 5, 6, 7]]> : tensor<2x4xi64>, concat_dim = 3 : si32, split_count = 4 : si32, split_dim = 2 : si32}> : (tensor<1x1x4096x128xf32>, tensor<1x1x1024x512xf32>) -> tensor<1x1x1024x512xf32>
  // CHECK: "ttnn.point_to_point"
  %5 = "ttir.mesh_shard"(%3) <{shard_dims = array<i64: 2, 3>, shard_direction = #ttcore.shard_direction<shard_to_full>, shard_shape = array<i64: 1, 1, 2, 4>, shard_type = #ttcore.shard_type<devices>}> : (tensor<1x1x1024x512xf32>) -> tensor<1x1x2048x2048xf32>
  // CHECK: "ttnn.mesh_shard"
  return %5 : tensor<1x1x2048x2048xf32>
}

func.func public @collective_permute_broadcast_cluster_1(%arg0: tensor<1x1x256x512xf32>) -> (tensor<1x1x256x512xf32> {jax.result_info = ""}) {
  %1 = "ttir.mesh_shard"(%arg0) <{shard_dims = array<i64: 2, 3>, shard_direction = #ttcore.shard_direction<full_to_shard>, shard_shape = array<i64: 1, 1, 2, 4>, shard_type = #ttcore.shard_type<devices>}> : (tensor<1x1x256x512xf32>) -> tensor<1x1x128x64xf32>
  // CHECK: "ttnn.mesh_shard"
  %2 = ttir.empty() : tensor<1x1x128x64xf32>
  %3 = "ttir.collective_broadcast"(%1, %2) <{replica_groups = dense<[[0, 1, 2, 3], [4, 5, 6, 7]]> : tensor<2x4xi64>}> : (tensor<1x1x128x64xf32>, tensor<1x1x128x64xf32>) -> tensor<1x1x128x64xf32>
  // CHECK: "ttnn.point_to_point"
  %5 = "ttir.mesh_shard"(%3) <{shard_dims = array<i64: 2, 3>, shard_direction = #ttcore.shard_direction<shard_to_full>, shard_shape = array<i64: 1, 1, 2, 4>, shard_type = #ttcore.shard_type<devices>}> : (tensor<1x1x128x64xf32>) -> tensor<1x1x256x512xf32>
  // CHECK: "ttnn.mesh_shard"
  return %5 : tensor<1x1x256x512xf32>
}

func.func public @collective_permute_broadcast_cluster_0(%arg0: tensor<1x1x256x512xf32>) -> (tensor<1x1x256x512xf32> {jax.result_info = ""}) {
  %1 = "ttir.mesh_shard"(%arg0) <{shard_dims = array<i64: 2, 3>, shard_direction = #ttcore.shard_direction<full_to_shard>, shard_shape = array<i64: 1, 1, 2, 4>, shard_type = #ttcore.shard_type<devices>}> : (tensor<1x1x256x512xf32>) -> tensor<1x1x128x64xf32>
  // CHECK: "ttnn.mesh_shard"
  %2 = ttir.empty() : tensor<1x1x128x64xf32>
  %3 = "ttir.collective_broadcast"(%1, %2) <{replica_groups = dense<[[0, 4], [1, 5], [2, 6], [3, 7]]> : tensor<4x2xi64>}> : (tensor<1x1x128x64xf32>, tensor<1x1x128x64xf32>) -> tensor<1x1x128x64xf32>
  // CHECK: "ttnn.point_to_point"
  %5 = "ttir.mesh_shard"(%3) <{shard_dims = array<i64: 2, 3>, shard_direction = #ttcore.shard_direction<shard_to_full>, shard_shape = array<i64: 1, 1, 2, 4>, shard_type = #ttcore.shard_type<devices>}> : (tensor<1x1x128x64xf32>) -> tensor<1x1x256x512xf32>
  // CHECK: "ttnn.mesh_shard"
  return %5 : tensor<1x1x256x512xf32>
}
