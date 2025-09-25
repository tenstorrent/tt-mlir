// RUN: ttmlir-opt --ttir-to-ttnn-backend-pipeline="system-desc-path=%system_desc_path% mesh-shape=1,2" -o %t.mlir %s
// RUN: FileCheck %s --input-file=%t.mlir
// RUN: ttmlir-translate --ttnn-to-flatbuffer -o %t.ttnn %t.mlir

func.func @all_gather_cluster1(%arg0: tensor<1x1x32x128xf32>) -> tensor<1x1x32x128xf32> {
  %1 = "ttir.mesh_shard"(%arg0) <{shard_dims = array<i64: -1, 3>, shard_direction = #ttcore.shard_direction<full_to_shard>, shard_shape = array<i64: 1, 1, 1, 2>, shard_type = #ttcore.shard_type<devices>}> : (tensor<1x1x32x128xf32>) -> tensor<1x1x32x64xf32>
  // CHECK: "ttnn.mesh_shard"
  %2 = ttir.empty() : tensor<1x1x32x128xf32>
  %3 = "ttir.all_gather"(%1, %2) <{all_gather_dim = 3 : si32, cluster_axis = 1 : ui32}> : (tensor<1x1x32x64xf32>, tensor<1x1x32x128xf32>) -> tensor<1x1x32x128xf32>
  // CHECK: "ttnn.all_gather"
  %5 = "ttir.mesh_shard"(%3) <{shard_dims = array<i64: -1>, shard_direction = #ttcore.shard_direction<shard_to_full>, shard_shape = array<i64: 1>, shard_type = #ttcore.shard_type<replicate>}> : (tensor<1x1x32x128xf32>) -> tensor<1x1x32x128xf32>
  // CHECK: "ttnn.mesh_shard"
  return %5 : tensor<1x1x32x128xf32>
}

func.func @all_reduce_cluster1(%arg0: tensor<1x1x256x512xf32>) -> tensor<1x1x256x256xf32> {
  %1 = "ttir.mesh_shard"(%arg0) <{shard_dims = array<i64: -1, 3>, shard_direction = #ttcore.shard_direction<full_to_shard>, shard_shape = array<i64: 1, 1, 1, 2>, shard_type = #ttcore.shard_type<devices>}> : (tensor<1x1x256x512xf32>) -> tensor<1x1x256x256xf32>
  // CHECK: "ttnn.mesh_shard"
  %2 = ttir.empty() : tensor<1x1x256x256xf32>
  %3 = "ttir.all_reduce"(%1, %2) <{cluster_axis = 1 : ui32, reduce_type = #ttcore.reduce_type<sum>}> : (tensor<1x1x256x256xf32>, tensor<1x1x256x256xf32>) -> tensor<1x1x256x256xf32>
  // CHECK: "ttnn.reduce_scatter"
  // CHECK: "ttnn.all_gather"
  %5 = "ttir.mesh_shard"(%3) <{shard_dims = array<i64: -1>, shard_direction = #ttcore.shard_direction<shard_to_full>, shard_shape = array<i64: 1>, shard_type = #ttcore.shard_type<replicate>}> : (tensor<1x1x256x256xf32>) -> tensor<1x1x256x256xf32>
  // CHECK: "ttnn.mesh_shard"
  return %5 : tensor<1x1x256x256xf32>
}

func.func @reduce_scatter_cluster1(%arg0: tensor<1x1x8192x512xf32>) -> (tensor<1x1x8192x256xf32> {jax.result_info = ""}) {
  %1 = "ttir.mesh_shard"(%arg0) <{shard_dims = array<i64: -1, 3>, shard_direction = #ttcore.shard_direction<full_to_shard>, shard_shape = array<i64: 1, 1, 1, 2>, shard_type = #ttcore.shard_type<devices>}> : (tensor<1x1x8192x512xf32>) -> tensor<1x1x8192x256xf32>
  // CHECK: "ttnn.mesh_shard"
  %2 = ttir.empty() : tensor<1x1x8192x128xf32>
  %3 = "ttir.reduce_scatter"(%1, %2) <{cluster_axis = 1 : ui32, reduce_type = #ttcore.reduce_type<sum>, scatter_dim = 3 : si32}> : (tensor<1x1x8192x256xf32>, tensor<1x1x8192x128xf32>) -> tensor<1x1x8192x128xf32>
  // CHECK: "ttnn.reduce_scatter"
  %5 = "ttir.mesh_shard"(%3) <{shard_dims = array<i64: -1, 3>, shard_direction = #ttcore.shard_direction<shard_to_full>, shard_shape = array<i64: 1, 1, 1, 2>, shard_type = #ttcore.shard_type<devices>}> : (tensor<1x1x8192x128xf32>) -> tensor<1x1x8192x256xf32>
  // CHECK: "ttnn.mesh_shard"
  return %5 : tensor<1x1x8192x256xf32>
}

func.func @collective_permute_cluster_1(%arg0: tensor<1x1x8192x512xf32>) -> (tensor<1x1x8192x512xf32> {jax.result_info = ""}) {
  %1 = "ttir.mesh_shard"(%arg0) <{shard_dims = array<i64: -1, 3>, shard_direction = #ttcore.shard_direction<full_to_shard>, shard_shape = array<i64: 1, 1, 1, 2>, shard_type = #ttcore.shard_type<devices>}> : (tensor<1x1x8192x512xf32>) -> tensor<1x1x8192x256xf32>
  // CHECK: "ttnn.mesh_shard"
  %2 = ttir.empty() : tensor<1x1x8192x256xf32>
  %3 = "ttir.collective_permute"(%1, %2) <{source_target_pairs = dense<[[0, 1], [1, 0]]> : tensor<2x2xi64>}> : (tensor<1x1x8192x256xf32>, tensor<1x1x8192x256xf32>) -> tensor<1x1x8192x256xf32>
  // CHECK: "ttnn.collective_permute"
  %5 = "ttir.mesh_shard"(%3) <{shard_dims = array<i64: -1, 3>, shard_direction = #ttcore.shard_direction<shard_to_full>, shard_shape = array<i64: 1, 1, 1, 2>, shard_type = #ttcore.shard_type<devices>}> : (tensor<1x1x8192x256xf32>) -> tensor<1x1x8192x512xf32>
  // CHECK: "ttnn.mesh_shard"
  return %5 : tensor<1x1x8192x512xf32>
}

func.func public @collective_permute_cluster_1_partial_target_pairs(%arg0: tensor<1x1x8192x512xf32>) -> (tensor<1x1x8192x512xf32> {jax.result_info = ""}) {
  %1 = "ttir.mesh_shard"(%arg0) <{shard_dims = array<i64: -1, 3>, shard_direction = #ttcore.shard_direction<full_to_shard>, shard_shape = array<i64: 1, 1, 1, 2>, shard_type = #ttcore.shard_type<devices>}> : (tensor<1x1x8192x512xf32>) -> tensor<1x1x8192x256xf32>
  // CHECK: "ttnn.mesh_shard"
  %2 = ttir.empty() : tensor<1x1x8192x256xf32>
  %3 = "ttir.collective_permute"(%1, %2) <{source_target_pairs = dense<[[0, 1]]> : tensor<1x2xi64>}> : (tensor<1x1x8192x256xf32>, tensor<1x1x8192x256xf32>) -> tensor<1x1x8192x256xf32>
  // CHECK: "ttnn.collective_permute"
  %5 = "ttir.mesh_shard"(%3) <{shard_dims = array<i64: -1, 3>, shard_direction = #ttcore.shard_direction<shard_to_full>, shard_shape = array<i64: 1, 1, 1, 2>, shard_type = #ttcore.shard_type<devices>}> : (tensor<1x1x8192x256xf32>) -> tensor<1x1x8192x512xf32>
  // CHECK: "ttnn.mesh_shard"
  return %5 : tensor<1x1x8192x512xf32>
}

func.func public @all_to_all_same_dim_cluster_axis_1(%arg0: tensor<1x1x8192x512xf32>) -> (tensor<1x1x8192x512xf32> {jax.result_info = ""}) {
  %1 = "ttir.mesh_shard"(%arg0) <{shard_dims = array<i64: 2, 3>, shard_direction = #ttcore.shard_direction<full_to_shard>, shard_shape = array<i64: 1, 1, 1, 2>, shard_type = #ttcore.shard_type<devices>}> : (tensor<1x1x8192x512xf32>) -> tensor<1x1x8192x256xf32>
  // CHECK: "ttnn.mesh_shard"
  %2 = ttir.empty() : tensor<1x1x8192x256xf32>
  %3 = "ttir.all_to_all"(%1, %2) <{replica_groups = dense<[[0, 1]]> : tensor<1x2xi64>, concat_dim = 2 : si32, split_count = 2 : si32, split_dim = 2 : si32}> : (tensor<1x1x8192x256xf32>, tensor<1x1x8192x256xf32>) -> tensor<1x1x8192x256xf32>
  // CHECK: "ttnn.point_to_point"
  %5 = "ttir.mesh_shard"(%3) <{shard_dims = array<i64: 2, 3>, shard_direction = #ttcore.shard_direction<shard_to_full>, shard_shape = array<i64: 1, 1, 1, 2>, shard_type = #ttcore.shard_type<devices>}> : (tensor<1x1x8192x256xf32>) -> tensor<1x1x8192x512xf32>
  // CHECK: "ttnn.mesh_shard"
  return %5 : tensor<1x1x8192x512xf32>
}

func.func public @all_to_all_different_dim_cluster_axis_1(%arg0: tensor<1x1x8192x512xf32>) -> (tensor<1x1x4096x1024xf32> {jax.result_info = ""}) {
  %1 = "ttir.mesh_shard"(%arg0) <{shard_dims = array<i64: 2, 3>, shard_direction = #ttcore.shard_direction<full_to_shard>, shard_shape = array<i64: 1, 1, 1, 2>, shard_type = #ttcore.shard_type<devices>}> : (tensor<1x1x8192x512xf32>) -> tensor<1x1x8192x256xf32>
  // CHECK: "ttnn.mesh_shard"
  %2 = ttir.empty() : tensor<1x1x4096x512xf32>
  %3 = "ttir.all_to_all"(%1, %2) <{replica_groups = dense<[[0, 1]]> : tensor<1x2xi64>, concat_dim = 3 : si32, split_count = 2 : si32, split_dim = 2 : si32}> : (tensor<1x1x8192x256xf32>, tensor<1x1x4096x512xf32>) -> tensor<1x1x4096x512xf32>
  // CHECK: "ttnn.point_to_point"
  %5 = "ttir.mesh_shard"(%3) <{shard_dims = array<i64: 2, 3>, shard_direction = #ttcore.shard_direction<shard_to_full>, shard_shape = array<i64: 1, 1, 1, 2>, shard_type = #ttcore.shard_type<devices>}> : (tensor<1x1x4096x512xf32>) -> tensor<1x1x4096x1024xf32>
  // CHECK: "ttnn.mesh_shard"
  return %5 : tensor<1x1x4096x1024xf32>
}

func.func public @collective_permute_broadcast_cluster_1(%arg0: tensor<1x1x256x512xf32>) -> (tensor<1x1x256x512xf32> {jax.result_info = ""}) {
  %1 = "ttir.mesh_shard"(%arg0) <{shard_dims = array<i64: -1, 3>, shard_direction = #ttcore.shard_direction<full_to_shard>, shard_shape = array<i64: 1, 1, 1, 2>, shard_type = #ttcore.shard_type<devices>}> : (tensor<1x1x256x512xf32>) -> tensor<1x1x256x256xf32>
  // CHECK: "ttnn.mesh_shard"
  %2 = ttir.empty() : tensor<1x1x256x256xf32>
  %3 = "ttir.collective_broadcast"(%1, %2) <{replica_groups = dense<[[0, 1]]> : tensor<1x2xi64>}> : (tensor<1x1x256x256xf32>, tensor<1x1x256x256xf32>) -> tensor<1x1x256x256xf32>
  // CHECK: "ttnn.point_to_point"
  %5 = "ttir.mesh_shard"(%3) <{shard_dims = array<i64: -1, 3>, shard_direction = #ttcore.shard_direction<shard_to_full>, shard_shape = array<i64: 1, 1, 1, 2>, shard_type = #ttcore.shard_type<devices>}> : (tensor<1x1x256x256xf32>) -> tensor<1x1x256x512xf32>
  // CHECK: "ttnn.mesh_shard"
  return %5 : tensor<1x1x256x512xf32>
}


func.func public @main(%arg0: tensor<8192x784xf32> {mhlo.layout_mode = "default"}, %arg1: tensor<784x16384xf32> {mhlo.layout_mode = "default"}) -> (tensor<8192x16384xf32> {jax.result_info = "", mhlo.layout_mode = "default"}) {
  %1 = "ttir.mesh_shard"(%arg0) <{shard_dims = array<i64: -1, 1>, shard_direction = #ttcore.shard_direction<full_to_shard>, shard_shape = array<i64: 1, 2>, shard_type = #ttcore.shard_type<devices>}> : (tensor<8192x784xf32>) -> tensor<8192x392xf32>
  // CHECK: "ttnn.mesh_shard"
  %3 = "ttir.mesh_shard"(%arg1) <{shard_dims = array<i64: -1, 0>, shard_direction = #ttcore.shard_direction<full_to_shard>, shard_shape = array<i64: 2, 1>,  shard_type = #ttcore.shard_type<devices>}> : (tensor<784x16384xf32>) -> tensor<392x16384xf32>
  // CHECK: "ttnn.mesh_shard"
  %4 = ttir.empty() : tensor<8192x16384xf32>
  %5 = "ttir.matmul"(%1, %3, %4) : (tensor<8192x392xf32>, tensor<392x16384xf32>, tensor<8192x16384xf32>) -> tensor<8192x16384xf32>
  %6 = ttir.empty() : tensor<8192x16384xf32>
  %7 = "ttir.all_reduce"(%5, %6) <{all_gather_dim = 1 : si32, cluster_axis = 1 : ui32, reduce_type = #ttcore.reduce_type<sum>, scatter_dim = 1 : si32}> : (tensor<8192x16384xf32>, tensor<8192x16384xf32>) -> tensor<8192x16384xf32>
  %9 = "ttir.mesh_shard"(%7) <{shard_dims = array<i64: -1>, shard_direction = #ttcore.shard_direction<shard_to_full>, shard_shape = array<i64: 1>, shard_type = #ttcore.shard_type<replicate>}> : (tensor<8192x16384xf32>) -> tensor<8192x16384xf32>
  // CHECK: "ttnn.mesh_shard"
  return %9 : tensor<8192x16384xf32>
}
