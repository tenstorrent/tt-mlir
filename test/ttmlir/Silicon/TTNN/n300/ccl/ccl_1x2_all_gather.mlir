// RUN: ttmlir-opt --ttir-to-ttnn-backend-pipeline="system-desc-path=%system_desc_path% mesh-shape=1,2" -o %t.mlir %s
// RUN: FileCheck %s --input-file=%t.mlir
// RUN: ttmlir-translate --ttnn-to-flatbuffer -o %t.ttnn %t.mlir

// Shards along with dimension 3 of input tensor
func.func @all_gather_cluster1_gather_dim_3(%arg0: tensor<1x1x32x128xf32>) -> tensor<1x1x32x128xf32> {
  %0 = ttir.empty() : tensor<1x1x32x64xf32>
  %1 = "ttir.mesh_shard"(%arg0, %0) <{shard_dims = array<i64: -1, 3>, shard_direction = #ttcore.shard_direction<full_to_shard>, shard_shape = array<i64: 1, 1, 1, 2>, shard_type = #ttcore.shard_type<devices>}> : (tensor<1x1x32x128xf32>, tensor<1x1x32x64xf32>) -> tensor<1x1x32x64xf32>
  // CHECK: "ttnn.mesh_shard"
  %2 = ttir.empty() : tensor<1x1x32x128xf32>
  %3 = "ttir.all_gather"(%1, %2) <{all_gather_dim = 3 : si32, replica_groups = dense<[[0, 1]]> : tensor<1x2xi64>, cluster_axis = 1 : ui32}> : (tensor<1x1x32x64xf32>, tensor<1x1x32x128xf32>) -> tensor<1x1x32x128xf32>
  // CHECK: "ttnn.all_gather"
  %4 = ttir.empty() : tensor<1x1x32x128xf32>
  %5 = "ttir.mesh_shard"(%3, %4) <{shard_dims = array<i64: -1>, shard_direction = #ttcore.shard_direction<shard_to_full>, shard_shape = array<i64: 1>, shard_type = #ttcore.shard_type<replicate>}> : (tensor<1x1x32x128xf32>, tensor<1x1x32x128xf32>) -> tensor<1x1x32x128xf32>
  // CHECK: "ttnn.mesh_shard"
  return %5 : tensor<1x1x32x128xf32>
}

// Shards along with dimension 3 of input tensor
func.func @all_gather_cluster1_gather_dim_3_bf16(%arg0: tensor<1x1x32x128xbf16>) -> tensor<1x1x32x128xbf16> {
  %0 = ttir.empty() : tensor<1x1x32x64xbf16>
  %1 = "ttir.mesh_shard"(%arg0, %0) <{shard_dims = array<i64: -1, 3>, shard_direction = #ttcore.shard_direction<full_to_shard>, shard_shape = array<i64: 1, 1, 1, 2>, shard_type = #ttcore.shard_type<devices>}> : (tensor<1x1x32x128xbf16>, tensor<1x1x32x64xbf16>) -> tensor<1x1x32x64xbf16>
  // CHECK: "ttnn.mesh_shard"
  %2 = ttir.empty() : tensor<1x1x32x128xbf16>
  %3 = "ttir.all_gather"(%1, %2) <{all_gather_dim = 3 : si32, replica_groups = dense<[[0, 1]]> : tensor<1x2xi64>, cluster_axis = 1 : ui32}> : (tensor<1x1x32x64xbf16>, tensor<1x1x32x128xbf16>) -> tensor<1x1x32x128xbf16>
  // CHECK: "ttnn.all_gather"
  %4 = ttir.empty() : tensor<1x1x32x128xbf16>
  %5 = "ttir.mesh_shard"(%3, %4) <{shard_dims = array<i64: -1>, shard_direction = #ttcore.shard_direction<shard_to_full>, shard_shape = array<i64: 1>, shard_type = #ttcore.shard_type<replicate>}> : (tensor<1x1x32x128xbf16>, tensor<1x1x32x128xbf16>) -> tensor<1x1x32x128xbf16>
  // CHECK: "ttnn.mesh_shard"
  return %5 : tensor<1x1x32x128xbf16>
}

// Shards along with dimension 2 of input tensor
func.func @all_gather_cluster1_gather_dim_2(%arg0: tensor<1x1x128x32xf32>) -> tensor<1x1x128x32xf32> {
  %0 = ttir.empty() : tensor<1x1x64x32xf32>
  %1 = "ttir.mesh_shard"(%arg0, %0) <{shard_dims = array<i64: -1, 2>, shard_direction = #ttcore.shard_direction<full_to_shard>, shard_shape = array<i64: 1, 1, 2, 1>, shard_type = #ttcore.shard_type<devices>}> : (tensor<1x1x128x32xf32>, tensor<1x1x64x32xf32>) -> tensor<1x1x64x32xf32>
  // CHECK: "ttnn.mesh_shard"
  %2 = ttir.empty() : tensor<1x1x128x32xf32>
  %3 = "ttir.all_gather"(%1, %2) <{all_gather_dim = 2 : si32, replica_groups = dense<[[0, 1]]> : tensor<1x2xi64>, cluster_axis = 1 : ui32}> : (tensor<1x1x64x32xf32>, tensor<1x1x128x32xf32>) -> tensor<1x1x128x32xf32>
  // CHECK: "ttnn.all_gather"
  %4 = ttir.empty() : tensor<1x1x128x32xf32>
  %5 = "ttir.mesh_shard"(%3, %4) <{shard_dims = array<i64: -1>, shard_direction = #ttcore.shard_direction<shard_to_full>, shard_shape = array<i64: 1>, shard_type = #ttcore.shard_type<replicate>}> : (tensor<1x1x128x32xf32>, tensor<1x1x128x32xf32>) -> tensor<1x1x128x32xf32>
  // CHECK: "ttnn.mesh_shard"
  return %5 : tensor<1x1x128x32xf32>
}

// Shards along with dimension 2 of input tensor
func.func @all_gather_cluster1_gather_dim_2_bf16(%arg0: tensor<1x1x128x32xbf16>) -> tensor<1x1x128x32xbf16> {
  %0 = ttir.empty() : tensor<1x1x64x32xbf16>
  %1 = "ttir.mesh_shard"(%arg0, %0) <{shard_dims = array<i64: -1, 2>, shard_direction = #ttcore.shard_direction<full_to_shard>, shard_shape = array<i64: 1, 1, 2, 1>, shard_type = #ttcore.shard_type<devices>}> : (tensor<1x1x128x32xbf16>, tensor<1x1x64x32xbf16>) -> tensor<1x1x64x32xbf16>
  // CHECK: "ttnn.mesh_shard"
  %2 = ttir.empty() : tensor<1x1x128x32xbf16>
  %3 = "ttir.all_gather"(%1, %2) <{all_gather_dim = 2 : si32, replica_groups = dense<[[0, 1]]> : tensor<1x2xi64>, cluster_axis = 1 : ui32}> : (tensor<1x1x64x32xbf16>, tensor<1x1x128x32xbf16>) -> tensor<1x1x128x32xbf16>
  // CHECK: "ttnn.all_gather"
  %4 = ttir.empty() : tensor<1x1x128x32xbf16>
  %5 = "ttir.mesh_shard"(%3, %4) <{shard_dims = array<i64: -1>, shard_direction = #ttcore.shard_direction<shard_to_full>, shard_shape = array<i64: 1>, shard_type = #ttcore.shard_type<replicate>}> : (tensor<1x1x128x32xbf16>, tensor<1x1x128x32xbf16>) -> tensor<1x1x128x32xbf16>
  // CHECK: "ttnn.mesh_shard"
  return %5 : tensor<1x1x128x32xbf16>
}

// Shards along with dimension 1 of input tensor
func.func @all_gather_cluster1_gather_dim_1(%arg0: tensor<1x128x32x32xf32>) -> tensor<1x128x32x32xf32> {
  %0 = ttir.empty() : tensor<1x64x32x32xf32>
  %1 = "ttir.mesh_shard"(%arg0, %0) <{shard_dims = array<i64: -1, 1>, shard_direction = #ttcore.shard_direction<full_to_shard>, shard_shape = array<i64: 1, 2, 1, 1>, shard_type = #ttcore.shard_type<devices>}> : (tensor<1x128x32x32xf32>, tensor<1x64x32x32xf32>) -> tensor<1x64x32x32xf32>
  // CHECK: "ttnn.mesh_shard"
  %2 = ttir.empty() : tensor<1x128x32x32xf32>
  %3 = "ttir.all_gather"(%1, %2) <{all_gather_dim = 1 : si32, replica_groups = dense<[[0, 1]]> : tensor<1x2xi64>, cluster_axis = 1 : ui32}> : (tensor<1x64x32x32xf32>, tensor<1x128x32x32xf32>) -> tensor<1x128x32x32xf32>
  // CHECK: "ttnn.all_gather"
  %4 = ttir.empty() : tensor<1x128x32x32xf32>
  %5 = "ttir.mesh_shard"(%3, %4) <{shard_dims = array<i64: -1>, shard_direction = #ttcore.shard_direction<shard_to_full>, shard_shape = array<i64: 1>, shard_type = #ttcore.shard_type<replicate>}> : (tensor<1x128x32x32xf32>, tensor<1x128x32x32xf32>) -> tensor<1x128x32x32xf32>
  // CHECK: "ttnn.mesh_shard"
  return %5 : tensor<1x128x32x32xf32>
}

// Shards along with dimension 1 of input tensor
func.func @all_gather_cluster1_gather_dim_1_bf16(%arg0: tensor<1x128x32x32xbf16>) -> tensor<1x128x32x32xbf16> {
  %0 = ttir.empty() : tensor<1x64x32x32xbf16>
  %1 = "ttir.mesh_shard"(%arg0, %0) <{shard_dims = array<i64: -1, 1>, shard_direction = #ttcore.shard_direction<full_to_shard>, shard_shape = array<i64: 1, 2, 1, 1>, shard_type = #ttcore.shard_type<devices>}> : (tensor<1x128x32x32xbf16>, tensor<1x64x32x32xbf16>) -> tensor<1x64x32x32xbf16>
  // CHECK: "ttnn.mesh_shard"
  %2 = ttir.empty() : tensor<1x128x32x32xbf16>
  %3 = "ttir.all_gather"(%1, %2) <{all_gather_dim = 1 : si32, replica_groups = dense<[[0, 1]]> : tensor<1x2xi64>, cluster_axis = 1 : ui32}> : (tensor<1x64x32x32xbf16>, tensor<1x128x32x32xbf16>) -> tensor<1x128x32x32xbf16>
  // CHECK: "ttnn.all_gather"
  %4 = ttir.empty() : tensor<1x128x32x32xbf16>
  %5 = "ttir.mesh_shard"(%3, %4) <{shard_dims = array<i64: -1>, shard_direction = #ttcore.shard_direction<shard_to_full>, shard_shape = array<i64: 1>, shard_type = #ttcore.shard_type<replicate>}> : (tensor<1x128x32x32xbf16>, tensor<1x128x32x32xbf16>) -> tensor<1x128x32x32xbf16>
  // CHECK: "ttnn.mesh_shard"
  return %5 : tensor<1x128x32x32xbf16>
}

// Shards along with dimension 0 of input tensor
func.func @all_gather_cluster1_gather_dim_0(%arg0: tensor<128x1x32x32xf32>) -> tensor<128x1x32x32xf32> {
  %0 = ttir.empty() : tensor<64x1x32x32xf32>
  %1 = "ttir.mesh_shard"(%arg0, %0) <{shard_dims = array<i64: -1, 0>, shard_direction = #ttcore.shard_direction<full_to_shard>, shard_shape = array<i64: 2, 1, 1, 1>, shard_type = #ttcore.shard_type<devices>}> : (tensor<128x1x32x32xf32>, tensor<64x1x32x32xf32>) -> tensor<64x1x32x32xf32>
  // CHECK: "ttnn.mesh_shard"
  %2 = ttir.empty() : tensor<128x1x32x32xf32>
  %3 = "ttir.all_gather"(%1, %2) <{all_gather_dim = 0 : si32, replica_groups = dense<[[0, 1]]> : tensor<1x2xi64>, cluster_axis = 1 : ui32}> : (tensor<64x1x32x32xf32>, tensor<128x1x32x32xf32>) -> tensor<128x1x32x32xf32>
  // CHECK: "ttnn.all_gather"
  %4 = ttir.empty() : tensor<128x1x32x32xf32>
  %5 = "ttir.mesh_shard"(%3, %4) <{shard_dims = array<i64: -1>, shard_direction = #ttcore.shard_direction<shard_to_full>, shard_shape = array<i64: 1>, shard_type = #ttcore.shard_type<replicate>}> : (tensor<128x1x32x32xf32>, tensor<128x1x32x32xf32>) -> tensor<128x1x32x32xf32>
  // CHECK: "ttnn.mesh_shard"
  return %5 : tensor<128x1x32x32xf32>
}

// Shards along with dimension 0 of input tensor
func.func @all_gather_cluster1_gather_dim_0_bf16(%arg0: tensor<128x1x32x32xbf16>) -> tensor<128x1x32x32xbf16> {
  %0 = ttir.empty() : tensor<64x1x32x32xbf16>
  %1 = "ttir.mesh_shard"(%arg0, %0) <{shard_dims = array<i64: -1, 0>, shard_direction = #ttcore.shard_direction<full_to_shard>, shard_shape = array<i64: 2, 1, 1, 1>, shard_type = #ttcore.shard_type<devices>}> : (tensor<128x1x32x32xbf16>, tensor<64x1x32x32xbf16>) -> tensor<64x1x32x32xbf16>
  // CHECK: "ttnn.mesh_shard"
  %2 = ttir.empty() : tensor<128x1x32x32xbf16>
  %3 = "ttir.all_gather"(%1, %2) <{all_gather_dim = 0 : si32, replica_groups = dense<[[0, 1]]> : tensor<1x2xi64>, cluster_axis = 1 : ui32}> : (tensor<64x1x32x32xbf16>, tensor<128x1x32x32xbf16>) -> tensor<128x1x32x32xbf16>
  // CHECK: "ttnn.all_gather"
  %4 = ttir.empty() : tensor<128x1x32x32xbf16>
  %5 = "ttir.mesh_shard"(%3, %4) <{shard_dims = array<i64: -1>, shard_direction = #ttcore.shard_direction<shard_to_full>, shard_shape = array<i64: 1>, shard_type = #ttcore.shard_type<replicate>}> : (tensor<128x1x32x32xbf16>, tensor<128x1x32x32xbf16>) -> tensor<128x1x32x32xbf16>
  // CHECK: "ttnn.mesh_shard"
  return %5 : tensor<128x1x32x32xbf16>
}

// These tests will soon be migrated to the 'builder' tests, but they are left commented out here for reference.
// // verify rank coverage : 2
// func.func @all_gather_cluster1_set_rank_to_2(%arg0: tensor<32x128xf32>) -> tensor<32x128xf32> {
//   %0 = ttir.empty() : tensor<32x64xf32>
//   %1 = "ttir.mesh_shard"(%arg0, %0) <{shard_dims = array<i64: -1, 1>, shard_direction = #ttcore.shard_direction<full_to_shard>, shard_shape = array<i64: 1, 2>, shard_type = #ttcore.shard_type<devices>}> : (tensor<32x128xf32>, tensor<32x64xf32>) -> tensor<32x64xf32>
//   // C/HECK: "ttnn.mesh_shard"
//   %2 = ttir.empty() : tensor<32x128xf32>
//   %3 = "ttir.all_gather"(%1, %2) <{all_gather_dim = 1 : si32, replica_groups = dense<[[0, 1]]> : tensor<1x2xi64>, cluster_axis = 1 : ui32}> : (tensor<32x64xf32>, tensor<32x128xf32>) -> tensor<32x128xf32>
//   // C/HECK: "ttnn.all_gather"
//   %4 = ttir.empty() : tensor<32x128xf32>
//   %5 = "ttir.mesh_shard"(%3, %4) <{shard_dims = array<i64: -1>, shard_direction = #ttcore.shard_direction<shard_to_full>, shard_shape = array<i64: 1>, shard_type = #ttcore.shard_type<replicate>}> : (tensor<32x128xf32>, tensor<32x128xf32>) -> tensor<32x128xf32>
//   // C/HECK: "ttnn.mesh_shard"
//   return %5 : tensor<32x128xf32>
// }

// // verify rank coverage : 2
// func.func @all_gather_cluster1_set_rank_to_2_bf16(%arg0: tensor<32x128xbf16>) -> tensor<32x128xbf16> {
//   %0 = ttir.empty() : tensor<32x64xbf16>
//   %1 = "ttir.mesh_shard"(%arg0, %0) <{shard_dims = array<i64: -1, 1>, shard_direction = #ttcore.shard_direction<full_to_shard>, shard_shape = array<i64: 1, 2>, shard_type = #ttcore.shard_type<devices>}> : (tensor<32x128xbf16>, tensor<32x64xbf16>) -> tensor<32x64xbf16>
//   // C/HECK: "ttnn.mesh_shard"
//   %2 = ttir.empty() : tensor<32x128xbf16>
//   %3 = "ttir.all_gather"(%1, %2) <{all_gather_dim = 1 : si32, replica_groups = dense<[[0, 1]]> : tensor<1x2xi64>, cluster_axis = 1 : ui32}> : (tensor<32x64xbf16>, tensor<32x128xbf16>) -> tensor<32x128xbf16>
//   // C/HECK: "ttnn.all_gather"
//   %4 = ttir.empty() : tensor<32x128xbf16>
//   %5 = "ttir.mesh_shard"(%3, %4) <{shard_dims = array<i64: -1>, shard_direction = #ttcore.shard_direction<shard_to_full>, shard_shape = array<i64: 1>, shard_type = #ttcore.shard_type<replicate>}> : (tensor<32x128xbf16>, tensor<32x128xbf16>) -> tensor<32x128xbf16>
//   // C/HECK: "ttnn.mesh_shard"
//   return %5 : tensor<32x128xbf16>
// }

// // verify rank coverage : 3
// func.func @all_gather_cluster1_set_rank_to_3(%arg0: tensor<1x32x128xf32>) -> tensor<1x32x128xf32> {
//   %0 = ttir.empty() : tensor<1x32x64xf32>
//   %1 = "ttir.mesh_shard"(%arg0, %0) <{shard_dims = array<i64: -1, 2>, shard_direction = #ttcore.shard_direction<full_to_shard>, shard_shape = array<i64: 1, 1, 2>, shard_type = #ttcore.shard_type<devices>}> : (tensor<1x32x128xf32>, tensor<1x32x64xf32>) -> tensor<1x32x64xf32>
//   // C/HECK: "ttnn.mesh_shard"
//   %2 = ttir.empty() : tensor<1x32x128xf32>
//   %3 = "ttir.all_gather"(%1, %2) <{all_gather_dim = 2 : si32, replica_groups = dense<[[0, 1]]> : tensor<1x2xi64>, cluster_axis = 1 : ui32}> : (tensor<1x32x64xf32>, tensor<1x32x128xf32>) -> tensor<1x32x128xf32>
//   // C/HECK: "ttnn.all_gather"
//   %4 = ttir.empty() : tensor<1x32x128xf32>
//   %5 = "ttir.mesh_shard"(%3, %4) <{shard_dims = array<i64: -1>, shard_direction = #ttcore.shard_direction<shard_to_full>, shard_shape = array<i64: 1>, shard_type = #ttcore.shard_type<replicate>}> : (tensor<1x32x128xf32>, tensor<1x32x128xf32>) -> tensor<1x32x128xf32>
//   // C/HECK: "ttnn.mesh_shard"
//   return %5 : tensor<1x32x128xf32>
// }

// // verify rank coverage : 3
// func.func @all_gather_cluster1_set_rank_to_3_bf16(%arg0: tensor<1x32x128xbf16>) -> tensor<1x32x128xbf16> {
//   %0 = ttir.empty() : tensor<1x32x64xbf16>
//   %1 = "ttir.mesh_shard"(%arg0, %0) <{shard_dims = array<i64: -1, 2>, shard_direction = #ttcore.shard_direction<full_to_shard>, shard_shape = array<i64: 1, 1, 2>, shard_type = #ttcore.shard_type<devices>}> : (tensor<1x32x128xbf16>, tensor<1x32x64xbf16>) -> tensor<1x32x64xbf16>
//   // C/HECK: "ttnn.mesh_shard"
//   %2 = ttir.empty() : tensor<1x32x128xbf16>
//   %3 = "ttir.all_gather"(%1, %2) <{all_gather_dim = 2 : si32, replica_groups = dense<[[0, 1]]> : tensor<1x2xi64>, cluster_axis = 1 : ui32}> : (tensor<1x32x64xbf16>, tensor<1x32x128xbf16>) -> tensor<1x32x128xbf16>
//   // C/HECK: "ttnn.all_gather"
//   %4 = ttir.empty() : tensor<1x32x128xbf16>
//   %5 = "ttir.mesh_shard"(%3, %4) <{shard_dims = array<i64: -1>, shard_direction = #ttcore.shard_direction<shard_to_full>, shard_shape = array<i64: 1>, shard_type = #ttcore.shard_type<replicate>}> : (tensor<1x32x128xbf16>, tensor<1x32x128xbf16>) -> tensor<1x32x128xbf16>
//   // C/HECK: "ttnn.mesh_shard"
//   return %5 : tensor<1x32x128xbf16>
// }

// // verify rank coverage : 5 - increase rank on head
// func.func @all_gather_cluster1_set_rank_to_5_head(%arg0: tensor<1x1x1x32x128xf32>) -> tensor<1x1x1x32x128xf32> {
//   %0 = ttir.empty() : tensor<1x1x1x32x64xf32>
//   %1 = "ttir.mesh_shard"(%arg0, %0) <{shard_dims = array<i64: -1, 4>, shard_direction = #ttcore.shard_direction<full_to_shard>, shard_shape = array<i64: 1, 1, 1, 1, 2>, shard_type = #ttcore.shard_type<devices>}> : (tensor<1x1x1x32x128xf32>, tensor<1x1x1x32x64xf32>) -> tensor<1x1x1x32x64xf32>
//   // C/HECK: "ttnn.mesh_shard"
//   %2 = ttir.empty() : tensor<1x1x1x32x128xf32>
//   %3 = "ttir.all_gather"(%1, %2) <{all_gather_dim = 4 : si32, replica_groups = dense<[[0, 1]]> : tensor<1x2xi64>, cluster_axis = 1 : ui32}> : (tensor<1x1x1x32x64xf32>, tensor<1x1x1x32x128xf32>) -> tensor<1x1x1x32x128xf32>
//   // C/HECK: "ttnn.all_gather"
//   %4 = ttir.empty() : tensor<1x1x1x32x128xf32>
//   %5 = "ttir.mesh_shard"(%3, %4) <{shard_dims = array<i64: -1>, shard_direction = #ttcore.shard_direction<shard_to_full>, shard_shape = array<i64: 1>, shard_type = #ttcore.shard_type<replicate>}> : (tensor<1x1x1x32x128xf32>, tensor<1x1x1x32x128xf32>) -> tensor<1x1x1x32x128xf32>
//   // C/HECK: "ttnn.mesh_shard"
//   return %5 : tensor<1x1x1x32x128xf32>
// }

// // verify rank coverage : 5 - increase rank on head
// func.func @all_gather_cluster1_set_rank_to_5_head_bf16(%arg0: tensor<1x1x1x32x128xbf16>) -> tensor<1x1x1x32x128xbf16> {
//   %0 = ttir.empty() : tensor<1x1x1x32x64xbf16>
//   %1 = "ttir.mesh_shard"(%arg0, %0) <{shard_dims = array<i64: -1, 4>, shard_direction = #ttcore.shard_direction<full_to_shard>, shard_shape = array<i64: 1, 1, 1, 1, 2>, shard_type = #ttcore.shard_type<devices>}> : (tensor<1x1x1x32x128xbf16>, tensor<1x1x1x32x64xbf16>) -> tensor<1x1x1x32x64xbf16>
//   // C/HECK: "ttnn.mesh_shard"
//   %2 = ttir.empty() : tensor<1x1x1x32x128xbf16>
//   %3 = "ttir.all_gather"(%1, %2) <{all_gather_dim = 4 : si32, replica_groups = dense<[[0, 1]]> : tensor<1x2xi64>, cluster_axis = 1 : ui32}> : (tensor<1x1x1x32x64xbf16>, tensor<1x1x1x32x128xbf16>) -> tensor<1x1x1x32x128xbf16>
//   // C/HECK: "ttnn.all_gather"
//   %4 = ttir.empty() : tensor<1x1x1x32x128xbf16>
//   %5 = "ttir.mesh_shard"(%3, %4) <{shard_dims = array<i64: -1>, shard_direction = #ttcore.shard_direction<shard_to_full>, shard_shape = array<i64: 1>, shard_type = #ttcore.shard_type<replicate>}> : (tensor<1x1x1x32x128xbf16>, tensor<1x1x1x32x128xbf16>) -> tensor<1x1x1x32x128xbf16>
//   // C/HECK: "ttnn.mesh_shard"
//   return %5 : tensor<1x1x1x32x128xbf16>
// }

// // verify rank coverage : 5 - increase rank on tail
// func.func @all_gather_cluster1_set_rank_to_5_tail(%arg0: tensor<1x1x32x128x32xf32>) -> tensor<1x1x32x128x32xf32> {
//   %0 = ttir.empty() : tensor<1x1x32x64x32xf32>
//   %1 = "ttir.mesh_shard"(%arg0, %0) <{shard_dims = array<i64: -1, 3>, shard_direction = #ttcore.shard_direction<full_to_shard>, shard_shape = array<i64: 1, 1, 1, 2, 1>, shard_type = #ttcore.shard_type<devices>}> : (tensor<1x1x32x128x32xf32>, tensor<1x1x32x64x32xf32>) -> tensor<1x1x32x64x32xf32>
//   // C/HECK: "ttnn.mesh_shard"
//   %2 = ttir.empty() : tensor<1x1x32x128x32xf32>
//   %3 = "ttir.all_gather"(%1, %2) <{all_gather_dim = 3 : si32, replica_groups = dense<[[0, 1]]> : tensor<1x2xi64>, cluster_axis = 1 : ui32}> : (tensor<1x1x32x64x32xf32>, tensor<1x1x32x128x32xf32>) -> tensor<1x1x32x128x32xf32>
//   // C/HECK: "ttnn.all_gather"
//   %4 = ttir.empty() : tensor<1x1x32x128x32xf32>
//   %5 = "ttir.mesh_shard"(%3, %4) <{shard_dims = array<i64: -1>, shard_direction = #ttcore.shard_direction<shard_to_full>, shard_shape = array<i64: 1>, shard_type = #ttcore.shard_type<replicate>}> : (tensor<1x1x32x128x32xf32>, tensor<1x1x32x128x32xf32>) -> tensor<1x1x32x128x32xf32>
//   // C/HECK: "ttnn.mesh_shard"
//   return %5 : tensor<1x1x32x128x32xf32>
// }

// // verify rank coverage : 5 - increase rank on tail
// func.func @all_gather_cluster1_set_rank_to_5_tail_bf16(%arg0: tensor<1x1x32x128x32xbf16>) -> tensor<1x1x32x128x32xbf16> {
//   %0 = ttir.empty() : tensor<1x1x32x64x32xbf16>
//   %1 = "ttir.mesh_shard"(%arg0, %0) <{shard_dims = array<i64: -1, 3>, shard_direction = #ttcore.shard_direction<full_to_shard>, shard_shape = array<i64: 1, 1, 1, 2, 1>, shard_type = #ttcore.shard_type<devices>}> : (tensor<1x1x32x128x32xbf16>, tensor<1x1x32x64x32xbf16>) -> tensor<1x1x32x64x32xbf16>
//   // C/HECK: "ttnn.mesh_shard"
//   %2 = ttir.empty() : tensor<1x1x32x128x32xbf16>
//   %3 = "ttir.all_gather"(%1, %2) <{all_gather_dim = 3 : si32, replica_groups = dense<[[0, 1]]> : tensor<1x2xi64>, cluster_axis = 1 : ui32}> : (tensor<1x1x32x64x32xbf16>, tensor<1x1x32x128x32xbf16>) -> tensor<1x1x32x128x32xbf16>
//   // C/HECK: "ttnn.all_gather"
//   %4 = ttir.empty() : tensor<1x1x32x128x32xbf16>
//   %5 = "ttir.mesh_shard"(%3, %4) <{shard_dims = array<i64: -1>, shard_direction = #ttcore.shard_direction<shard_to_full>, shard_shape = array<i64: 1>, shard_type = #ttcore.shard_type<replicate>}> : (tensor<1x1x32x128x32xbf16>, tensor<1x1x32x128x32xbf16>) -> tensor<1x1x32x128x32xbf16>
//   // C/HECK: "ttnn.mesh_shard"
//   return %5 : tensor<1x1x32x128x32xbf16>
// }

// // verify rank coverage : 6 - increase ranks on head
// func.func @all_gather_cluster1_set_rank_to_6_head(%arg0: tensor<1x1x1x1x32x128xf32>) -> tensor<1x1x1x1x32x128xf32> {
//   %0 = ttir.empty() : tensor<1x1x1x1x32x64xf32>
//   %1 = "ttir.mesh_shard"(%arg0, %0) <{shard_dims = array<i64: -1, 5>, shard_direction = #ttcore.shard_direction<full_to_shard>, shard_shape = array<i64: 1, 1, 1, 1, 1, 2>, shard_type = #ttcore.shard_type<devices>}> : (tensor<1x1x1x1x32x128xf32>, tensor<1x1x1x1x32x64xf32>) -> tensor<1x1x1x1x32x64xf32>
//   // C/HECK: "ttnn.mesh_shard"
//   %2 = ttir.empty() : tensor<1x1x1x1x32x128xf32>
//   %3 = "ttir.all_gather"(%1, %2) <{all_gather_dim = 5 : si32, replica_groups = dense<[[0, 1]]> : tensor<1x2xi64>, cluster_axis = 1 : ui32}> : (tensor<1x1x1x1x32x64xf32>, tensor<1x1x1x1x32x128xf32>) -> tensor<1x1x1x1x32x128xf32>
//   // C/HECK: "ttnn.all_gather"
//   %4 = ttir.empty() : tensor<1x1x1x1x32x128xf32>
//   %5 = "ttir.mesh_shard"(%3, %4) <{shard_dims = array<i64: -1>, shard_direction = #ttcore.shard_direction<shard_to_full>, shard_shape = array<i64: 1>, shard_type = #ttcore.shard_type<replicate>}> : (tensor<1x1x1x1x32x128xf32>, tensor<1x1x1x1x32x128xf32>) -> tensor<1x1x1x1x32x128xf32>
//   // C/HECK: "ttnn.mesh_shard"
//   return %5 : tensor<1x1x1x1x32x128xf32>
// }

// // verify rank coverage : 6 - increase ranks on head
// func.func @all_gather_cluster1_set_rank_to_6_head_bf16(%arg0: tensor<1x1x1x1x32x128xbf16>) -> tensor<1x1x1x1x32x128xbf16> {
//   %0 = ttir.empty() : tensor<1x1x1x1x32x64xbf16>
//   %1 = "ttir.mesh_shard"(%arg0, %0) <{shard_dims = array<i64: -1, 5>, shard_direction = #ttcore.shard_direction<full_to_shard>, shard_shape = array<i64: 1, 1, 1, 1, 1, 2>, shard_type = #ttcore.shard_type<devices>}> : (tensor<1x1x1x1x32x128xbf16>, tensor<1x1x1x1x32x64xbf16>) -> tensor<1x1x1x1x32x64xbf16>
//   // C/HECK: "ttnn.mesh_shard"
//   %2 = ttir.empty() : tensor<1x1x1x1x32x128xbf16>
//   %3 = "ttir.all_gather"(%1, %2) <{all_gather_dim = 5 : si32, replica_groups = dense<[[0, 1]]> : tensor<1x2xi64>, cluster_axis = 1 : ui32}> : (tensor<1x1x1x1x32x64xbf16>, tensor<1x1x1x1x32x128xbf16>) -> tensor<1x1x1x1x32x128xbf16>
//   // C/HECK: "ttnn.all_gather"
//   %4 = ttir.empty() : tensor<1x1x1x1x32x128xbf16>
//   %5 = "ttir.mesh_shard"(%3, %4) <{shard_dims = array<i64: -1>, shard_direction = #ttcore.shard_direction<shard_to_full>, shard_shape = array<i64: 1>, shard_type = #ttcore.shard_type<replicate>}> : (tensor<1x1x1x1x32x128xbf16>, tensor<1x1x1x1x32x128xbf16>) -> tensor<1x1x1x1x32x128xbf16>
//   // C/HECK: "ttnn.mesh_shard"
//   return %5 : tensor<1x1x1x1x32x128xbf16>
// }

// // verify rank coverage : 6 - increase ranks on tail
// func.func @all_gather_cluster1_set_rank_to_6_tail(%arg0: tensor<1x1x32x128x32x32xf32>) -> tensor<1x1x32x128x32x32xf32> {
//   %0 = ttir.empty() : tensor<1x1x32x64x32x32xf32>
//   %1 = "ttir.mesh_shard"(%arg0, %0) <{shard_dims = array<i64: -1, 3>, shard_direction = #ttcore.shard_direction<full_to_shard>, shard_shape = array<i64: 1, 1, 1, 2, 1, 1>, shard_type = #ttcore.shard_type<devices>}> : (tensor<1x1x32x128x32x32xf32>, tensor<1x1x32x64x32x32xf32>) -> tensor<1x1x32x64x32x32xf32>
//   // C/HECK: "ttnn.mesh_shard"
//   %2 = ttir.empty() : tensor<1x1x32x128x32x32xf32>
//   %3 = "ttir.all_gather"(%1, %2) <{all_gather_dim = 3 : si32, replica_groups = dense<[[0, 1]]> : tensor<1x2xi64>, cluster_axis = 1 : ui32}> : (tensor<1x1x32x64x32x32xf32>, tensor<1x1x32x128x32x32xf32>) -> tensor<1x1x32x128x32x32xf32>
//   // C/HECK: "ttnn.all_gather"
//   %4 = ttir.empty() : tensor<1x1x32x128x32x32xf32>
//   %5 = "ttir.mesh_shard"(%3, %4) <{shard_dims = array<i64: -1>, shard_direction = #ttcore.shard_direction<shard_to_full>, shard_shape = array<i64: 1>, shard_type = #ttcore.shard_type<replicate>}> : (tensor<1x1x32x128x32x32xf32>, tensor<1x1x32x128x32x32xf32>) -> tensor<1x1x32x128x32x32xf32>
//   // C/HECK: "ttnn.mesh_shard"
//   return %5 : tensor<1x1x32x128x32x32xf32>
// }

// // verify rank coverage : 6 - increase ranks on tail
// func.func @all_gather_cluster1_set_rank_to_6_tail_bf16(%arg0: tensor<1x1x32x128x32x32xbf16>) -> tensor<1x1x32x128x32x32xbf16> {
//   %0 = ttir.empty() : tensor<1x1x32x64x32x32xbf16>
//   %1 = "ttir.mesh_shard"(%arg0, %0) <{shard_dims = array<i64: -1, 3>, shard_direction = #ttcore.shard_direction<full_to_shard>, shard_shape = array<i64: 1, 1, 1, 2, 1, 1>, shard_type = #ttcore.shard_type<devices>}> : (tensor<1x1x32x128x32x32xbf16>, tensor<1x1x32x64x32x32xbf16>) -> tensor<1x1x32x64x32x32xbf16>
//   // C/HECK: "ttnn.mesh_shard"
//   %2 = ttir.empty() : tensor<1x1x32x128x32x32xbf16>
//   %3 = "ttir.all_gather"(%1, %2) <{all_gather_dim = 3 : si32, replica_groups = dense<[[0, 1]]> : tensor<1x2xi64>, cluster_axis = 1 : ui32}> : (tensor<1x1x32x64x32x32xbf16>, tensor<1x1x32x128x32x32xbf16>) -> tensor<1x1x32x128x32x32xbf16>
//   // C/HECK: "ttnn.all_gather"
//   %4 = ttir.empty() : tensor<1x1x32x128x32x32xbf16>
//   %5 = "ttir.mesh_shard"(%3, %4) <{shard_dims = array<i64: -1>, shard_direction = #ttcore.shard_direction<shard_to_full>, shard_shape = array<i64: 1>, shard_type = #ttcore.shard_type<replicate>}> : (tensor<1x1x32x128x32x32xbf16>, tensor<1x1x32x128x32x32xbf16>) -> tensor<1x1x32x128x32x32xbf16>
//   // C/HECK: "ttnn.mesh_shard"
//   return %5 : tensor<1x1x32x128x32x32xbf16>
// }

// func.func @all_gather_cluster1_set_rank_to_8_head(%arg0: tensor<1x1x1x1x1x1x32x128xf32>) -> tensor<1x1x1x1x1x1x32x128xf32> {
//   %0 = ttir.empty() : tensor<1x1x1x1x1x1x32x64xf32>
//   %1 = "ttir.mesh_shard"(%arg0, %0) <{shard_dims = array<i64: -1, 7>, shard_direction = #ttcore.shard_direction<full_to_shard>, shard_shape = array<i64: 1, 1, 1, 1, 1, 1, 1, 2>, shard_type = #ttcore.shard_type<devices>}> : (tensor<1x1x1x1x1x1x32x128xf32>, tensor<1x1x1x1x1x1x32x64xf32>) -> tensor<1x1x1x1x1x1x32x64xf32>
//   // C/HECK: "ttnn.mesh_shard"
//   %2 = ttir.empty() : tensor<1x1x1x1x1x1x32x128xf32>
//   %3 = "ttir.all_gather"(%1, %2) <{all_gather_dim = 7 : si32, replica_groups = dense<[[0, 1]]> : tensor<1x2xi64>, cluster_axis = 1 : ui32}> : (tensor<1x1x1x1x1x1x32x64xf32>, tensor<1x1x1x1x1x1x32x128xf32>) -> tensor<1x1x1x1x1x1x32x128xf32>
//   // C/HECK: "ttnn.all_gather"
//   %4 = ttir.empty() : tensor<1x1x1x1x1x1x32x128xf32>
//   %5 = "ttir.mesh_shard"(%3, %4) <{shard_dims = array<i64: -1>, shard_direction = #ttcore.shard_direction<shard_to_full>, shard_shape = array<i64: 1>, shard_type = #ttcore.shard_type<replicate>}> : (tensor<1x1x1x1x1x1x32x128xf32>, tensor<1x1x1x1x1x1x32x128xf32>) -> tensor<1x1x1x1x1x1x32x128xf32>
//   // C/HECK: "ttnn.mesh_shard"
//   return %5 : tensor<1x1x1x1x1x1x32x128xf32>
// }

// func.func @all_gather_cluster1_set_rank_to_8_head_bf16(%arg0: tensor<1x1x1x1x1x1x32x128xbf16>) -> tensor<1x1x1x1x1x1x32x128xbf16> {
//   %0 = ttir.empty() : tensor<1x1x1x1x1x1x32x64xbf16>
//   %1 = "ttir.mesh_shard"(%arg0, %0) <{shard_dims = array<i64: -1, 7>, shard_direction = #ttcore.shard_direction<full_to_shard>, shard_shape = array<i64: 1, 1, 1, 1, 1, 1, 1, 2>, shard_type = #ttcore.shard_type<devices>}> : (tensor<1x1x1x1x1x1x32x128xbf16>, tensor<1x1x1x1x1x1x32x64xbf16>) -> tensor<1x1x1x1x1x1x32x64xbf16>
//   // C/HECK: "ttnn.mesh_shard"
//   %2 = ttir.empty() : tensor<1x1x1x1x1x1x32x128xbf16>
//   %3 = "ttir.all_gather"(%1, %2) <{all_gather_dim = 7 : si32, replica_groups = dense<[[0, 1]]> : tensor<1x2xi64>, cluster_axis = 1 : ui32}> : (tensor<1x1x1x1x1x1x32x64xbf16>, tensor<1x1x1x1x1x1x32x128xbf16>) -> tensor<1x1x1x1x1x1x32x128xbf16>
//   // C/HECK: "ttnn.all_gather"
//   %4 = ttir.empty() : tensor<1x1x1x1x1x1x32x128xbf16>
//   %5 = "ttir.mesh_shard"(%3, %4) <{shard_dims = array<i64: -1>, shard_direction = #ttcore.shard_direction<shard_to_full>, shard_shape = array<i64: 1>, shard_type = #ttcore.shard_type<replicate>}> : (tensor<1x1x1x1x1x1x32x128xbf16>, tensor<1x1x1x1x1x1x32x128xbf16>) -> tensor<1x1x1x1x1x1x32x128xbf16>
//   // C/HECK: "ttnn.mesh_shard"
//   return %5 : tensor<1x1x1x1x1x1x32x128xbf16>
// }

// func.func @all_gather_cluster1_set_rank_to_8_tail(%arg0: tensor<1x1x1x1x32x128x32x32xf32>) -> tensor<1x1x1x1x32x128x32x32xf32> {
//   %0 = ttir.empty() : tensor<1x1x1x1x32x64x32x32xf32>
//   %1 = "ttir.mesh_shard"(%arg0, %0) <{shard_dims = array<i64: -1, 5>, shard_direction = #ttcore.shard_direction<full_to_shard>, shard_shape = array<i64: 1, 1, 1, 1, 1, 2, 1, 1>, shard_type = #ttcore.shard_type<devices>}> : (tensor<1x1x1x1x32x128x32x32xf32>, tensor<1x1x1x1x32x64x32x32xf32>) -> tensor<1x1x1x1x32x64x32x32xf32>
//   // C/HECK: "ttnn.mesh_shard"
//   %2 = ttir.empty() : tensor<1x1x1x1x32x128x32x32xf32>
//   %3 = "ttir.all_gather"(%1, %2) <{all_gather_dim = 5 : si32, replica_groups = dense<[[0, 1]]> : tensor<1x2xi64>, cluster_axis = 1 : ui32}> : (tensor<1x1x1x1x32x64x32x32xf32>, tensor<1x1x1x1x32x128x32x32xf32>) -> tensor<1x1x1x1x32x128x32x32xf32>
//   // C/HECK: "ttnn.all_gather"
//   %4 = ttir.empty() : tensor<1x1x1x1x32x128x32x32xf32>
//   %5 = "ttir.mesh_shard"(%3, %4) <{shard_dims = array<i64: -1>, shard_direction = #ttcore.shard_direction<shard_to_full>, shard_shape = array<i64: 1>, shard_type = #ttcore.shard_type<replicate>}> : (tensor<1x1x1x1x32x128x32x32xf32>, tensor<1x1x1x1x32x128x32x32xf32>) -> tensor<1x1x1x1x32x128x32x32xf32>
//   // C/HECK: "ttnn.mesh_shard"
//   return %5 : tensor<1x1x1x1x32x128x32x32xf32>
// }

// func.func @all_gather_cluster1_set_rank_to_8_tail_bf16(%arg0: tensor<1x1x1x1x32x128x32x32xbf16>) -> tensor<1x1x1x1x32x128x32x32xbf16> {
//   %0 = ttir.empty() : tensor<1x1x1x1x32x64x32x32xbf16>
//   %1 = "ttir.mesh_shard"(%arg0, %0) <{shard_dims = array<i64: -1, 5>, shard_direction = #ttcore.shard_direction<full_to_shard>, shard_shape = array<i64: 1, 1, 1, 1, 1, 2, 1, 1>, shard_type = #ttcore.shard_type<devices>}> : (tensor<1x1x1x1x32x128x32x32xbf16>, tensor<1x1x1x1x32x64x32x32xbf16>) -> tensor<1x1x1x1x32x64x32x32xbf16>
//   // C/HECK: "ttnn.mesh_shard"
//   %2 = ttir.empty() : tensor<1x1x1x1x32x128x32x32xbf16>
//   %3 = "ttir.all_gather"(%1, %2) <{all_gather_dim = 5 : si32, replica_groups = dense<[[0, 1]]> : tensor<1x2xi64>, cluster_axis = 1 : ui32}> : (tensor<1x1x1x1x32x64x32x32xbf16>, tensor<1x1x1x1x32x128x32x32xbf16>) -> tensor<1x1x1x1x32x128x32x32xbf16>
//   // C/HECK: "ttnn.all_gather"
//   %4 = ttir.empty() : tensor<1x1x1x1x32x128x32x32xbf16>
//   %5 = "ttir.mesh_shard"(%3, %4) <{shard_dims = array<i64: -1>, shard_direction = #ttcore.shard_direction<shard_to_full>, shard_shape = array<i64: 1>, shard_type = #ttcore.shard_type<replicate>}> : (tensor<1x1x1x1x32x128x32x32xbf16>, tensor<1x1x1x1x32x128x32x32xbf16>) -> tensor<1x1x1x1x32x128x32x32xbf16>
//   // C/HECK: "ttnn.mesh_shard"
//   return %5 : tensor<1x1x1x1x32x128x32x32xbf16>
// }

// func.func @all_gather_cluster1_set_rank_to_16_gather_dim_0(%arg0: tensor<128x1x1x1x1x1x1x1x1x1x1x1x1x1x32x32xf32>) -> tensor<128x1x1x1x1x1x1x1x1x1x1x1x1x1x32x32xf32> {
//   %0 = ttir.empty() : tensor<64x1x1x1x1x1x1x1x1x1x1x1x1x1x32x32xf32>
//   %1 = "ttir.mesh_shard"(%arg0, %0) <{shard_dims = array<i64: -1, 0>, shard_direction = #ttcore.shard_direction<full_to_shard>, shard_shape = array<i64: 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1>, shard_type = #ttcore.shard_type<devices>}> : (tensor<128x1x1x1x1x1x1x1x1x1x1x1x1x1x32x32xf32>, tensor<64x1x1x1x1x1x1x1x1x1x1x1x1x1x32x32xf32>) -> tensor<64x1x1x1x1x1x1x1x1x1x1x1x1x1x32x32xf32>
//   // C/HECK: "ttnn.mesh_shard"
//   %2 = ttir.empty() : tensor<128x1x1x1x1x1x1x1x1x1x1x1x1x1x32x32xf32>
//   %3 = "ttir.all_gather"(%1, %2) <{all_gather_dim = 0 : si32, replica_groups = dense<[[0, 1]]> : tensor<1x2xi64>, cluster_axis = 1 : ui32}> : (tensor<64x1x1x1x1x1x1x1x1x1x1x1x1x1x32x32xf32>, tensor<128x1x1x1x1x1x1x1x1x1x1x1x1x1x32x32xf32>) -> tensor<128x1x1x1x1x1x1x1x1x1x1x1x1x1x32x32xf32>
//   // C/HECK: "ttnn.all_gather"
//   %4 = ttir.empty() : tensor<128x1x1x1x1x1x1x1x1x1x1x1x1x1x32x32xf32>
//   %5 = "ttir.mesh_shard"(%3, %4) <{shard_dims = array<i64: -1>, shard_direction = #ttcore.shard_direction<shard_to_full>, shard_shape = array<i64: 1>, shard_type = #ttcore.shard_type<replicate>}> : (tensor<128x1x1x1x1x1x1x1x1x1x1x1x1x1x32x32xf32>, tensor<128x1x1x1x1x1x1x1x1x1x1x1x1x1x32x32xf32>) -> tensor<128x1x1x1x1x1x1x1x1x1x1x1x1x1x32x32xf32>
//   // C/HECK: "ttnn.mesh_shard"
//   return %5 : tensor<128x1x1x1x1x1x1x1x1x1x1x1x1x1x32x32xf32>
// }

// func.func @all_gather_cluster1_set_rank_to_16_gather_dim_0_bf16(%arg0: tensor<128x1x1x1x1x1x1x1x1x1x1x1x1x1x32x32xbf16>) -> tensor<128x1x1x1x1x1x1x1x1x1x1x1x1x1x32x32xbf16> {
//   %0 = ttir.empty() : tensor<64x1x1x1x1x1x1x1x1x1x1x1x1x1x32x32xbf16>
//   %1 = "ttir.mesh_shard"(%arg0, %0) <{shard_dims = array<i64: -1, 0>, shard_direction = #ttcore.shard_direction<full_to_shard>, shard_shape = array<i64: 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1>, shard_type = #ttcore.shard_type<devices>}> : (tensor<128x1x1x1x1x1x1x1x1x1x1x1x1x1x32x32xbf16>, tensor<64x1x1x1x1x1x1x1x1x1x1x1x1x1x32x32xbf16>) -> tensor<64x1x1x1x1x1x1x1x1x1x1x1x1x1x32x32xbf16>
//   // C/HECK: "ttnn.mesh_shard"
//   %2 = ttir.empty() : tensor<128x1x1x1x1x1x1x1x1x1x1x1x1x1x32x32xbf16>
//   %3 = "ttir.all_gather"(%1, %2) <{all_gather_dim = 0 : si32, replica_groups = dense<[[0, 1]]> : tensor<1x2xi64>, cluster_axis = 1 : ui32}> : (tensor<64x1x1x1x1x1x1x1x1x1x1x1x1x1x32x32xbf16>, tensor<128x1x1x1x1x1x1x1x1x1x1x1x1x1x32x32xbf16>) -> tensor<128x1x1x1x1x1x1x1x1x1x1x1x1x1x32x32xbf16>
//   // C/HECK: "ttnn.all_gather"
//   %4 = ttir.empty() : tensor<128x1x1x1x1x1x1x1x1x1x1x1x1x1x32x32xbf16>
//   %5 = "ttir.mesh_shard"(%3, %4) <{shard_dims = array<i64: -1>, shard_direction = #ttcore.shard_direction<shard_to_full>, shard_shape = array<i64: 1>, shard_type = #ttcore.shard_type<replicate>}> : (tensor<128x1x1x1x1x1x1x1x1x1x1x1x1x1x32x32xbf16>, tensor<128x1x1x1x1x1x1x1x1x1x1x1x1x1x32x32xbf16>) -> tensor<128x1x1x1x1x1x1x1x1x1x1x1x1x1x32x32xbf16>
//   // C/HECK: "ttnn.mesh_shard"
//   return %5 : tensor<128x1x1x1x1x1x1x1x1x1x1x1x1x1x32x32xbf16>
// }

// func.func @all_gather_cluster1_set_rank_to_16_gather_dim_1(%arg0: tensor<1x128x1x1x1x1x1x1x1x1x1x1x1x1x32x32xf32>) -> tensor<1x128x1x1x1x1x1x1x1x1x1x1x1x1x32x32xf32> {
//   %0 = ttir.empty() : tensor<1x64x1x1x1x1x1x1x1x1x1x1x1x1x32x32xf32>
//   %1 = "ttir.mesh_shard"(%arg0, %0) <{shard_dims = array<i64: -1, 1>, shard_direction = #ttcore.shard_direction<full_to_shard>, shard_shape = array<i64: 1, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1>, shard_type = #ttcore.shard_type<devices>}> : (tensor<1x128x1x1x1x1x1x1x1x1x1x1x1x1x32x32xf32>, tensor<1x64x1x1x1x1x1x1x1x1x1x1x1x1x32x32xf32>) -> tensor<1x64x1x1x1x1x1x1x1x1x1x1x1x1x32x32xf32>
//   // C/HECK: "ttnn.mesh_shard"
//   %2 = ttir.empty() : tensor<1x128x1x1x1x1x1x1x1x1x1x1x1x1x32x32xf32>
//   %3 = "ttir.all_gather"(%1, %2) <{all_gather_dim = 1 : si32, replica_groups = dense<[[0, 1]]> : tensor<1x2xi64>, cluster_axis = 1 : ui32}> : (tensor<1x64x1x1x1x1x1x1x1x1x1x1x1x1x32x32xf32>, tensor<1x128x1x1x1x1x1x1x1x1x1x1x1x1x32x32xf32>) -> tensor<1x128x1x1x1x1x1x1x1x1x1x1x1x1x32x32xf32>
//   // C/HECK: "ttnn.all_gather"
//   %4 = ttir.empty() : tensor<1x128x1x1x1x1x1x1x1x1x1x1x1x1x32x32xf32>
//   %5 = "ttir.mesh_shard"(%3, %4) <{shard_dims = array<i64: -1>, shard_direction = #ttcore.shard_direction<shard_to_full>, shard_shape = array<i64: 1>, shard_type = #ttcore.shard_type<replicate>}> : (tensor<1x128x1x1x1x1x1x1x1x1x1x1x1x1x32x32xf32>, tensor<1x128x1x1x1x1x1x1x1x1x1x1x1x1x32x32xf32>) -> tensor<1x128x1x1x1x1x1x1x1x1x1x1x1x1x32x32xf32>
//   // C/HECK: "ttnn.mesh_shard"
//   return %5 : tensor<1x128x1x1x1x1x1x1x1x1x1x1x1x1x32x32xf32>
// }

// func.func @all_gather_cluster1_set_rank_to_16_gather_dim_1_bf16(%arg0: tensor<1x128x1x1x1x1x1x1x1x1x1x1x1x1x32x32xbf16>) -> tensor<1x128x1x1x1x1x1x1x1x1x1x1x1x1x32x32xbf16> {
//   %0 = ttir.empty() : tensor<1x64x1x1x1x1x1x1x1x1x1x1x1x1x32x32xbf16>
//   %1 = "ttir.mesh_shard"(%arg0, %0) <{shard_dims = array<i64: -1, 1>, shard_direction = #ttcore.shard_direction<full_to_shard>, shard_shape = array<i64: 1, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1>, shard_type = #ttcore.shard_type<devices>}> : (tensor<1x128x1x1x1x1x1x1x1x1x1x1x1x1x32x32xbf16>, tensor<1x64x1x1x1x1x1x1x1x1x1x1x1x1x32x32xbf16>) -> tensor<1x64x1x1x1x1x1x1x1x1x1x1x1x1x32x32xbf16>
//   // C/HECK: "ttnn.mesh_shard"
//   %2 = ttir.empty() : tensor<1x128x1x1x1x1x1x1x1x1x1x1x1x1x32x32xbf16>
//   %3 = "ttir.all_gather"(%1, %2) <{all_gather_dim = 1 : si32, replica_groups = dense<[[0, 1]]> : tensor<1x2xi64>, cluster_axis = 1 : ui32}> : (tensor<1x64x1x1x1x1x1x1x1x1x1x1x1x1x32x32xbf16>, tensor<1x128x1x1x1x1x1x1x1x1x1x1x1x1x32x32xbf16>) -> tensor<1x128x1x1x1x1x1x1x1x1x1x1x1x1x32x32xbf16>
//   // C/HECK: "ttnn.all_gather"
//   %4 = ttir.empty() : tensor<1x128x1x1x1x1x1x1x1x1x1x1x1x1x32x32xbf16>
//   %5 = "ttir.mesh_shard"(%3, %4) <{shard_dims = array<i64: -1>, shard_direction = #ttcore.shard_direction<shard_to_full>, shard_shape = array<i64: 1>, shard_type = #ttcore.shard_type<replicate>}> : (tensor<1x128x1x1x1x1x1x1x1x1x1x1x1x1x32x32xbf16>, tensor<1x128x1x1x1x1x1x1x1x1x1x1x1x1x32x32xbf16>) -> tensor<1x128x1x1x1x1x1x1x1x1x1x1x1x1x32x32xbf16>
//   // C/HECK: "ttnn.mesh_shard"
//   return %5 : tensor<1x128x1x1x1x1x1x1x1x1x1x1x1x1x32x32xbf16>
// }

// func.func @all_gather_cluster1_set_rank_to_16_head(%arg0: tensor<1x1x1x1x1x1x1x1x1x1x1x1x1x1x32x128xf32>) -> tensor<1x1x1x1x1x1x1x1x1x1x1x1x1x1x32x128xf32> {
//   %0 = ttir.empty() : tensor<1x1x1x1x1x1x1x1x1x1x1x1x1x1x32x64xf32>
//   %1 = "ttir.mesh_shard"(%arg0, %0) <{shard_dims = array<i64: -1, 15>, shard_direction = #ttcore.shard_direction<full_to_shard>, shard_shape = array<i64: 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2>, shard_type = #ttcore.shard_type<devices>}> : (tensor<1x1x1x1x1x1x1x1x1x1x1x1x1x1x32x128xf32>, tensor<1x1x1x1x1x1x1x1x1x1x1x1x1x1x32x64xf32>) -> tensor<1x1x1x1x1x1x1x1x1x1x1x1x1x1x32x64xf32>
//   // C/HECK: "ttnn.mesh_shard"
//   %2 = ttir.empty() : tensor<1x1x1x1x1x1x1x1x1x1x1x1x1x1x32x128xf32>
//   %3 = "ttir.all_gather"(%1, %2) <{all_gather_dim = 15 : si32, replica_groups = dense<[[0, 1]]> : tensor<1x2xi64>, cluster_axis = 1 : ui32}> : (tensor<1x1x1x1x1x1x1x1x1x1x1x1x1x1x32x64xf32>, tensor<1x1x1x1x1x1x1x1x1x1x1x1x1x1x32x128xf32>) -> tensor<1x1x1x1x1x1x1x1x1x1x1x1x1x1x32x128xf32>
//   // C/HECK: "ttnn.all_gather"
//   %4 = ttir.empty() : tensor<1x1x1x1x1x1x1x1x1x1x1x1x1x1x32x128xf32>
//   %5 = "ttir.mesh_shard"(%3, %4) <{shard_dims = array<i64: -1>, shard_direction = #ttcore.shard_direction<shard_to_full>, shard_shape = array<i64: 1>, shard_type = #ttcore.shard_type<replicate>}> : (tensor<1x1x1x1x1x1x1x1x1x1x1x1x1x1x32x128xf32>, tensor<1x1x1x1x1x1x1x1x1x1x1x1x1x1x32x128xf32>) -> tensor<1x1x1x1x1x1x1x1x1x1x1x1x1x1x32x128xf32>
//   // C/HECK: "ttnn.mesh_shard"
//   return %5 : tensor<1x1x1x1x1x1x1x1x1x1x1x1x1x1x32x128xf32>
// }

// func.func @all_gather_cluster1_set_rank_to_16_head_bf16(%arg0: tensor<1x1x1x1x1x1x1x1x1x1x1x1x1x1x32x128xbf16>) -> tensor<1x1x1x1x1x1x1x1x1x1x1x1x1x1x32x128xbf16> {
//   %0 = ttir.empty() : tensor<1x1x1x1x1x1x1x1x1x1x1x1x1x1x32x64xbf16>
//   %1 = "ttir.mesh_shard"(%arg0, %0) <{shard_dims = array<i64: -1, 15>, shard_direction = #ttcore.shard_direction<full_to_shard>, shard_shape = array<i64: 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2>, shard_type = #ttcore.shard_type<devices>}> : (tensor<1x1x1x1x1x1x1x1x1x1x1x1x1x1x32x128xbf16>, tensor<1x1x1x1x1x1x1x1x1x1x1x1x1x1x32x64xbf16>) -> tensor<1x1x1x1x1x1x1x1x1x1x1x1x1x1x32x64xbf16>
//   // C/HECK: "ttnn.mesh_shard"
//   %2 = ttir.empty() : tensor<1x1x1x1x1x1x1x1x1x1x1x1x1x1x32x128xbf16>
//   %3 = "ttir.all_gather"(%1, %2) <{all_gather_dim = 15 : si32, replica_groups = dense<[[0, 1]]> : tensor<1x2xi64>, cluster_axis = 1 : ui32}> : (tensor<1x1x1x1x1x1x1x1x1x1x1x1x1x1x32x64xbf16>, tensor<1x1x1x1x1x1x1x1x1x1x1x1x1x1x32x128xbf16>) -> tensor<1x1x1x1x1x1x1x1x1x1x1x1x1x1x32x128xbf16>
//   // C/HECK: "ttnn.all_gather"
//   %4 = ttir.empty() : tensor<1x1x1x1x1x1x1x1x1x1x1x1x1x1x32x128xbf16>
//   %5 = "ttir.mesh_shard"(%3, %4) <{shard_dims = array<i64: -1>, shard_direction = #ttcore.shard_direction<shard_to_full>, shard_shape = array<i64: 1>, shard_type = #ttcore.shard_type<replicate>}> : (tensor<1x1x1x1x1x1x1x1x1x1x1x1x1x1x32x128xbf16>, tensor<1x1x1x1x1x1x1x1x1x1x1x1x1x1x32x128xbf16>) -> tensor<1x1x1x1x1x1x1x1x1x1x1x1x1x1x32x128xbf16>
//   // C/HECK: "ttnn.mesh_shard"
//   return %5 : tensor<1x1x1x1x1x1x1x1x1x1x1x1x1x1x32x128xbf16>
// }

func.func @all_gather_cluster1_increased_dim_on_gathered_axis(%arg0: tensor<1x1x32x8192xf32>) -> tensor<1x1x32x8192xf32> {
  %0 = ttir.empty() : tensor<1x1x32x4096xf32>
  %1 = "ttir.mesh_shard"(%arg0, %0) <{shard_dims = array<i64: -1, 3>, shard_direction = #ttcore.shard_direction<full_to_shard>, shard_shape = array<i64: 1, 1, 1, 2>, shard_type = #ttcore.shard_type<devices>}> : (tensor<1x1x32x8192xf32>, tensor<1x1x32x4096xf32>) -> tensor<1x1x32x4096xf32>
  // CHECK: "ttnn.mesh_shard"
  %2 = ttir.empty() : tensor<1x1x32x8192xf32>
  %3 = "ttir.all_gather"(%1, %2) <{all_gather_dim = 3 : si32, replica_groups = dense<[[0, 1]]> : tensor<1x2xi64>, cluster_axis = 1 : ui32}> : (tensor<1x1x32x4096xf32>, tensor<1x1x32x8192xf32>) -> tensor<1x1x32x8192xf32>
  // CHECK: "ttnn.all_gather"
  %4 = ttir.empty() : tensor<1x1x32x8192xf32>
  %5 = "ttir.mesh_shard"(%3, %4) <{shard_dims = array<i64: -1>, shard_direction = #ttcore.shard_direction<shard_to_full>, shard_shape = array<i64: 1>, shard_type = #ttcore.shard_type<replicate>}> : (tensor<1x1x32x8192xf32>, tensor<1x1x32x8192xf32>) -> tensor<1x1x32x8192xf32>
  // CHECK: "ttnn.mesh_shard"
  return %5 : tensor<1x1x32x8192xf32>
}

func.func @all_gather_cluster1_increased_dim_on_gathered_axis_bf16(%arg0: tensor<1x1x32x8192xbf16>) -> tensor<1x1x32x8192xbf16> {
  %0 = ttir.empty() : tensor<1x1x32x4096xbf16>
  %1 = "ttir.mesh_shard"(%arg0, %0) <{shard_dims = array<i64: -1, 3>, shard_direction = #ttcore.shard_direction<full_to_shard>, shard_shape = array<i64: 1, 1, 1, 2>, shard_type = #ttcore.shard_type<devices>}> : (tensor<1x1x32x8192xbf16>, tensor<1x1x32x4096xbf16>) -> tensor<1x1x32x4096xbf16>
  // CHECK: "ttnn.mesh_shard"
  %2 = ttir.empty() : tensor<1x1x32x8192xbf16>
  %3 = "ttir.all_gather"(%1, %2) <{all_gather_dim = 3 : si32, replica_groups = dense<[[0, 1]]> : tensor<1x2xi64>, cluster_axis = 1 : ui32}> : (tensor<1x1x32x4096xbf16>, tensor<1x1x32x8192xbf16>) -> tensor<1x1x32x8192xbf16>
  // CHECK: "ttnn.all_gather"
  %4 = ttir.empty() : tensor<1x1x32x8192xbf16>
  %5 = "ttir.mesh_shard"(%3, %4) <{shard_dims = array<i64: -1>, shard_direction = #ttcore.shard_direction<shard_to_full>, shard_shape = array<i64: 1>, shard_type = #ttcore.shard_type<replicate>}> : (tensor<1x1x32x8192xbf16>, tensor<1x1x32x8192xbf16>) -> tensor<1x1x32x8192xbf16>
  // CHECK: "ttnn.mesh_shard"
  return %5 : tensor<1x1x32x8192xbf16>
}

func.func @all_gather_cluster1_increased_dim_on_nongathered_axis(%arg0: tensor<1x1x8192x128xf32>) -> tensor<1x1x8192x128xf32> {
  %0 = ttir.empty() : tensor<1x1x8192x64xf32>
  %1 = "ttir.mesh_shard"(%arg0, %0) <{shard_dims = array<i64: -1, 3>, shard_direction = #ttcore.shard_direction<full_to_shard>, shard_shape = array<i64: 1, 1, 1, 2>, shard_type = #ttcore.shard_type<devices>}> : (tensor<1x1x8192x128xf32>, tensor<1x1x8192x64xf32>) -> tensor<1x1x8192x64xf32>
  // CHECK: "ttnn.mesh_shard"
  %2 = ttir.empty() : tensor<1x1x8192x128xf32>
  %3 = "ttir.all_gather"(%1, %2) <{all_gather_dim = 3 : si32, replica_groups = dense<[[0, 1]]> : tensor<1x2xi64>, cluster_axis = 1 : ui32}> : (tensor<1x1x8192x64xf32>, tensor<1x1x8192x128xf32>) -> tensor<1x1x8192x128xf32>
  // CHECK: "ttnn.all_gather"
  %4 = ttir.empty() : tensor<1x1x8192x128xf32>
  %5 = "ttir.mesh_shard"(%3, %4) <{shard_dims = array<i64: -1>, shard_direction = #ttcore.shard_direction<shard_to_full>, shard_shape = array<i64: 1>, shard_type = #ttcore.shard_type<replicate>}> : (tensor<1x1x8192x128xf32>, tensor<1x1x8192x128xf32>) -> tensor<1x1x8192x128xf32>
  // CHECK: "ttnn.mesh_shard"
  return %5 : tensor<1x1x8192x128xf32>
}

func.func @all_gather_cluster1_increased_dim_on_nongathered_axis_bf16(%arg0: tensor<1x1x8192x128xbf16>) -> tensor<1x1x8192x128xbf16> {
  %0 = ttir.empty() : tensor<1x1x8192x64xbf16>
  %1 = "ttir.mesh_shard"(%arg0, %0) <{shard_dims = array<i64: -1, 3>, shard_direction = #ttcore.shard_direction<full_to_shard>, shard_shape = array<i64: 1, 1, 1, 2>, shard_type = #ttcore.shard_type<devices>}> : (tensor<1x1x8192x128xbf16>, tensor<1x1x8192x64xbf16>) -> tensor<1x1x8192x64xbf16>
  // CHECK: "ttnn.mesh_shard"
  %2 = ttir.empty() : tensor<1x1x8192x128xbf16>
  %3 = "ttir.all_gather"(%1, %2) <{all_gather_dim = 3 : si32, replica_groups = dense<[[0, 1]]> : tensor<1x2xi64>, cluster_axis = 1 : ui32}> : (tensor<1x1x8192x64xbf16>, tensor<1x1x8192x128xbf16>) -> tensor<1x1x8192x128xbf16>
  // CHECK: "ttnn.all_gather"
  %4 = ttir.empty() : tensor<1x1x8192x128xbf16>
  %5 = "ttir.mesh_shard"(%3, %4) <{shard_dims = array<i64: -1>, shard_direction = #ttcore.shard_direction<shard_to_full>, shard_shape = array<i64: 1>, shard_type = #ttcore.shard_type<replicate>}> : (tensor<1x1x8192x128xbf16>, tensor<1x1x8192x128xbf16>) -> tensor<1x1x8192x128xbf16>
  // CHECK: "ttnn.mesh_shard"
  return %5 : tensor<1x1x8192x128xbf16>
}

func.func @all_gather_cluster1_further_increased_dim_on_gathered_axis(%arg0: tensor<1x1x32x131072xf32>) -> tensor<1x1x32x131072xf32> {
  %0 = ttir.empty() : tensor<1x1x32x65536xf32>
  %1 = "ttir.mesh_shard"(%arg0, %0) <{shard_dims = array<i64: -1, 3>, shard_direction = #ttcore.shard_direction<full_to_shard>, shard_shape = array<i64: 1, 1, 1, 2>, shard_type = #ttcore.shard_type<devices>}> : (tensor<1x1x32x131072xf32>, tensor<1x1x32x65536xf32>) -> tensor<1x1x32x65536xf32>
  // CHECK: "ttnn.mesh_shard"
  %2 = ttir.empty() : tensor<1x1x32x131072xf32>
  %3 = "ttir.all_gather"(%1, %2) <{all_gather_dim = 3 : si32, replica_groups = dense<[[0, 1]]> : tensor<1x2xi64>, cluster_axis = 1 : ui32}> : (tensor<1x1x32x65536xf32>, tensor<1x1x32x131072xf32>) -> tensor<1x1x32x131072xf32>
  // CHECK: "ttnn.all_gather"
  %4 = ttir.empty() : tensor<1x1x32x131072xf32>
  %5 = "ttir.mesh_shard"(%3, %4) <{shard_dims = array<i64: -1>, shard_direction = #ttcore.shard_direction<shard_to_full>, shard_shape = array<i64: 1>, shard_type = #ttcore.shard_type<replicate>}> : (tensor<1x1x32x131072xf32>, tensor<1x1x32x131072xf32>) -> tensor<1x1x32x131072xf32>
  // CHECK: "ttnn.mesh_shard"
  return %5 : tensor<1x1x32x131072xf32>
}

func.func @all_gather_cluster1_further_increased_dim_on_nongathered_axis(%arg0: tensor<1x1x131072x128xf32>) -> tensor<1x1x131072x128xf32> {
  %0 = ttir.empty() : tensor<1x1x131072x64xf32>
  %1 = "ttir.mesh_shard"(%arg0, %0) <{shard_dims = array<i64: -1, 3>, shard_direction = #ttcore.shard_direction<full_to_shard>, shard_shape = array<i64: 1, 1, 1, 2>, shard_type = #ttcore.shard_type<devices>}> : (tensor<1x1x131072x128xf32>, tensor<1x1x131072x64xf32>) -> tensor<1x1x131072x64xf32>
  // CHECK: "ttnn.mesh_shard"
  %2 = ttir.empty() : tensor<1x1x131072x128xf32>
  %3 = "ttir.all_gather"(%1, %2) <{all_gather_dim = 3 : si32, replica_groups = dense<[[0, 1]]> : tensor<1x2xi64>, cluster_axis = 1 : ui32}> : (tensor<1x1x131072x64xf32>, tensor<1x1x131072x128xf32>) -> tensor<1x1x131072x128xf32>
  // CHECK: "ttnn.all_gather"
  %4 = ttir.empty() : tensor<1x1x131072x128xf32>
  %5 = "ttir.mesh_shard"(%3, %4) <{shard_dims = array<i64: -1>, shard_direction = #ttcore.shard_direction<shard_to_full>, shard_shape = array<i64: 1>, shard_type = #ttcore.shard_type<replicate>}> : (tensor<1x1x131072x128xf32>, tensor<1x1x131072x128xf32>) -> tensor<1x1x131072x128xf32>
  // CHECK: "ttnn.mesh_shard"
  return %5 : tensor<1x1x131072x128xf32>
}

func.func @all_gather_cluster1_further_increased_dim_on_nongathered_axis_bf16(%arg0: tensor<1x1x131072x128xbf16>) -> tensor<1x1x131072x128xbf16> {
  %0 = ttir.empty() : tensor<1x1x131072x64xbf16>
  %1 = "ttir.mesh_shard"(%arg0, %0) <{shard_dims = array<i64: -1, 3>, shard_direction = #ttcore.shard_direction<full_to_shard>, shard_shape = array<i64: 1, 1, 1, 2>, shard_type = #ttcore.shard_type<devices>}> : (tensor<1x1x131072x128xbf16>, tensor<1x1x131072x64xbf16>) -> tensor<1x1x131072x64xbf16>
  // CHECK: "ttnn.mesh_shard"
  %2 = ttir.empty() : tensor<1x1x131072x128xbf16>
  %3 = "ttir.all_gather"(%1, %2) <{all_gather_dim = 3 : si32, replica_groups = dense<[[0, 1]]> : tensor<1x2xi64>, cluster_axis = 1 : ui32}> : (tensor<1x1x131072x64xbf16>, tensor<1x1x131072x128xbf16>) -> tensor<1x1x131072x128xbf16>
  // CHECK: "ttnn.all_gather"
  %4 = ttir.empty() : tensor<1x1x131072x128xbf16>
  %5 = "ttir.mesh_shard"(%3, %4) <{shard_dims = array<i64: -1>, shard_direction = #ttcore.shard_direction<shard_to_full>, shard_shape = array<i64: 1>, shard_type = #ttcore.shard_type<replicate>}> : (tensor<1x1x131072x128xbf16>, tensor<1x1x131072x128xbf16>) -> tensor<1x1x131072x128xbf16>
  // CHECK: "ttnn.mesh_shard"
  return %5 : tensor<1x1x131072x128xbf16>
}
