// RUN: ttmlir-opt --ttir-to-ttnn-backend-pipeline="system-desc-path=%system_desc_path% mesh-shape=8,4" -o %t.mlir %s
// RUN: FileCheck %s --input-file=%t.mlir
// RUN: ttmlir-translate --ttnn-to-flatbuffer -o %t.ttnn %t.mlir

func.func @reduce_scatter_cluster1(%arg0: tensor<1x1x8192x512xf32>) -> (tensor<1x1x8192x128xf32> {jax.result_info = ""}) {
  %1 = "ttir.mesh_shard"(%arg0) <{shard_dims = array<i64: 2, 3>, shard_direction = #ttcore.shard_direction<full_to_shard>, shard_shape = array<i64: 1, 1, 8, 4>, shard_type = #ttcore.shard_type<devices>}> : (tensor<1x1x8192x512xf32>) -> tensor<1x1x1024x128xf32>
  // CHECK: "ttnn.mesh_shard"
  %2 = ttir.empty() : tensor<1x1x1024x32xf32>
  %3 = "ttir.reduce_scatter"(%1, %2) <{cluster_axis = 1 : ui32, reduce_type = #ttcore.reduce_type<sum>, scatter_dim = 3 : si32}> : (tensor<1x1x1024x128xf32>, tensor<1x1x1024x32xf32>) -> tensor<1x1x1024x32xf32>
  // CHECK: "ttnn.reduce_scatter"
  %5 = "ttir.mesh_shard"(%3) <{shard_dims = array<i64: 2, 3>, shard_direction = #ttcore.shard_direction<shard_to_full>, shard_shape = array<i64: 1, 1, 8, 4>, shard_type = #ttcore.shard_type<devices>}> : (tensor<1x1x1024x32xf32>) -> tensor<1x1x8192x128xf32>
  // CHECK: "ttnn.mesh_shard"
  return %5 : tensor<1x1x8192x128xf32>
}

func.func @reduce_scatter_2d_input(%arg0: tensor<8192x512xf32>) -> tensor<1024x128xf32> {
  %0 = ttir.empty() : tensor<1024x128xf32>
  %1 = "ttir.reduce_scatter"(%arg0, %0) <{cluster_axis = 1 : ui32, reduce_type = #ttcore.reduce_type<sum>, scatter_dim = 1 : si32}> : (tensor<8192x512xf32>, tensor<1024x128xf32>) -> tensor<1024x128xf32>
  // CHECK: %{{.*}} = "ttnn.reshape"(%{{.*}}) <{shape = [1 : i32, 1 : i32, 8192 : i32, 512 : i32]}> : (tensor<8192x512xf32, #{{.*}}>) -> tensor<1x1x8192x512xf32, #{{.*}}>
  // CHECK: %{{.*}} = "ttnn.reduce_scatter"(%{{.*}}, %{{.*}}) <{cluster_axis = 1 : ui32, num_links = 1 : ui32, reduce_type = #ttcore.reduce_type<sum>, scatter_dim = 3 : si32}> : (tensor<1x1x8192x512xf32, #{{.*}}>, !ttnn.device) -> tensor<1x1x1024x128xf32, #{{.*}}>
  // CHECK: %{{.*}} = "ttnn.reshape"(%{{.*}}) <{shape = [1024 : i32, 128 : i32]}> : (tensor<1x1x1024x128xf32, #{{.*}}>) -> tensor<1024x128xf32, #{{.*}}>
  return %1 : tensor<1024x128xf32>
}
