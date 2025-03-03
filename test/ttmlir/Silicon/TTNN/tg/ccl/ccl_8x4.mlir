// RUN: ttmlir-opt --ttir-to-ttnn-backend-pipeline="system-desc-path=%system_desc_path% mesh-shape=8,4" %s > %t.mlir
// RUN: FileCheck %s --input-file=%t.mlir
// RUN: ttmlir-translate --ttnn-to-flatbuffer %t.mlir > %t.ttnn

func.func public @all_gather_cluster0(%arg0: tensor<1x1x8192x512xf32>) -> (tensor<1x1x65536x512xf32> {jax.result_info = ""}) {
  %0 = tensor.empty() : tensor<1x1x1024x128xf32>
  %1 = "ttir.mesh_shard"(%arg0, %0) <{shard_dims = array<i64: 2, 3>, shard_direction = #tt.shard_direction<full_to_shard>, shard_shape = array<i64: 1, 1, 8, 4>, shard_type = #tt.shard_type<devices>}> : (tensor<1x1x8192x512xf32>, tensor<1x1x1024x128xf32>) -> tensor<1x1x1024x128xf32>
  // CHECK: "ttnn.mesh_shard"
  %2 = tensor.empty() : tensor<1x1x8192x128xf32>
  %3 = "ttir.all_gather"(%1, %2) <{all_gather_dim = 2 : si32, cluster_axis = 0 : ui32}> : (tensor<1x1x1024x128xf32>, tensor<1x1x8192x128xf32>) -> tensor<1x1x8192x128xf32>
  // CHECK: "ttnn.all_gather"
  %4 = tensor.empty() : tensor<1x1x65536x512xf32>
  %5 = "ttir.mesh_shard"(%3, %4) <{shard_dims = array<i64: 2, 3>, shard_direction = #tt.shard_direction<shard_to_full>, shard_shape = array<i64: 1, 1, 8, 4>, shard_type = #tt.shard_type<devices>}> : (tensor<1x1x8192x128xf32>, tensor<1x1x65536x512xf32>) -> tensor<1x1x65536x512xf32>
  // CHECK: "ttnn.mesh_shard"
  return %5 : tensor<1x1x65536x512xf32>
}

func.func @all_gather_cluster1(%arg0: tensor<1x1x256x512xf32>) -> tensor<1x1x256x512xf32> {
  %0 = tensor.empty() : tensor<1x1x32x128xf32>
  %1 = "ttir.mesh_shard"(%arg0, %0) <{shard_dims = array<i64: 2, 3>, shard_direction = #tt.shard_direction<full_to_shard>, shard_shape = array<i64: 1, 1, 8, 4>, shard_type = #tt.shard_type<devices>}> : (tensor<1x1x256x512xf32>, tensor<1x1x32x128xf32>) -> tensor<1x1x32x128xf32>
  // CHECK: "ttnn.mesh_shard"
  %2 = tensor.empty() : tensor<1x1x32x512xf32>
  %3 = "ttir.all_gather"(%1, %2) <{all_gather_dim = 3 : si32}> : (tensor<1x1x32x128xf32>, tensor<1x1x32x512xf32>) -> tensor<1x1x32x512xf32>
  // CHECK: "ttnn.all_gather"
  %4 = tensor.empty() : tensor<1x1x256x512xf32>
  %5 = "ttir.mesh_shard"(%3, %4) <{shard_dims = array<i64: 2, -1>, shard_direction = #tt.shard_direction<shard_to_full>, shard_shape = array<i64: 1, 1, 8, 1>, shard_type = #tt.shard_type<devices>}> : (tensor<1x1x32x512xf32>, tensor<1x1x256x512xf32>) -> tensor<1x1x256x512xf32>
  // CHECK: "ttnn.mesh_shard"
  return %5 : tensor<1x1x256x512xf32>
}

func.func @all_reduce_cluster0(%arg0: tensor<1x1x8192x512xf32>) -> (tensor<1x1x8192x64xf32> {jax.result_info = ""}) {
  %0 = tensor.empty() : tensor<1x1x2048x64xf32>
  %1 = "ttir.mesh_shard"(%arg0, %0) <{shard_dims = array<i64: 3, 2>, shard_direction = #tt.shard_direction<full_to_shard>, shard_shape = array<i64: 1, 1, 4, 8>, shard_type = #tt.shard_type<devices>}> : (tensor<1x1x8192x512xf32>, tensor<1x1x2048x64xf32>) -> tensor<1x1x2048x64xf32>
  // CHECK: "ttnn.mesh_shard"
  %2 = tensor.empty() : tensor<1x1x2048x64xf32>
  %3 = "ttir.all_reduce"(%1, %2) <{cluster_axis = 0 : ui32, reduce_type = #tt.reduce_type<sum>}> : (tensor<1x1x2048x64xf32>, tensor<1x1x2048x64xf32>) -> tensor<1x1x2048x64xf32>
  // CHECK: "ttnn.reduce_scatter"
  // CHECK: "ttnn.all_gather"
  %4 = tensor.empty() : tensor<1x1x8192x64xf32>
  %5 = "ttir.mesh_shard"(%3, %4) <{shard_dims = array<i64: -1, 2>, shard_direction = #tt.shard_direction<shard_to_full>, shard_shape = array<i64: 1, 1, 4, 1>, shard_type = #tt.shard_type<devices>}> : (tensor<1x1x2048x64xf32>, tensor<1x1x8192x64xf32>) -> tensor<1x1x8192x64xf32>
  // CHECK: "ttnn.mesh_shard"
  return %5 : tensor<1x1x8192x64xf32>
}

func.func @all_reduce_cluster1(%arg0: tensor<1x1x8192x512xf32>) -> (tensor<1x1x8192x128xf32> {jax.result_info = ""}) {
  %0 = tensor.empty() : tensor<1x1x1024x128xf32>
  %1 = "ttir.mesh_shard"(%arg0, %0) <{shard_dims = array<i64: 2, 3>, shard_direction = #tt.shard_direction<full_to_shard>, shard_shape = array<i64: 1, 1, 8, 4>, shard_type = #tt.shard_type<devices>}> : (tensor<1x1x8192x512xf32>, tensor<1x1x1024x128xf32>) -> tensor<1x1x1024x128xf32>
  // CHECK: "ttnn.mesh_shard"
  %2 = tensor.empty() : tensor<1x1x1024x128xf32>
  %3 = "ttir.all_reduce"(%1, %2) <{cluster_axis = 1 : ui32, reduce_type = #tt.reduce_type<sum>}> : (tensor<1x1x1024x128xf32>, tensor<1x1x1024x128xf32>) -> tensor<1x1x1024x128xf32>
  // CHECK: "ttnn.reduce_scatter"
  // CHECK: "ttnn.all_gather"
  %4 = tensor.empty() : tensor<1x1x8192x128xf32>
  %5 = "ttir.mesh_shard"(%3, %4) <{shard_dims = array<i64: 2, -1>, shard_direction = #tt.shard_direction<shard_to_full>, shard_shape = array<i64: 1, 1, 8, 1>, shard_type = #tt.shard_type<devices>}> : (tensor<1x1x1024x128xf32>, tensor<1x1x8192x128xf32>) -> tensor<1x1x8192x128xf32>
  // CHECK: "ttnn.mesh_shard"
  return %5 : tensor<1x1x8192x128xf32>
}
