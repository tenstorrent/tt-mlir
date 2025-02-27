// RUN: ttmlir-opt --ttir-to-ttnn-backend-pipeline="system-desc-path=%system_desc_path% mesh-shape=2,4" %s > %t.mlir
// RUN: FileCheck %s --input-file=%t.mlir
// RUN: ttmlir-translate --ttnn-to-flatbuffer %t.mlir > %t.ttnn

func.func @reduce_scatter_cluster1(%arg0: tensor<1x1x8192x512xf32>) -> (tensor<1x1x8192x128xf32> {jax.result_info = ""}) {
  %0 = tensor.empty() : tensor<1x1x4096x128xf32>
  %1 = "ttir.mesh_shard"(%arg0, %0) <{shard_dims = array<i64: 2, 3>, shard_direction = #tt.shard_direction<full_to_shard>, shard_shape = array<i64: 1, 1, 2, 4>, shard_type = #tt.shard_type<devices>}> : (tensor<1x1x8192x512xf32>, tensor<1x1x4096x128xf32>) -> tensor<1x1x4096x128xf32>
  // CHECK: "ttnn.mesh_shard"
  %2 = tensor.empty() : tensor<1x1x4096x32xf32>
  %3 = "ttir.reduce_scatter"(%1, %2) <{cluster_axis = 1 : ui32, reduce_type = #tt.reduce_type<sum>, scatter_dim = 3 : si32}> : (tensor<1x1x4096x128xf32>, tensor<1x1x4096x32xf32>) -> tensor<1x1x4096x32xf32>
  // CHECK: "ttnn.reduce_scatter"
  %4 = tensor.empty() : tensor<1x1x8192x128xf32>
  %5 = "ttir.mesh_shard"(%3, %4) <{shard_dims = array<i64: 2, 3>, shard_direction = #tt.shard_direction<shard_to_full>, shard_shape = array<i64: 1, 1, 2, 4>, shard_type = #tt.shard_type<devices>}> : (tensor<1x1x4096x32xf32>, tensor<1x1x8192x128xf32>) -> tensor<1x1x8192x128xf32>
  // CHECK: "ttnn.mesh_shard"
  return %5 : tensor<1x1x8192x128xf32>
}
