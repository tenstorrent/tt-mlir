// RUN: ttmlir-opt --ttir-to-ttnn-backend-pipeline="system-desc-path=%system_desc_path% mesh-shape=1,2" %s > %t.mlir
// RUN: FileCheck %s --input-file=%t.mlir
// RUN: ttmlir-translate --ttnn-to-flatbuffer %t.mlir > %t.ttnn

func.func public @jit_tensor_parallel_n300(%arg0: tensor<64x1x1024x2048xf32>, %arg1: tensor<1x1x2048x512xf32>) -> (tensor<64x1x1024x512xf32> {jax.result_info = ""}) {
  %0 = tensor.empty() : tensor<64x1x1024x1024xf32>
  %1 = "ttir.mesh_shard"(%arg0, %0) <{shard_dims = array<i64: -1, 3>, shard_direction = #tt.shard_direction<full_to_shard>, shard_shape = array<i64: 1, 1, 1, 2>, shard_type = #tt.shard_type<devices>}> : (tensor<64x1x1024x2048xf32>, tensor<64x1x1024x1024xf32>) -> tensor<64x1x1024x1024xf32>
  // CHECK: "ttnn.mesh_shard"
  %2 = tensor.empty() : tensor<1x1x1024x512xf32>
  %3 = "ttir.mesh_shard"(%arg1, %2) <{shard_dims = array<i64: -1, 2>, shard_direction = #tt.shard_direction<full_to_shard>, shard_shape = array<i64: 1, 1, 2, 1>, shard_type = #tt.shard_type<devices>}> : (tensor<1x1x2048x512xf32>, tensor<1x1x1024x512xf32>) -> tensor<1x1x1024x512xf32>
  // CHECK: "ttnn.mesh_shard"
  %4 = tensor.empty() : tensor<64x1024x1024xf32>
  %5 = "ttir.reshape"(%1, %4) <{shape = [64 : i32, 1024 : i32, 1024 : i32]}> : (tensor<64x1x1024x1024xf32>, tensor<64x1024x1024xf32>) -> tensor<64x1024x1024xf32>
  // CHECK: = "ttnn.reshape"
  %6 = tensor.empty() : tensor<1x1024x512xf32>
  %7 = "ttir.reshape"(%3, %6) <{shape = [1 : i32, 1024 : i32, 512 : i32]}> : (tensor<1x1x1024x512xf32>, tensor<1x1024x512xf32>) -> tensor<1x1024x512xf32>
  // CHECK: = "ttnn.reshape"
  %8 = "ttir.dot_general"(%5, %7) <{batch_dims_lhs = array<i64>, batch_dims_rhs = array<i64>, contract_dims_lhs = array<i64: 2>, contract_dims_rhs = array<i64: 1>}> : (tensor<64x1024x1024xf32>, tensor<1x1024x512xf32>) -> tensor<64x1024x1x512xf32>
  // CHECK: "ttnn.matmul"
  %9 = tensor.empty() : tensor<64x1x1024x512xf32>
  %10 = "ttir.permute"(%8, %9) <{permutation = array<i64: 0, 2, 1, 3>}> : (tensor<64x1024x1x512xf32>, tensor<64x1x1024x512xf32>) -> tensor<64x1x1024x512xf32>
  // CHECK: "ttnn.permute"
  %11 = tensor.empty() : tensor<64x1x512x512xf32>
  %12 = "ttir.reduce_scatter"(%10, %11) <{cluster_axis = 1 : ui32, reduce_type = #tt.reduce_type<sum>, scatter_dim = 2 : si32}> : (tensor<64x1x1024x512xf32>, tensor<64x1x512x512xf32>) -> tensor<64x1x512x512xf32>
  // CHECK: "ttnn.reduce_scatter"
  %13 = tensor.empty() : tensor<64x1x1024x512xf32>
  %14 = "ttir.mesh_shard"(%12, %13) <{shard_dims = array<i64: -1, 2>, shard_direction = #tt.shard_direction<shard_to_full>, shard_shape = array<i64: 1, 1, 2, 1>, shard_type = #tt.shard_type<devices>}> : (tensor<64x1x512x512xf32>, tensor<64x1x1024x512xf32>) -> tensor<64x1x1024x512xf32>
  // CHECK: "ttnn.mesh_shard"
  return %14 : tensor<64x1x1024x512xf32>
}

func.func public @jit_data_parallel_n300(%arg0: tensor<64x1x1024x2048xf32>, %arg1: tensor<1x1x2048x512xf32>) -> (tensor<64x1x1024x512xf32> {jax.result_info = ""}) {
  %0 = tensor.empty() : tensor<32x1x1024x2048xf32>
  %1 = "ttir.mesh_shard"(%arg0, %0) <{shard_dims = array<i64: -1, 0>, shard_direction = #tt.shard_direction<full_to_shard>, shard_shape = array<i64: 2, 1, 1, 1>, shard_type = #tt.shard_type<devices>}> : (tensor<64x1x1024x2048xf32>, tensor<32x1x1024x2048xf32>) -> tensor<32x1x1024x2048xf32>
  // CHECK: "ttnn.mesh_shard"
  %2 = tensor.empty() : tensor<1x1x2048x512xf32>
  %3 = "ttir.mesh_shard"(%arg1, %2) <{shard_dims = array<i64: -1>, shard_direction = #tt.shard_direction<full_to_shard>, shard_shape = array<i64: 1>, shard_type = #tt.shard_type<replicate>}> : (tensor<1x1x2048x512xf32>, tensor<1x1x2048x512xf32>) -> tensor<1x1x2048x512xf32>
  // CHECK: "ttnn.mesh_shard"
  %4 = tensor.empty() : tensor<32x1024x2048xf32>
  %5 = "ttir.reshape"(%1, %4) <{shape = [32 : i32, 1024 : i32, 2048 : i32]}> : (tensor<32x1x1024x2048xf32>, tensor<32x1024x2048xf32>) -> tensor<32x1024x2048xf32>
  // CHECK: = "ttnn.reshape"
  %6 = tensor.empty() : tensor<1x2048x512xf32>
  %7 = "ttir.reshape"(%3, %6) <{shape = [1 : i32, 2048 : i32, 512 : i32]}> : (tensor<1x1x2048x512xf32>, tensor<1x2048x512xf32>) -> tensor<1x2048x512xf32>
  // CHECK: = "ttnn.reshape"
  %8 = "ttir.dot_general"(%5, %7) <{batch_dims_lhs = array<i64>, batch_dims_rhs = array<i64>, contract_dims_lhs = array<i64: 2>, contract_dims_rhs = array<i64: 1>}> : (tensor<32x1024x2048xf32>, tensor<1x2048x512xf32>) -> tensor<32x1024x1x512xf32>
  // CHECK: "ttnn.matmul"
  %9 = tensor.empty() : tensor<32x1x1024x512xf32>
  %10 = "ttir.permute"(%8, %9) <{permutation = array<i64: 0, 2, 1, 3>}> : (tensor<32x1024x1x512xf32>, tensor<32x1x1024x512xf32>) -> tensor<32x1x1024x512xf32>
  // CHECK: "ttnn.permute"
  %11 = tensor.empty() : tensor<64x1x1024x512xf32>
  %12 = "ttir.mesh_shard"(%10, %11) <{shard_dims = array<i64: -1, 0>, shard_direction = #tt.shard_direction<shard_to_full>, shard_shape = array<i64: 2, 1, 1, 1>, shard_type = #tt.shard_type<devices>}> : (tensor<32x1x1024x512xf32>, tensor<64x1x1024x512xf32>) -> tensor<64x1x1024x512xf32>
  // CHECK: "ttnn.mesh_shard"
  return %12 : tensor<64x1x1024x512xf32>
}
