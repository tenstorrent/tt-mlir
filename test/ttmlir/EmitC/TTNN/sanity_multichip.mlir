// RUN: ttmlir-opt --ttir-to-ttnn-backend-pipeline="system-desc-path=%system_desc_path%" %s > %t.mlir
// RUN: ttmlir-translate --ttnn-to-flatbuffer %t.mlir > %basename_t.ttnn
// RUN: ttmlir-opt --ttnn-modify-signatures-for-dylib --convert-ttnn-to-emitc %t.mlir > %t2.mlir
// RUN: ttmlir-translate --mlir-to-cpp %t2.mlir > %basename_t.cpp

module @jit_fwd attributes {mhlo.num_partitions = 2 : i32, mhlo.num_replicas = 1 : i32, tt.meshes = #tt.meshes<[<"mesh_gspmd" = 1x2>]>} {
  func.func public @main(%arg0: tensor<256x256xf32>, %arg1: tensor<256x256xf32>) -> (tensor<256x256xf32> {jax.result_info = ""}) {
    %0 = tensor.empty() : tensor<256x128xf32>
    %1 = "ttir.mesh_shard"(%arg0, %0) <{shard_dims = array<i64: -1, 1>, shard_direction = #tt.shard_direction<full_to_shard>, shard_shape = array<i64: 1, 2>, shard_type = #tt.shard_type<identity>}> : (tensor<256x256xf32>, tensor<256x128xf32>) -> tensor<256x128xf32>
    %2 = tensor.empty() : tensor<256x128xf32>
    %3 = "ttir.mesh_shard"(%arg1, %2) <{shard_dims = array<i64: -1, 1>, shard_direction = #tt.shard_direction<full_to_shard>, shard_shape = array<i64: 1, 2>, shard_type = #tt.shard_type<identity>}> : (tensor<256x256xf32>, tensor<256x128xf32>) -> tensor<256x128xf32>
    %4 = call @shmap_body(%1, %3) : (tensor<256x128xf32>, tensor<256x128xf32>) -> tensor<256x128xf32>
    %5 = tensor.empty() : tensor<256x256xf32>
    %6 = "ttir.mesh_shard"(%4, %5) <{shard_dims = array<i64: -1, 1>, shard_direction = #tt.shard_direction<shard_to_full>, shard_shape = array<i64: 1, 2>, shard_type = #tt.shard_type<identity>}> : (tensor<256x128xf32>, tensor<256x256xf32>) -> tensor<256x256xf32>
    return %6 : tensor<256x256xf32>
  }
  func.func private @shmap_body(%arg0: tensor<256x128xf32>, %arg1: tensor<256x128xf32>) -> (tensor<256x128xf32> {jax.result_info = "[('batch',), ('model',)]"}) {
    %0 = tensor.empty() : tensor<256x128xf32>
    %1 = "ttir.add"(%arg0, %arg1, %0) <{operandSegmentSizes = array<i32: 2, 1>}> : (tensor<256x128xf32>, tensor<256x128xf32>, tensor<256x128xf32>) -> tensor<256x128xf32>
    return %1 : tensor<256x128xf32>
  }
}
