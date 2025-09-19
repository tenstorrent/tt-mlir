// RUN: ttmlir-opt --ttir-to-ttmetal-pipeline="system-desc-path=%system_desc_path% mesh-shape=1,2" -o %t.mlir %s
// RUN: FileCheck %s --input-file=%t.mlir
// RUN: ttmlir-translate --ttmetal-to-flatbuffer -o %t.ttm %t.mlir

module @jit_matmul_shardy1 attributes {mhlo.num_partitions = 8 : i32, mhlo.num_replicas = 1 : i32, ttcore.meshes = #ttcore.meshes<[<"mesh" = 1x2>]>} {
  func.func public @main(%arg0: tensor<512x1024xf32> {ttcore.shard_status = #ttcore.shard_status<unsharded>}, %arg1: tensor<512x1024xf32> {ttcore.shard_status = #ttcore.shard_status<unsharded>}) -> (tensor<512x1024xf32> {jax.result_info = "", ttcore.shard_status = #ttcore.shard_status<unsharded>}) {
    %1 = "ttir.mesh_shard"(%arg0) <{shard_dims = array<i64: 0, 1>, shard_direction = #ttcore.shard_direction<full_to_shard>, shard_shape = array<i64: 1, 2>, shard_type = #ttcore.shard_type<devices>}> : (tensor<512x1024xf32>) -> tensor<512x512xf32>
    // CHECK: "ttmetal.mesh_shard"
    %3 = "ttir.mesh_shard"(%arg1) <{shard_dims = array<i64: 0, 1>, shard_direction = #ttcore.shard_direction<full_to_shard>, shard_shape = array<i64: 1, 2>, shard_type = #ttcore.shard_type<devices>}> : (tensor<512x1024xf32>) -> tensor<512x512xf32>
    // CHECK: "ttmetal.mesh_shard"
    %4 = ttir.empty() : tensor<512x512xf32>
    %5 = "ttir.add"(%1, %3, %4) : (tensor<512x512xf32>, tensor<512x512xf32>, tensor<512x512xf32>) -> tensor<512x512xf32>
    %7 = "ttir.mesh_shard"(%5) <{shard_dims = array<i64: 0, 1>, shard_direction = #ttcore.shard_direction<shard_to_full>, shard_shape = array<i64: 1, 2>, shard_type = #ttcore.shard_type<devices>}> : (tensor<512x512xf32>) -> tensor<512x1024xf32>
    // CHECK: "ttmetal.mesh_shard"
    return %7 : tensor<512x1024xf32>
  }
}
