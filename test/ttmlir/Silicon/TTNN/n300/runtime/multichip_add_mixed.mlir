// RUN: ttmlir-opt --ttir-to-ttnn-backend-pipeline="system-desc-path=%system_desc_path% mesh-shape=1,2" -o %t.mlir %s
// RUN: FileCheck %s --input-file=%t.mlir
// RUN: ttmlir-translate --ttnn-to-flatbuffer -o %t.ttnn %t.mlir

// One input sharded across devices, the other replicated to all devices.
module @multichip_add_mixed attributes {ttcore.meshes = #ttcore.meshes<[<"mesh" = 1x2>]>} {
  func.func @main(%arg0: tensor<64x128xf32> {ttcore.shard_status = #ttcore.shard_status<unsharded>}, %arg1: tensor<64x64xf32> {ttcore.shard_status = #ttcore.shard_status<unsharded>}) -> (tensor<64x128xf32> {ttcore.shard_status = #ttcore.shard_status<unsharded>}) {
    %0 = "ttir.mesh_shard"(%arg0) <{shard_dims = array<i64: -1, 1>, shard_direction = #ttcore.shard_direction<full_to_shard>, shard_shape = array<i64: 1, 2>, shard_type = #ttcore.shard_type<devices>}> : (tensor<64x128xf32>) -> tensor<64x64xf32>
    // CHECK: "ttnn.distribute_tensor"
    %1 = "ttir.mesh_shard"(%arg1) <{shard_dims = array<i64: -1>, shard_direction = #ttcore.shard_direction<full_to_shard>, shard_shape = array<i64: 1>, shard_type = #ttcore.shard_type<replicate>}> : (tensor<64x64xf32>) -> tensor<64x64xf32>
    // CHECK: "ttnn.distribute_tensor"
    %2 = "ttir.add"(%0, %1) : (tensor<64x64xf32>, tensor<64x64xf32>) -> tensor<64x64xf32>
    // CHECK: "ttnn.add"
    %3 = "ttir.mesh_shard"(%2) <{shard_dims = array<i64: -1, 1>, shard_direction = #ttcore.shard_direction<shard_to_full>, shard_shape = array<i64: 1, 2>, shard_type = #ttcore.shard_type<devices>}> : (tensor<64x64xf32>) -> tensor<64x128xf32>
    // CHECK: "ttnn.aggregate_tensor"
    return %3 : tensor<64x128xf32>
  }
}
