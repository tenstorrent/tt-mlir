// RUN: ttmlir-opt --ttir-to-ttnn-backend-pipeline="system-desc-path=%system_desc_path% mesh-shape=1,2" -o %t.mlir %s
// RUN: FileCheck %s --input-file=%t.mlir
// RUN: ttmlir-translate --ttnn-to-flatbuffer -o %t.ttnn %t.mlir

// CHECK-LABEL: func.func private @forward_const_eval_0
// CHECK: ttnn.distribute_tensor
// CHECK: ttnn.aggregate_tensor

// CHECK-LABEL: func.func @forward
module {
  func.func @forward(%arg0: tensor<4096x4096xbf16> {ttcore.argument_type = #ttcore.argument_type<parameter>}) -> tensor<4096x4096xbf16> {
    %0 = "ttir.mesh_shard"(%arg0) <{shard_dims = array<i64: -1, 1>, shard_direction = #ttcore.shard_direction<full_to_shard>, shard_shape = array<i64: 1, 2>, shard_type = #ttcore.shard_type<devices>}> : (tensor<4096x4096xbf16>) -> tensor<4096x512xbf16>
    %1 = "ttir.mesh_shard"(%0) <{shard_dims = array<i64: -1, 1>, shard_direction = #ttcore.shard_direction<shard_to_full>, shard_shape = array<i64: 1, 2>, shard_type = #ttcore.shard_type<devices>}> : (tensor<4096x512xbf16>) -> tensor<4096x4096xbf16>
    return %1 : tensor<4096x4096xbf16>
  }
}
