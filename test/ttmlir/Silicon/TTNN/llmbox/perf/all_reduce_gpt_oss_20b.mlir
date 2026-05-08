// RUN: ttmlir-opt --ttir-to-ttnn-backend-pipeline="system-desc-path=%system_desc_path% mesh-shape=1,8" -o %t.mlir %s
// RUN: FileCheck %s --input-file=%t.mlir
// RUN: ttmlir-translate --ttnn-to-flatbuffer -o %t.ttnn %t.mlir

// Regression test for tt-metal#39648: all_reduce on a 1x8 mesh with local
// shape 544x2880 (bf16) used to OOM in L1.
func.func @all_reduce_gpt_oss_20b(%arg0: tensor<1x1x4352x2880xbf16>) -> tensor<1x1x4352x2880xbf16> {
  %1 = "ttir.mesh_shard"(%arg0) <{shard_dims = array<i64: -1, 2>, shard_direction = #ttcore.shard_direction<full_to_shard>, shard_shape = array<i64: 1, 1, 8, 1>, shard_type = #ttcore.shard_type<devices>}> : (tensor<1x1x4352x2880xbf16>) -> tensor<1x1x544x2880xbf16>
  %3 = "ttir.all_reduce"(%1) <{cluster_axis = 1 : ui32, reduce_type = #ttcore.reduce_type<sum>}> : (tensor<1x1x544x2880xbf16>) -> tensor<1x1x544x2880xbf16>
  // CHECK: "ttnn.all_reduce"
  %5 = "ttir.mesh_shard"(%3) <{shard_dims = array<i64: -1, 2>, shard_direction = #ttcore.shard_direction<shard_to_full>, shard_shape = array<i64: 1, 1, 8, 1>, shard_type = #ttcore.shard_type<devices>}> : (tensor<1x1x544x2880xbf16>) -> tensor<1x1x4352x2880xbf16>
  return %5 : tensor<1x1x4352x2880xbf16>
}
