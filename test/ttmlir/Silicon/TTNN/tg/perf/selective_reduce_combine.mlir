// RUN: ttmlir-opt --ttir-to-ttnn-backend-pipeline="system-desc-path=%system_desc_path% mesh-shape=4,8" -o %t.mlir %s
// RUN: FileCheck %s --input-file=%t.mlir
// RUN: ttmlir-translate --ttnn-to-flatbuffer -o %t.ttnn %t.mlir

// GPT-OSS MoE selective_reduce_combine shapes:
// mesh=(4,8), hidden=2880, batch=128, seq=1, K=4, experts=128
// cluster_axis=0, per-device: 4 local experts, 4 tokens (128/32)
// Tensors are replicated across the mesh since the op handles its own
// cross-device routing via fabric.

func.func @selective_reduce_combine_gpt_oss(
    %arg0: tensor<4x4x1x2880xbf16>,
    %arg1: tensor<4x4x1x2880xbf16>,
    %arg2: tensor<1x4x1x4xi64>,
    %arg3: tensor<1x4x1x1xi64>
) -> tensor<4x4x1x2880xbf16> {
  %0 = "ttir.mesh_shard"(%arg0) <{shard_dims = array<i64>, shard_direction = #ttcore.shard_direction<full_to_shard>, shard_shape = array<i64: 1, 1, 1, 1>, shard_type = #ttcore.shard_type<replicate>}> : (tensor<4x4x1x2880xbf16>) -> tensor<4x4x1x2880xbf16>
  // CHECK: "ttnn.distribute_tensor"
  %1 = "ttir.mesh_shard"(%arg1) <{shard_dims = array<i64>, shard_direction = #ttcore.shard_direction<full_to_shard>, shard_shape = array<i64: 1, 1, 1, 1>, shard_type = #ttcore.shard_type<replicate>}> : (tensor<4x4x1x2880xbf16>) -> tensor<4x4x1x2880xbf16>
  // CHECK: "ttnn.distribute_tensor"
  %2 = "ttir.mesh_shard"(%arg2) <{shard_dims = array<i64>, shard_direction = #ttcore.shard_direction<full_to_shard>, shard_shape = array<i64: 1, 1, 1, 1>, shard_type = #ttcore.shard_type<replicate>}> : (tensor<1x4x1x4xi64>) -> tensor<1x4x1x4xi64>
  // CHECK: "ttnn.distribute_tensor"
  %3 = "ttir.mesh_shard"(%arg3) <{shard_dims = array<i64>, shard_direction = #ttcore.shard_direction<full_to_shard>, shard_shape = array<i64: 1, 1, 1, 1>, shard_type = #ttcore.shard_type<replicate>}> : (tensor<1x4x1x1xi64>) -> tensor<1x4x1x1xi64>
  // CHECK: "ttnn.distribute_tensor"
  %4 = "ttir.selective_reduce_combine"(%0, %1, %2, %3) <{hidden_size = 2880 : ui32, batch_size = 128 : ui32, seq_size = 1 : ui32, select_experts_k = 4 : ui32, experts = 128 : ui32, axis = 0 : ui32}> : (tensor<4x4x1x2880xbf16>, tensor<4x4x1x2880xbf16>, tensor<1x4x1x4xi64>, tensor<1x4x1x1xi64>) -> tensor<4x4x1x2880xbf16>
  // CHECK: "ttnn.selective_reduce_combine"
  %5 = "ttir.mesh_shard"(%4) <{shard_dims = array<i64>, shard_direction = #ttcore.shard_direction<shard_to_full>, shard_shape = array<i64: 1, 1, 1, 1>, shard_type = #ttcore.shard_type<replicate>}> : (tensor<4x4x1x2880xbf16>) -> tensor<4x4x1x2880xbf16>
  // CHECK: "ttnn.aggregate_tensor"
  return %5 : tensor<4x4x1x2880xbf16>
}
