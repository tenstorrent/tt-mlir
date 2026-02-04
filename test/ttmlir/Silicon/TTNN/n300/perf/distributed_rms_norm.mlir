// RUN: ttmlir-opt --ttir-to-ttnn-backend-pipeline="system-desc-path=%system_desc_path% mesh-shape=1,2" -o %t.mlir %s
// RUN: FileCheck %s --input-file=%t.mlir
// RUN: ttmlir-translate --ttnn-to-flatbuffer -o %t.ttnn %t.mlir

// Test distributed RMS norm with all-gather across 2 devices
func.func @distributed_rms_norm_basic(%arg0: tensor<1x1x32x128xbf16>) -> tensor<1x1x32x128xbf16> {
  // Distribute input tensor across devices (shard on dim 3)
  %0 = "ttir.mesh_shard"(%arg0) <{shard_dims = array<i64: -1, 3>, shard_direction = #ttcore.shard_direction<full_to_shard>, shard_shape = array<i64: 1, 1, 1, 2>, shard_type = #ttcore.shard_type<devices>}> : (tensor<1x1x32x128xbf16>) -> tensor<1x1x32x64xbf16>
  // CHECK: "ttnn.distribute_tensor"

  // Distributed RMS norm + all-gather
  %1 = "ttir.distributed_rms_norm"(%0) <{cluster_axis = 1 : ui32, epsilon = 1.000000e-05 : f32}> : (tensor<1x1x32x64xbf16>) -> tensor<1x1x32x128xbf16>
  // CHECK: "ttnn.distributed_rms_norm"

  // Aggregate result back
  %2 = "ttir.mesh_shard"(%1) <{shard_dims = array<i64: -1>, shard_direction = #ttcore.shard_direction<shard_to_full>, shard_shape = array<i64: 1>, shard_type = #ttcore.shard_type<replicate>}> : (tensor<1x1x32x128xbf16>) -> tensor<1x1x32x128xbf16>
  // CHECK: "ttnn.aggregate_tensor"

  return %2 : tensor<1x1x32x128xbf16>
}

// Test distributed RMS norm with weight
func.func @distributed_rms_norm_with_weight(%arg0: tensor<1x1x32x128xbf16>, %weight: tensor<128xbf16>) -> tensor<1x1x32x128xbf16> {
  // Distribute input tensor across devices
  %0 = "ttir.mesh_shard"(%arg0) <{shard_dims = array<i64: -1, 3>, shard_direction = #ttcore.shard_direction<full_to_shard>, shard_shape = array<i64: 1, 1, 1, 2>, shard_type = #ttcore.shard_type<devices>}> : (tensor<1x1x32x128xbf16>) -> tensor<1x1x32x64xbf16>
  // CHECK: "ttnn.distribute_tensor"

  // Distributed RMS norm with weight + all-gather
  %1 = "ttir.distributed_rms_norm"(%0, %weight) <{cluster_axis = 1 : ui32, epsilon = 1.000000e-05 : f32}> : (tensor<1x1x32x64xbf16>, tensor<128xbf16>) -> tensor<1x1x32x128xbf16>
  // CHECK: "ttnn.distributed_rms_norm"

  // Aggregate result back
  %2 = "ttir.mesh_shard"(%1) <{shard_dims = array<i64: -1>, shard_direction = #ttcore.shard_direction<shard_to_full>, shard_shape = array<i64: 1>, shard_type = #ttcore.shard_type<replicate>}> : (tensor<1x1x32x128xbf16>) -> tensor<1x1x32x128xbf16>
  // CHECK: "ttnn.aggregate_tensor"

  return %2 : tensor<1x1x32x128xbf16>
}

// Test distributed RMS norm with weight and residual
func.func @distributed_rms_norm_with_residual(%arg0: tensor<1x1x32x128xbf16>, %weight: tensor<128xbf16>, %residual: tensor<1x1x32x128xbf16>) -> tensor<1x1x32x128xbf16> {
  // Distribute input tensor across devices
  %0 = "ttir.mesh_shard"(%arg0) <{shard_dims = array<i64: -1, 3>, shard_direction = #ttcore.shard_direction<full_to_shard>, shard_shape = array<i64: 1, 1, 1, 2>, shard_type = #ttcore.shard_type<devices>}> : (tensor<1x1x32x128xbf16>) -> tensor<1x1x32x64xbf16>

  // Distribute residual tensor across devices
  %1 = "ttir.mesh_shard"(%residual) <{shard_dims = array<i64: -1, 3>, shard_direction = #ttcore.shard_direction<full_to_shard>, shard_shape = array<i64: 1, 1, 1, 2>, shard_type = #ttcore.shard_type<devices>}> : (tensor<1x1x32x128xbf16>) -> tensor<1x1x32x64xbf16>

  // Distributed RMS norm with weight and residual + all-gather
  %2 = "ttir.distributed_rms_norm"(%0, %weight, %1) <{cluster_axis = 1 : ui32, epsilon = 1.000000e-12 : f32}> : (tensor<1x1x32x64xbf16>, tensor<128xbf16>, tensor<1x1x32x64xbf16>) -> tensor<1x1x32x128xbf16>
  // CHECK: "ttnn.distributed_rms_norm"

  // Aggregate result back
  %3 = "ttir.mesh_shard"(%2) <{shard_dims = array<i64: -1>, shard_direction = #ttcore.shard_direction<shard_to_full>, shard_shape = array<i64: 1>, shard_type = #ttcore.shard_type<replicate>}> : (tensor<1x1x32x128xbf16>) -> tensor<1x1x32x128xbf16>
  // CHECK: "ttnn.aggregate_tensor"

  return %3 : tensor<1x1x32x128xbf16>
}
