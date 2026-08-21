// REQUIRES: stablehlo
// RUN: ttmlir-opt --fuse-distributed-custom-calls --split-input-file -o %t.mlir %s
// RUN: FileCheck %s --input-file=%t.mlir

// Test that all_gather + rms_norm + composite sdy.all_slice fuses into distributed_rms_norm.
// CHECK-LABEL: func.func @fuse_rms_norm_composite_all_slice
module {
  func.func @fuse_rms_norm_composite_all_slice(%arg0: tensor<4x32xf32>,
                                               %arg1: tensor<32xf32>) -> tensor<4x32xf32> {
    %gathered = "stablehlo.all_gather"(%arg0) <{
      all_gather_dim = 1 : i64,
      channel_handle = #stablehlo.channel_handle<handle = 1, type = 1>,
      replica_groups = dense<[[0, 1]]> : tensor<1x2xi64>,
      use_global_device_ids
    }> : (tensor<4x32xf32>) -> tensor<4x64xf32>
    %weight = "stablehlo.all_gather"(%arg1) <{
      all_gather_dim = 0 : i64,
      channel_handle = #stablehlo.channel_handle<handle = 2, type = 1>,
      replica_groups = dense<[[0, 1]]> : tensor<1x2xi64>,
      use_global_device_ids
    }> : (tensor<32xf32>) -> tensor<64xf32>
    %norm = stablehlo.custom_call @tenstorrent.rms_norm(%gathered, %weight) {
      tt.composite_attributes = {
        epsilon = 1.000000e-05 : f32,
        normalized_shape = dense<64> : tensor<1xi64>
      },
      tt.has_custom_sharding
    } : (tensor<4x64xf32>, tensor<64xf32>) -> tensor<4x64xf32>
    %result = stablehlo.composite "sdy.all_slice" %norm
      {decomposition = @all_slice_impl} : (tensor<4x64xf32>) -> tensor<4x32xf32>
    // CHECK: stablehlo.custom_call @tenstorrent.distributed_rms_norm
    // CHECK-SAME: cluster_axis = 1 : i32
    // CHECK-NOT: stablehlo.all_gather
    // CHECK-NOT: stablehlo.custom_call @tenstorrent.rms_norm
    // CHECK-NOT: stablehlo.composite "sdy.all_slice
    return %result : tensor<4x32xf32>
  }

  func.func private @all_slice_impl(%arg0: tensor<4x64xf32>) -> tensor<4x32xf32> {
    %0 = stablehlo.slice %arg0 [0:4, 0:32] : (tensor<4x64xf32>) -> tensor<4x32xf32>
    return %0 : tensor<4x32xf32>
  }
}

// -----

// Test that all_gather + layer_norm + all_slice preserves affine operands and
// fuses into distributed_layer_norm.
// CHECK-LABEL: func.func @fuse_layer_norm_composite_all_slice
module {
  func.func @fuse_layer_norm_composite_all_slice(
      %arg0: tensor<4x32xf32>, %weight: tensor<32xf32>,
      %bias: tensor<32xf32>) -> tensor<4x32xf32> {
    %gathered = "stablehlo.all_gather"(%arg0) <{
      all_gather_dim = 1 : i64,
      channel_handle = #stablehlo.channel_handle<handle = 1, type = 1>,
      replica_groups = dense<[[0, 1]]> : tensor<1x2xi64>,
      use_global_device_ids
    }> : (tensor<4x32xf32>) -> tensor<4x64xf32>
    %gathered_weight = "stablehlo.all_gather"(%weight) <{
      all_gather_dim = 0 : i64,
      channel_handle = #stablehlo.channel_handle<handle = 2, type = 1>,
      replica_groups = dense<[[0, 1]]> : tensor<1x2xi64>,
      use_global_device_ids
    }> : (tensor<32xf32>) -> tensor<64xf32>
    %gathered_bias = "stablehlo.all_gather"(%bias) <{
      all_gather_dim = 0 : i64,
      channel_handle = #stablehlo.channel_handle<handle = 3, type = 1>,
      replica_groups = dense<[[0, 1]]> : tensor<1x2xi64>,
      use_global_device_ids
    }> : (tensor<32xf32>) -> tensor<64xf32>
    %norm = stablehlo.custom_call @tenstorrent.layer_norm(
        %gathered, %gathered_weight, %gathered_bias) {
      tt.composite_attributes = {
        epsilon = 1.000000e-05 : f32,
        normalized_shape = dense<64> : tensor<1xi64>
      },
      tt.has_custom_sharding
    } : (tensor<4x64xf32>, tensor<64xf32>, tensor<64xf32>)
        -> tensor<4x64xf32>
    %result = stablehlo.composite "sdy.all_slice" %norm
      {decomposition = @all_slice_impl} :
      (tensor<4x64xf32>) -> tensor<4x32xf32>
    // CHECK: stablehlo.custom_call @tenstorrent.distributed_layer_norm
    // CHECK-SAME: (%arg0, %arg1, %arg2)
    // CHECK-SAME: cluster_axis = 1 : i32
    // CHECK-NOT: stablehlo.all_gather
    // CHECK-NOT: stablehlo.custom_call @tenstorrent.layer_norm
    return %result : tensor<4x32xf32>
  }

  func.func private @all_slice_impl(
      %arg0: tensor<4x64xf32>) -> tensor<4x32xf32> {
    %0 = stablehlo.slice %arg0 [0:4, 0:32] :
      (tensor<4x64xf32>) -> tensor<4x32xf32>
    return %0 : tensor<4x32xf32>
  }
}

// -----

// Test that all_gather + rms_norm + decomposed sdy.all_slice fuses into distributed_rms_norm.
// The decomposed form (reshape -> all_to_all -> slice -> reshape) is emitted by
// UpdateGlobalToLocalShapes and not simplified by ShardyToStableHLOAllSliceOpRewritePattern
// when the all_slice input is not fully replicated.
// CHECK-LABEL: func.func @fuse_rms_norm_inlined_all_slice
module {
  func.func @fuse_rms_norm_inlined_all_slice(%arg0: tensor<4x32xf32>,
                                             %arg1: tensor<32xf32>) -> tensor<4x32xf32> {
    %gathered = "stablehlo.all_gather"(%arg0) <{
      all_gather_dim = 1 : i64,
      channel_handle = #stablehlo.channel_handle<handle = 1, type = 1>,
      replica_groups = dense<[[0, 1]]> : tensor<1x2xi64>,
      use_global_device_ids
    }> : (tensor<4x32xf32>) -> tensor<4x64xf32>
    %weight = "stablehlo.all_gather"(%arg1) <{
      all_gather_dim = 0 : i64,
      channel_handle = #stablehlo.channel_handle<handle = 2, type = 1>,
      replica_groups = dense<[[0, 1]]> : tensor<1x2xi64>,
      use_global_device_ids
    }> : (tensor<32xf32>) -> tensor<64xf32>
    %norm = stablehlo.custom_call @tenstorrent.rms_norm(%gathered, %weight) {
      tt.composite_attributes = {
        epsilon = 1.000000e-05 : f32,
        normalized_shape = dense<64> : tensor<1xi64>
      },
      tt.has_custom_sharding
    } : (tensor<4x64xf32>, tensor<64xf32>) -> tensor<4x64xf32>
    %reshape1 = stablehlo.reshape %norm : (tensor<4x64xf32>) -> tensor<4x2x32xf32>
    %all_to_all = "stablehlo.all_to_all"(%reshape1) <{
      channel_handle = #stablehlo.channel_handle<handle = 3, type = 1>,
      concat_dimension = 0 : i64,
      replica_groups = dense<[[0, 1]]> : tensor<1x2xi64>,
      split_count = 2 : i64,
      split_dimension = 1 : i64
    }> : (tensor<4x2x32xf32>) -> tensor<8x1x32xf32>
    %sliced = stablehlo.slice %all_to_all [0:4, 0:1, 0:32] : (tensor<8x1x32xf32>) -> tensor<4x1x32xf32>
    %reshape2 = stablehlo.reshape %sliced : (tensor<4x1x32xf32>) -> tensor<4x32xf32>
    // CHECK: stablehlo.custom_call @tenstorrent.distributed_rms_norm
    // CHECK-SAME: cluster_axis = 1 : i32
    // CHECK-NOT: stablehlo.all_gather
    // CHECK-NOT: stablehlo.custom_call @tenstorrent.rms_norm
    // CHECK-NOT: stablehlo.all_to_all
    return %reshape2 : tensor<4x32xf32>
  }
}

// -----

// Shardy 2D-mesh AdaLN sandwich: all_to_all (split L, concat D) + layer_norm +
// inverse all_to_all. This is cheaper than all_gather + all_slice when L is
// already sharded. Fuses into distributed_layer_norm on the local shard.
// CHECK-LABEL: func.func @fuse_layer_norm_inverse_all_to_all
module {
  func.func @fuse_layer_norm_inverse_all_to_all(%arg0: tensor<1x16x8xf32>) -> tensor<1x16x8xf32> {
    %gathered = "stablehlo.all_to_all"(%arg0) <{
      channel_handle = #stablehlo.channel_handle<handle = 1, type = 1>,
      concat_dimension = 2 : i64,
      replica_groups = dense<[[0, 1, 2, 3], [4, 5, 6, 7]]> : tensor<2x4xi64>,
      split_count = 4 : i64,
      split_dimension = 1 : i64
    }> : (tensor<1x16x8xf32>) -> tensor<1x4x32xf32>
    %norm = stablehlo.custom_call @tenstorrent.layer_norm(%gathered) {
      tt.composite_attributes = {
        epsilon = 9.99999997E-7 : f32,
        normalized_shape = dense<32> : tensor<1xi64>
      },
      tt.has_custom_sharding
    } : (tensor<1x4x32xf32>) -> tensor<1x4x32xf32>
    %result = "stablehlo.all_to_all"(%norm) <{
      channel_handle = #stablehlo.channel_handle<handle = 1, type = 1>,
      concat_dimension = 1 : i64,
      replica_groups = dense<[[0, 1, 2, 3], [4, 5, 6, 7]]> : tensor<2x4xi64>,
      split_count = 4 : i64,
      split_dimension = 2 : i64
    }> : (tensor<1x4x32xf32>) -> tensor<1x16x8xf32>
    // CHECK: stablehlo.custom_call @tenstorrent.distributed_layer_norm
    // CHECK-SAME: (%arg0)
    // CHECK-SAME: cluster_axis = 1 : i32
    // CHECK-NOT: stablehlo.all_to_all
    // CHECK-NOT: stablehlo.custom_call @tenstorrent.layer_norm
    return %result : tensor<1x16x8xf32>
  }
}
