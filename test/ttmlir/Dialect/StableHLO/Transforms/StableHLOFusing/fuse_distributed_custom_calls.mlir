// REQUIRES: stablehlo
// RUN: ttmlir-opt --fuse-distributed-custom-calls --split-input-file -o %t.mlir %s
// RUN: FileCheck %s --input-file=%t.mlir

// Test that all_gather + rms_norm + composite sdy.all_slice fuses into distributed_rms_norm.
// CHECK-LABEL: func.func @fuse_rms_norm_composite_all_slice
module {
  func.func @fuse_rms_norm_composite_all_slice(%arg0: tensor<4x64xf32>,
                                               %arg1: tensor<64xf32>) -> tensor<4x64xf32> {
    %gathered = "stablehlo.all_gather"(%arg0) <{
      all_gather_dim = 0 : i64,
      channel_handle = #stablehlo.channel_handle<handle = 1, type = 1>,
      replica_groups = dense<[[0, 1]]> : tensor<1x2xi64>,
      use_global_device_ids
    }> : (tensor<4x64xf32>) -> tensor<8x64xf32>
    %norm = stablehlo.custom_call @tenstorrent.rms_norm(%gathered, %arg1) {
      tt.composite_attributes = {
        epsilon = 1.000000e-05 : f32,
        normalized_shape = dense<64> : tensor<1xi64>
      },
      tt.has_custom_sharding
    } : (tensor<8x64xf32>, tensor<64xf32>) -> tensor<8x64xf32>
    %result = stablehlo.composite "sdy.all_slice" %norm
      {decomposition = @all_slice_impl} : (tensor<8x64xf32>) -> tensor<4x64xf32>
    // CHECK: stablehlo.custom_call @tenstorrent.distributed_rms_norm
    // CHECK-SAME: cluster_axis = 1 : i32
    // CHECK-NOT: stablehlo.all_gather
    // CHECK-NOT: stablehlo.custom_call @tenstorrent.rms_norm
    // CHECK-NOT: stablehlo.composite "sdy.all_slice
    return %result : tensor<4x64xf32>
  }

  func.func private @all_slice_impl(%arg0: tensor<8x64xf32>) -> tensor<4x64xf32> {
    %0 = stablehlo.slice %arg0 [0:4, 0:64] : (tensor<8x64xf32>) -> tensor<4x64xf32>
    return %0 : tensor<4x64xf32>
  }
}

// -----

// Test that all_gather + rms_norm + decomposed sdy.all_slice fuses into distributed_rms_norm.
// The decomposed form (reshape -> all_to_all -> slice -> reshape) is emitted by
// UpdateGlobalToLocalShapes and not simplified by ShardyToStableHLOAllSliceOpRewritePattern
// when the all_slice input is not fully replicated.
// CHECK-LABEL: func.func @fuse_rms_norm_inlined_all_slice
module {
  func.func @fuse_rms_norm_inlined_all_slice(%arg0: tensor<4x64xf32>,
                                             %arg1: tensor<64xf32>) -> tensor<4x64xf32> {
    %gathered = "stablehlo.all_gather"(%arg0) <{
      all_gather_dim = 0 : i64,
      channel_handle = #stablehlo.channel_handle<handle = 1, type = 1>,
      replica_groups = dense<[[0, 1]]> : tensor<1x2xi64>,
      use_global_device_ids
    }> : (tensor<4x64xf32>) -> tensor<8x64xf32>
    %norm = stablehlo.custom_call @tenstorrent.rms_norm(%gathered, %arg1) {
      tt.composite_attributes = {
        epsilon = 1.000000e-05 : f32,
        normalized_shape = dense<64> : tensor<1xi64>
      },
      tt.has_custom_sharding
    } : (tensor<8x64xf32>, tensor<64xf32>) -> tensor<8x64xf32>
    %reshape1 = stablehlo.reshape %norm : (tensor<8x64xf32>) -> tensor<2x4x64xf32>
    %all_to_all = "stablehlo.all_to_all"(%reshape1) <{
      channel_handle = #stablehlo.channel_handle<handle = 2, type = 1>,
      concat_dimension = 1 : i64,
      replica_groups = dense<[[0, 1]]> : tensor<1x2xi64>,
      split_count = 2 : i64,
      split_dimension = 0 : i64
    }> : (tensor<2x4x64xf32>) -> tensor<1x8x64xf32>
    %sliced = stablehlo.slice %all_to_all [0:1, 0:4, 0:64] : (tensor<1x8x64xf32>) -> tensor<1x4x64xf32>
    %reshape2 = stablehlo.reshape %sliced : (tensor<1x4x64xf32>) -> tensor<4x64xf32>
    // CHECK: stablehlo.custom_call @tenstorrent.distributed_rms_norm
    // CHECK-SAME: cluster_axis = 1 : i32
    // CHECK-NOT: stablehlo.all_gather
    // CHECK-NOT: stablehlo.custom_call @tenstorrent.rms_norm
    // CHECK-NOT: stablehlo.all_to_all
    return %reshape2 : tensor<4x64xf32>
  }
}
