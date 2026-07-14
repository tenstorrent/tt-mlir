// RUN: ttmlir-opt --ttir-to-ttnn-backend-pipeline="enable-ttnn-decomposition-pass=false" -o %t %s
// RUN: FileCheck %s --input-file=%t

module {
  // Test basic group norm without optional operands
  func.func @forward(%arg0: tensor<1x1x64x480xbf16>) -> tensor<1x1x64x480xbf16> {
    // CHECK: "ttnn.group_norm"
    %1 = "ttir.group_norm"(%arg0) <{num_groups = 8 : i64, epsilon = 1.000000e-12 : f32, operandSegmentSizes = array<i32: 1, 0, 0, 0>}> : (tensor<1x1x64x480xbf16>) -> tensor<1x1x64x480xbf16>
    return %1 : tensor<1x1x64x480xbf16>
  }

  // Test group norm with bias
  func.func @forward_with_bias(%arg0: tensor<1x1x64x480xbf16>, %arg1: tensor<480xbf16>) -> tensor<1x1x64x480xbf16> {
    // CHECK: "ttnn.group_norm"
    %1 = "ttir.group_norm"(%arg0, %arg1) <{num_groups = 8 : i64, epsilon = 1.000000e-12 : f32, operandSegmentSizes = array<i32: 1, 0, 0, 1>}> : (tensor<1x1x64x480xbf16>, tensor<480xbf16>) -> tensor<1x1x64x480xbf16>
    return %1 : tensor<1x1x64x480xbf16>
  }

    // Test group norm with weight
  func.func @forward_with_weight(%arg0: tensor<1x1x64x480xbf16>, %arg1: tensor<480xbf16>) -> tensor<1x1x64x480xbf16> {
    // CHECK: "ttnn.group_norm"
    %1 = "ttir.group_norm"(%arg0, %arg1) <{num_groups = 8 : i64, epsilon = 1.000000e-12 : f32, operandSegmentSizes = array<i32: 1, 0, 1, 0>}> : (tensor<1x1x64x480xbf16>, tensor<480xbf16>) -> tensor<1x1x64x480xbf16>
    return %1 : tensor<1x1x64x480xbf16>
  }

  // Test group norm with weight and bias
  func.func @forward_with_weight_and_bias(%arg0: tensor<1x1x64x480xbf16>, %arg1: tensor<480xbf16>, %arg2: tensor<480xbf16>) -> tensor<1x1x64x480xbf16> {
    // CHECK: "ttnn.group_norm"
    %1 = "ttir.group_norm"(%arg0, %arg1, %arg2) <{num_groups = 8 : i64, epsilon = 1.000000e-05 : f32, operandSegmentSizes = array<i32: 1, 0, 1, 1>}> : (tensor<1x1x64x480xbf16>, tensor<480xbf16>, tensor<480xbf16>) -> tensor<1x1x64x480xbf16>
    return %1 : tensor<1x1x64x480xbf16>
  }

  // Test group norm with input mask only
  func.func @forward_with_input_mask(%arg0: tensor<1x1x64x480xbf16>, %arg1: tensor<1x8x32x64xbf16>) -> tensor<1x1x64x480xbf16> {
    // CHECK: "ttnn.group_norm"
    %1 = "ttir.group_norm"(%arg0, %arg1) <{num_groups = 8 : i64, epsilon = 1.000000e-12 : f32, operandSegmentSizes = array<i32: 1, 1, 0, 0>}> : (tensor<1x1x64x480xbf16>, tensor<1x8x32x64xbf16>) -> tensor<1x1x64x480xbf16>
    return %1 : tensor<1x1x64x480xbf16>
  }

  // Test group norm with input mask and bias
  func.func @forward_with_input_mask_and_bias(%arg0: tensor<1x1x64x480xbf16>, %arg1: tensor<1x8x32x64xbf16>, %arg2: tensor<480xbf16>) -> tensor<1x1x64x480xbf16> {
    // CHECK: "ttnn.group_norm"
    %1 = "ttir.group_norm"(%arg0, %arg1, %arg2) <{num_groups = 8 : i64, epsilon = 1.000000e-12 : f32, operandSegmentSizes = array<i32: 1, 1, 0, 1>}> : (tensor<1x1x64x480xbf16>, tensor<1x8x32x64xbf16>, tensor<480xbf16>) -> tensor<1x1x64x480xbf16>
    return %1 : tensor<1x1x64x480xbf16>
  }

  // Test group norm with input mask and weight
  func.func @forward_with_input_mask_and_weight(%arg0: tensor<1x1x64x480xbf16>, %arg1: tensor<1x8x32x64xbf16>, %arg2: tensor<480xbf16>) -> tensor<1x1x64x480xbf16> {
    // CHECK: "ttnn.group_norm"
    %1 = "ttir.group_norm"(%arg0, %arg1, %arg2) <{num_groups = 8 : i64, epsilon = 1.000000e-12 : f32, operandSegmentSizes = array<i32: 1, 1, 1, 0>}> : (tensor<1x1x64x480xbf16>, tensor<1x8x32x64xbf16>, tensor<480xbf16>) -> tensor<1x1x64x480xbf16>
    return %1 : tensor<1x1x64x480xbf16>
  }

  // Test group norm with input mask, weight and bias
  func.func @forward_with_input_mask_weight_and_bias(%arg0: tensor<1x1x64x480xbf16>, %arg1: tensor<1x8x32x64xbf16>, %arg2: tensor<480xbf16>, %arg3: tensor<480xbf16>) -> tensor<1x1x64x480xbf16> {
    // CHECK: "ttnn.group_norm"
    %1 = "ttir.group_norm"(%arg0, %arg1, %arg2, %arg3) <{num_groups = 8 : i64, epsilon = 1.000000e-05 : f32, operandSegmentSizes = array<i32: 1, 1, 1, 1>}> : (tensor<1x1x64x480xbf16>, tensor<1x8x32x64xbf16>, tensor<480xbf16>, tensor<480xbf16>) -> tensor<1x1x64x480xbf16>
    return %1 : tensor<1x1x64x480xbf16>
  }

  // Test group norm with reshape when dim1 != 1
  func.func @forward_with_reshape(%arg0: tensor<1x8x32x480xbf16>) -> tensor<1x8x32x480xbf16> {
    // CHECK: "ttnn.reshape"
    // CHECK: "ttnn.group_norm"
    // CHECK: "ttnn.reshape"
    %1 = "ttir.group_norm"(%arg0) <{num_groups = 8 : i64, epsilon = 1.000000e-12 : f32, operandSegmentSizes = array<i32: 1, 0, 0, 0>}> : (tensor<1x8x32x480xbf16>) -> tensor<1x8x32x480xbf16>
    return %1 : tensor<1x8x32x480xbf16>
  }

  // Test group norm with reshape and weight/bias
  func.func @forward_with_reshape_weight_bias(%arg0: tensor<1x8x32x480xbf16>, %arg1: tensor<480xbf16>, %arg2: tensor<480xbf16>) -> tensor<1x8x32x480xbf16> {
    // CHECK: "ttnn.reshape"
    // CHECK: "ttnn.group_norm"
    // CHECK: "ttnn.reshape"
    %1 = "ttir.group_norm"(%arg0, %arg1, %arg2) <{num_groups = 8 : i64, epsilon = 1.000000e-05 : f32, operandSegmentSizes = array<i32: 1, 0, 1, 1>}> : (tensor<1x8x32x480xbf16>, tensor<480xbf16>, tensor<480xbf16>) -> tensor<1x8x32x480xbf16>
    return %1 : tensor<1x8x32x480xbf16>
  }

  // Test group norm with NCHW input (channel_dim=1), requires permute + reshape
  func.func @forward_nchw(%arg0: tensor<1x480x1x64xbf16>) -> tensor<1x480x1x64xbf16> {
    // CHECK: "ttnn.permute"
    // CHECK: "ttnn.group_norm"
    // CHECK: "ttnn.permute"
    %1 = "ttir.group_norm"(%arg0) <{num_groups = 8 : i64, epsilon = 1.000000e-12 : f32, channel_dim = 1 : i64, operandSegmentSizes = array<i32: 1, 0, 0, 0>}> : (tensor<1x480x1x64xbf16>) -> tensor<1x480x1x64xbf16>
    return %1 : tensor<1x480x1x64xbf16>
  }

  // Test group norm with NCHW input and weight/bias
  func.func @forward_nchw_weight_bias(%arg0: tensor<1x480x1x64xbf16>, %arg1: tensor<480xbf16>, %arg2: tensor<480xbf16>) -> tensor<1x480x1x64xbf16> {
    // CHECK: "ttnn.permute"
    // CHECK: "ttnn.group_norm"
    // CHECK: "ttnn.permute"
    %1 = "ttir.group_norm"(%arg0, %arg1, %arg2) <{num_groups = 8 : i64, epsilon = 1.000000e-05 : f32, channel_dim = 1 : i64, operandSegmentSizes = array<i32: 1, 0, 1, 1>}> : (tensor<1x480x1x64xbf16>, tensor<480xbf16>, tensor<480xbf16>) -> tensor<1x480x1x64xbf16>
    return %1 : tensor<1x480x1x64xbf16>
  }

  // Test 5D group norm with NDHWC input (channel_dim=4): only spatial collapse,
  // no permute needed. Spatial dims D*H*W = 4*8*8 = 256 are flattened.
  func.func @forward_5d_ndhwc(%arg0: tensor<1x4x8x8x480xbf16>) -> tensor<1x4x8x8x480xbf16> {
    // CHECK-NOT: "ttnn.permute"
    // CHECK: "ttnn.reshape"
    // CHECK: "ttnn.group_norm"
    // CHECK: "ttnn.reshape"
    // CHECK-NOT: "ttnn.permute"
    %1 = "ttir.group_norm"(%arg0) <{num_groups = 8 : i64, epsilon = 1.000000e-12 : f32, channel_dim = 4 : i64, operandSegmentSizes = array<i32: 1, 0, 0, 0>}> : (tensor<1x4x8x8x480xbf16>) -> tensor<1x4x8x8x480xbf16>
    return %1 : tensor<1x4x8x8x480xbf16>
  }

  // Test 5D group norm with NCDHW input (channel_dim=1): needs both permute
  // (NCDHW -> NDHWC) and reshape (collapse spatial dims).
  func.func @forward_5d_ncdhw(%arg0: tensor<1x480x4x8x8xbf16>) -> tensor<1x480x4x8x8xbf16> {
    // CHECK: "ttnn.permute"
    // CHECK: "ttnn.reshape"
    // CHECK: "ttnn.group_norm"
    // CHECK: "ttnn.reshape"
    // CHECK: "ttnn.permute"
    %1 = "ttir.group_norm"(%arg0) <{num_groups = 8 : i64, epsilon = 1.000000e-12 : f32, channel_dim = 1 : i64, operandSegmentSizes = array<i32: 1, 0, 0, 0>}> : (tensor<1x480x4x8x8xbf16>) -> tensor<1x480x4x8x8xbf16>
    return %1 : tensor<1x480x4x8x8xbf16>
  }

  // Test 5D group norm with NCDHW input and weight/bias.
  func.func @forward_5d_ncdhw_weight_bias(%arg0: tensor<1x480x4x8x8xbf16>, %arg1: tensor<480xbf16>, %arg2: tensor<480xbf16>) -> tensor<1x480x4x8x8xbf16> {
    // CHECK: "ttnn.permute"
    // CHECK: "ttnn.reshape"
    // CHECK: "ttnn.group_norm"
    // CHECK: "ttnn.reshape"
    // CHECK: "ttnn.permute"
    %1 = "ttir.group_norm"(%arg0, %arg1, %arg2) <{num_groups = 8 : i64, epsilon = 1.000000e-05 : f32, channel_dim = 1 : i64, operandSegmentSizes = array<i32: 1, 0, 1, 1>}> : (tensor<1x480x4x8x8xbf16>, tensor<480xbf16>, tensor<480xbf16>) -> tensor<1x480x4x8x8xbf16>
    return %1 : tensor<1x480x4x8x8xbf16>
  }

  // Test non-tile-aligned flattened height: N*H*W = 50 is not a multiple of 32,
  // so the fused ttnn.group_norm kernel cannot represent this shape. The
  // conversion decomposes into primitive ops instead (no ttnn.group_norm).
  func.func @forward_non_tile_aligned(%arg0: tensor<1x1x50x480xbf16>) -> tensor<1x1x50x480xbf16> {
    // CHECK-LABEL: func.func @forward_non_tile_aligned
    // CHECK-NOT: "ttnn.group_norm"
    // Group-split reshape [N, 1, S, C] -> [N, S, G, Cpg].
    // CHECK: "ttnn.reshape"
    // CHECK-SAME: shape = [1 : i32, 50 : i32, 8 : i32, 60 : i32]
    // mean / center / variance.
    // CHECK: "ttnn.mean"
    // CHECK: "ttnn.subtract"
    // CHECK: "ttnn.multiply"
    // CHECK: "ttnn.mean"
    // eps + rsqrt + normalize.
    // CHECK: "ttnn.rsqrt"
    // CHECK: "ttnn.multiply"
    // Restore canonical [N, 1, S, C] shape; no affine tail without weight/bias.
    // CHECK: "ttnn.reshape"
    // CHECK-SAME: shape = [1 : i32, 1 : i32, 50 : i32, 480 : i32]
    %1 = "ttir.group_norm"(%arg0) <{num_groups = 8 : i64, epsilon = 1.000000e-12 : f32, operandSegmentSizes = array<i32: 1, 0, 0, 0>}> : (tensor<1x1x50x480xbf16>) -> tensor<1x1x50x480xbf16>
    return %1 : tensor<1x1x50x480xbf16>
  }

  // Test non-tile-aligned flattened height with weight and bias: decomposes and
  // emits the affine tail (weight multiply + bias add) after normalization.
  func.func @forward_non_tile_aligned_weight_bias(%arg0: tensor<1x1x50x480xbf16>, %arg1: tensor<480xbf16>, %arg2: tensor<480xbf16>) -> tensor<1x1x50x480xbf16> {
    // CHECK-LABEL: func.func @forward_non_tile_aligned_weight_bias
    // CHECK-NOT: "ttnn.group_norm"
    // CHECK: "ttnn.reshape"
    // CHECK-SAME: shape = [1 : i32, 50 : i32, 8 : i32, 60 : i32]
    // CHECK: "ttnn.rsqrt"
    // Restore canonical shape, then affine tail: weight reshape [C] -> [1,1,1,C]
    // and multiply, bias reshape and add.
    // CHECK: "ttnn.reshape"
    // CHECK-SAME: shape = [1 : i32, 1 : i32, 50 : i32, 480 : i32]
    // CHECK: "ttnn.reshape"
    // CHECK-SAME: shape = [1 : i32, 1 : i32, 1 : i32, 480 : i32]
    // CHECK: "ttnn.multiply"
    // CHECK: "ttnn.reshape"
    // CHECK-SAME: shape = [1 : i32, 1 : i32, 1 : i32, 480 : i32]
    // CHECK: "ttnn.add"
    %1 = "ttir.group_norm"(%arg0, %arg1, %arg2) <{num_groups = 8 : i64, epsilon = 1.000000e-05 : f32, operandSegmentSizes = array<i32: 1, 0, 1, 1>}> : (tensor<1x1x50x480xbf16>, tensor<480xbf16>, tensor<480xbf16>) -> tensor<1x1x50x480xbf16>
    return %1 : tensor<1x1x50x480xbf16>
  }
}
