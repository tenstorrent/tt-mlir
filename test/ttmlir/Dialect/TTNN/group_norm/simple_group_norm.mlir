// RUN: ttmlir-opt --ttir-to-ttnn-backend-pipeline -o %t %s
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
}
