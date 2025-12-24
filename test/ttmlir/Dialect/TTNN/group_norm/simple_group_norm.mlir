// RUN: ttmlir-opt --split-input-file %s | FileCheck %s

// CHECK-LABEL: func.func @group_norm_basic
func.func @group_norm_basic(%arg0: tensor<2x1x16x32xbf16>) -> tensor<2x1x16x32xbf16> {
  // CHECK: "ttnn.group_norm"
  %0 = "ttnn.group_norm"(%arg0) <{num_groups = 8 : i32, epsilon = 1.000000e-12 : f32, operandSegmentSizes = array<i32: 1, 0, 0>}> : (tensor<2x1x16x32xbf16>) -> tensor<2x1x16x32xbf16>
  return %0 : tensor<2x1x16x32xbf16>
}

// -----
// CHECK-LABEL: func.func @group_norm_with_weight
func.func @group_norm_with_weight(%arg0: tensor<2x1x16x32xbf16>, %arg1: tensor<32xbf16>) -> tensor<2x1x16x32xbf16> {
  // CHECK: "ttnn.group_norm"
  %0 = "ttnn.group_norm"(%arg0, %arg1) <{num_groups = 8 : i32, epsilon = 1.000000e-12 : f32, operandSegmentSizes = array<i32: 1, 1, 0>}> : (tensor<2x1x16x32xbf16>, tensor<32xbf16>) -> tensor<2x1x16x32xbf16>
  return %0 : tensor<2x1x16x32xbf16>
}

// -----
// CHECK-LABEL: func.func @group_norm_with_bias
func.func @group_norm_with_bias(%arg0: tensor<2x1x16x32xbf16>, %arg1: tensor<32xbf16>) -> tensor<2x1x16x32xbf16> {
  // CHECK: "ttnn.group_norm"
  %0 = "ttnn.group_norm"(%arg0, %arg1) <{num_groups = 8 : i32, epsilon = 1.000000e-12 : f32, operandSegmentSizes = array<i32: 1, 0, 1>}> : (tensor<2x1x16x32xbf16>, tensor<32xbf16>) -> tensor<2x1x16x32xbf16>
  return %0 : tensor<2x1x16x32xbf16>
}

// -----
// CHECK-LABEL: func.func @group_norm_with_weight_and_bias
func.func @group_norm_with_weight_and_bias(%arg0: tensor<2x1x16x32xbf16>, %arg1: tensor<32xbf16>, %arg2: tensor<32xbf16>) -> tensor<2x1x16x32xbf16> {
  // CHECK: "ttnn.group_norm"
  %0 = "ttnn.group_norm"(%arg0, %arg1, %arg2) <{num_groups = 8 : i32, epsilon = 1.000000e-12 : f32, operandSegmentSizes = array<i32: 1, 1, 1>}> : (tensor<2x1x16x32xbf16>, tensor<32xbf16>, tensor<32xbf16>) -> tensor<2x1x16x32xbf16>
  return %0 : tensor<2x1x16x32xbf16>
}

// -----
// CHECK-LABEL: func.func @group_norm_16_groups
func.func @group_norm_16_groups(%arg0: tensor<1x1x32x64xbf16>) -> tensor<1x1x32x64xbf16> {
  // CHECK: "ttnn.group_norm"
  %0 = "ttnn.group_norm"(%arg0) <{num_groups = 16 : i32, epsilon = 1.000000e-05 : f32, operandSegmentSizes = array<i32: 1, 0, 0>}> : (tensor<1x1x32x64xbf16>) -> tensor<1x1x32x64xbf16>
  return %0 : tensor<1x1x32x64xbf16>
}

// -----
// CHECK-LABEL: func.func @group_norm_32_groups
func.func @group_norm_32_groups(%arg0: tensor<4x1x8x32xbf16>) -> tensor<4x1x8x32xbf16> {
  // CHECK: "ttnn.group_norm"
  %0 = "ttnn.group_norm"(%arg0) <{num_groups = 32 : i32, epsilon = 1.000000e-12 : f32, operandSegmentSizes = array<i32: 1, 0, 0>}> : (tensor<4x1x8x32xbf16>) -> tensor<4x1x8x32xbf16>
  return %0 : tensor<4x1x8x32xbf16>
}
