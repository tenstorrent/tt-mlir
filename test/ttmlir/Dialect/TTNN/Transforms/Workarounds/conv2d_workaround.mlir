// RUN: ttmlir-opt --ttcore-register-device --ttnn-layout --ttnn-workaround %s | FileCheck %s

module attributes {} {
  func.func @conv2d_with_bias(%arg0: tensor<1x32x32x64xbf16>, %arg1: tensor<64x64x3x3xbf16>, %arg2: tensor<1x1x1x64xbf16>) -> tensor<1x30x30x64xbf16> {
    // CHECK: %[[DEVICE_OP:.*]] = "ttnn.get_device"
    %0 = "ttnn.get_device"() <{mesh_shape = #ttnn<mesh_shape 1x1>}> : () -> !ttnn.device
    %2 = "ttnn.reshape"(%arg0) <{shape = [1 : i32, 1 : i32, 1024 : i32, 64 : i32]}> : (tensor<1x32x32x64xbf16>) -> tensor<1x1x1024x64xbf16>
    // CHECK: %[[TO_LAYOUT_INPUT:.*]] = "ttnn.to_layout"
    // CHECK-SAME: dtype = #ttcore.supportedDataTypes<bf16>
    // CHECK-SAME: layout = #ttnn.layout<row_major>
    %3 = "ttnn.conv2d"(%2, %arg1, %arg2, %0) <{batch_size = 1 : i32, dilation = array<i32: 1, 1>, groups = 1 : i32, in_channels = 64 : i32, input_height = 32 : i32, input_width = 32 : i32, kernel_size = array<i32: 3, 3>, out_channels = 64 : i32, padding = array<i32: 0, 0>, stride = array<i32: 1, 1>}> : (tensor<1x1x1024x64xbf16>, tensor<64x64x3x3xbf16>, tensor<1x1x1x64xbf16>, !ttnn.device) -> tensor<1x1x900x64xbf16>
    // CHECK-NEXT: %[[CONV2D_RESULT:.*]] = "ttnn.conv2d"(%[[TO_LAYOUT_INPUT]], %[[CONV2D_WEIGHTS:.*]], %[[CONV2D_BIAS:.*]], %[[DEVICE_OP]])
    %4 = "ttnn.reshape"(%3) <{shape = [1 : i32, 30 : i32, 30 : i32, 64 : i32]}> : (tensor<1x1x900x64xbf16>) -> tensor<1x30x30x64xbf16>
    return %4 : tensor<1x30x30x64xbf16>
  }

  func.func @conv_transpose2d_with_bias(%arg0: tensor<1x32x32x64xbf16>, %arg1: tensor<64x64x3x3xbf16>, %arg2: tensor<1x1x1x64xbf16>) -> tensor<1x30x30x64xbf16> {
    // CHECK: %[[DEVICE_OP:.*]] = "ttnn.get_device"
    %0 = "ttnn.get_device"() <{mesh_shape = #ttnn<mesh_shape 1x1>}> : () -> !ttnn.device
    // CHECK: %[[TO_LAYOUT_INPUT:.*]] = "ttnn.to_layout"(%arg0, %[[DEVICE_OP]])
    // CHECK-SAME: dtype = #ttcore.supportedDataTypes<bf16>
    // CHECK-SAME: layout = #ttnn.layout<row_major>
    %3 = "ttnn.conv_transpose2d"(%arg0, %arg1, %arg2, %0) <{batch_size = 1 : i32, dilation = array<i32: 1, 1>, groups = 1 : i32, in_channels = 64 : i32, input_height = 32 : i32, input_width = 32 : i32, kernel_size = array<i32: 3, 3>, out_channels = 64 : i32, padding = array<i32: 0, 0>, stride = array<i32: 1, 1>, output_padding = array<i32: 0, 0>}> : (tensor<1x32x32x64xbf16>, tensor<64x64x3x3xbf16>, tensor<1x1x1x64xbf16>, !ttnn.device) -> tensor<1x1x900x64xbf16>
    // CHECK-NEXT: %[[CONV2D_RESULT:.*]] = "ttnn.conv_transpose2d"(%[[TO_LAYOUT_INPUT]], %[[CONV2D_WEIGHTS:.*]], %[[CONV2D_BIAS:.*]], %[[DEVICE_OP]])
    %4 = "ttnn.reshape"(%3) <{shape = [1 : i32, 30 : i32, 30 : i32, 64 : i32]}> : (tensor<1x1x900x64xbf16>) -> tensor<1x30x30x64xbf16>
    return %4 : tensor<1x30x30x64xbf16>
  }

  func.func @conv2d_without_bias(%arg0: tensor<1x32x32x64xbf16>, %arg1: tensor<64x64x3x3xbf16>) -> tensor<1x30x30x64xbf16> {
    // CHECK: %[[DEVICE_OP:.*]] = "ttnn.get_device"
    %0 = "ttnn.get_device"() <{mesh_shape = #ttnn<mesh_shape 1x1>}> : () -> !ttnn.device
    %2 = "ttnn.reshape"(%arg0) <{shape = [1 : i32, 1 : i32, 1024 : i32, 64 : i32]}> : (tensor<1x32x32x64xbf16>) -> tensor<1x1x1024x64xbf16>
    // CHECK: %[[TO_LAYOUT_INPUT:.*]] = "ttnn.to_layout"
    // CHECK-SAME: dtype = #ttcore.supportedDataTypes<bf16>
    // CHECK-SAME: layout = #ttnn.layout<row_major>
    %3 = "ttnn.conv2d"(%2, %arg1, %0) <{batch_size = 1 : i32, dilation = array<i32: 1, 1>, groups = 1 : i32, in_channels = 64 : i32, input_height = 32 : i32, input_width = 32 : i32, kernel_size = array<i32: 3, 3>, out_channels = 64 : i32, padding = array<i32: 0, 0>, stride = array<i32: 1, 1>}> : (tensor<1x1x1024x64xbf16>, tensor<64x64x3x3xbf16>, !ttnn.device) -> tensor<1x1x900x64xbf16>
    // CHECK-NEXT: %[[CONV2D_RESULT:.*]] = "ttnn.conv2d"(%[[TO_LAYOUT_INPUT]], %[[CONV2D_WEIGHTS:.*]], %[[DEVICE_OP]])
    %4 = "ttnn.reshape"(%3) <{shape = [1 : i32, 30 : i32, 30 : i32, 64 : i32]}> : (tensor<1x1x900x64xbf16>) -> tensor<1x30x30x64xbf16>
    return %4 : tensor<1x30x30x64xbf16>
  }

  func.func @conv_transpose2d_without_bias(%arg0: tensor<1x32x32x64xbf16>, %arg1: tensor<64x64x3x3xbf16>) -> tensor<1x30x30x64xbf16> {
    // CHECK: %[[DEVICE_OP:.*]] = "ttnn.get_device"
    %0 = "ttnn.get_device"() <{mesh_shape = #ttnn<mesh_shape 1x1>}> : () -> !ttnn.device
    // CHECK: %[[TO_LAYOUT_INPUT:.*]] = "ttnn.to_layout"(%arg0, %[[DEVICE_OP]])
    // CHECK-SAME: dtype = #ttcore.supportedDataTypes<bf16>
    // CHECK-SAME: layout = #ttnn.layout<row_major>
    %3 = "ttnn.conv_transpose2d"(%arg0, %arg1, %0) <{batch_size = 1 : i32, dilation = array<i32: 1, 1>, groups = 1 : i32, in_channels = 64 : i32, input_height = 32 : i32, input_width = 32 : i32, kernel_size = array<i32: 3, 3>, out_channels = 64 : i32, padding = array<i32: 0, 0>, stride = array<i32: 1, 1>, output_padding = array<i32: 0, 0>}> : (tensor<1x32x32x64xbf16>, tensor<64x64x3x3xbf16>, !ttnn.device) -> tensor<1x1x900x64xbf16>
    // CHECK-NEXT: %[[CONV2D_RESULT:.*]] = "ttnn.conv_transpose2d"(%[[TO_LAYOUT_INPUT]], %[[CONV2D_WEIGHTS:.*]], %[[DEVICE_OP]])
    %4 = "ttnn.reshape"(%3) <{shape = [1 : i32, 30 : i32, 30 : i32, 64 : i32]}> : (tensor<1x1x900x64xbf16>) -> tensor<1x30x30x64xbf16>
    return %4 : tensor<1x30x30x64xbf16>
  }
}
