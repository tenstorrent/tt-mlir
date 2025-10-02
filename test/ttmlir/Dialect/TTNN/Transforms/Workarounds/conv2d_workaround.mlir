// RUN: ttmlir-opt --ttcore-register-device --ttnn-layout --ttnn-workaround --mlir-print-local-scope -o %t %s
// RUN: FileCheck %s --input-file=%t

module attributes {} {

  func.func @conv_transpose2d_with_bias(%arg0: tensor<1x32x32x64xbf16>, %arg1: tensor<64x64x3x3xbf16>, %arg2: tensor<1x1x1x64xbf16>) -> tensor<1x30x30x64xbf16> {
    // CHECK: %[[DEVICE_OP:.*]] = "ttnn.get_device"
    %0 = "ttnn.get_device"() <{mesh_shape = #ttnn<mesh_shape 1x1>}> : () -> !ttnn.device
    // CHECK: %[[CONV2D_WEIGHTS:.*]] = "ttnn.to_layout"
    // CHECK-SAME: dtype = #ttcore.supportedDataTypes<bf16>
    // CHECK-SAME: layout = #ttnn.layout<row_major>
    // CHECK-SAME: system_memory
    // CHECK: %[[CONV2D_BIAS:.*]] = "ttnn.to_layout"
    // CHECK-SAME: dtype = #ttcore.supportedDataTypes<bf16>
    // CHECK-SAME: layout = #ttnn.layout<row_major>
    // CHECK-SAME: system_memory
    // CHECK: %[[CONV2D_INPUT:.*]] = "ttnn.to_layout"
    // CHECK-SAME: layout = #ttnn.layout<row_major>
    %1 = "ttnn.conv_transpose2d"(%arg0, %arg1, %arg2, %0)
          <{
            batch_size = 1 : i32,
            dilation = array<i32: 1, 1>,
            groups = 1 : i32,
            in_channels = 64 : i32,
            input_height = 32 : i32,
            input_width = 32 : i32,
            kernel_size = array<i32: 3, 3>,
            out_channels = 64 : i32,
            padding = array<i32: 0, 0>,
            stride = array<i32: 1, 1>,
            output_padding = array<i32: 0, 0>,
            dtype = #ttcore.supportedDataTypes<bf16>
          }> : (tensor<1x32x32x64xbf16>, tensor<64x64x3x3xbf16>, tensor<1x1x1x64xbf16>, !ttnn.device) -> tensor<1x1x900x64xbf16>
    // CHECK-NEXT: "ttnn.conv_transpose2d"(%[[CONV2D_INPUT]], %[[CONV2D_WEIGHTS]], %[[CONV2D_BIAS]], %[[DEVICE_OP]])
    // CHECK-SAME: !ttcore.tile<32x32, bf16>
    %2 = "ttnn.reshape"(%1) <{shape = [1 : i32, 30 : i32, 30 : i32, 64 : i32]}> : (tensor<1x1x900x64xbf16>) -> tensor<1x30x30x64xbf16>
    return %2 : tensor<1x30x30x64xbf16>
  }

  func.func @conv_transpose2d_without_bias(%arg0: tensor<1x32x32x64xbf16>, %arg1: tensor<64x64x3x3xbf16>) -> tensor<1x30x30x64xbf16> {
    // CHECK: %[[DEVICE_OP:.*]] = "ttnn.get_device"
    %0 = "ttnn.get_device"() <{mesh_shape = #ttnn<mesh_shape 1x1>}> : () -> !ttnn.device
    // CHECK: %[[CONV2D_WEIGHTS:.*]] = "ttnn.to_layout"
    // CHECK-SAME: dtype = #ttcore.supportedDataTypes<bf16>
    // CHECK-SAME: layout = #ttnn.layout<row_major>
    // CHECK-SAME: system_memory
    // CHECK: %[[CONV2D_INPUT:.*]] = "ttnn.to_layout"
    // CHECK-SAME: layout = #ttnn.layout<row_major>
    %1 = "ttnn.conv_transpose2d"(%arg0, %arg1, %0)
          <{
            batch_size = 1 : i32,
            dilation = array<i32: 1, 1>,
            groups = 1 : i32,
            in_channels = 64 : i32,
            input_height = 32 : i32,
            input_width = 32 : i32,
            kernel_size = array<i32: 3, 3>,
            out_channels = 64 : i32,
            padding = array<i32: 0, 0>,
            stride = array<i32: 1, 1>,
            output_padding = array<i32: 0, 0>,
            dtype = #ttcore.supportedDataTypes<bf16>
          }> : (tensor<1x32x32x64xbf16>, tensor<64x64x3x3xbf16>, !ttnn.device) -> tensor<1x1x900x64xbf16>
    // CHECK-NEXT: "ttnn.conv_transpose2d"(%[[CONV2D_INPUT]], %[[CONV2D_WEIGHTS]], %[[DEVICE_OP]])
    // CHECK-SAME: !ttcore.tile<32x32, bf16>
    %2 = "ttnn.reshape"(%1) <{shape = [1 : i32, 30 : i32, 30 : i32, 64 : i32]}> : (tensor<1x1x900x64xbf16>) -> tensor<1x30x30x64xbf16>
    return %2 : tensor<1x30x30x64xbf16>
  }
}
