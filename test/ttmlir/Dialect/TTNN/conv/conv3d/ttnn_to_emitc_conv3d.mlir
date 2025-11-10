// RUN: ttmlir-opt --ttnn-to-emitc -o %t %s
// RUN: FileCheck %s --input-file=%t

module {
  // Test simple Conv3d EmitC conversion
  func.func @test_conv3d_emitc(%arg0: tensor<1x8x28x28x4xbf16>, %arg1: tensor<1x1x108x16xbf16>) -> tensor<1x6x26x26x16xbf16> {
    %device = "ttnn.get_device"() <{mesh_shape = #ttnn<mesh_shape 1x1>}> : () -> !ttnn.device
    // CHECK: emitc.call_opaque "ttnn::conv3d"
    %0 = "ttnn.conv3d"(%arg0, %arg1, %device)
      <{
        in_channels = 4: i32,
        out_channels = 16: i32,
        batch_size = 1: i32,
        input_depth = 8: i32,
        input_height = 28: i32,
        input_width = 28: i32,
        kernel_size = array<i32: 3, 3, 3>,
        stride = array<i32: 1, 1, 1>,
        padding = array<i32: 0, 0, 0>,
        padding_mode = "zeros",
        groups = 1: i32
      }> : (tensor<1x8x28x28x4xbf16>, tensor<1x1x108x16xbf16>, !ttnn.device) -> tensor<1x6x26x26x16xbf16>
    return %0 : tensor<1x6x26x26x16xbf16>
  }

  // Test Conv3d with bias EmitC conversion
  func.func @test_conv3d_with_bias_emitc(%arg0: tensor<1x8x28x28x4xbf16>, %arg1: tensor<1x1x108x16xbf16>, %arg2: tensor<1x1x1x32x16xbf16>) -> tensor<1x6x26x26x16xbf16> {
    %device = "ttnn.get_device"() <{mesh_shape = #ttnn<mesh_shape 1x1>}> : () -> !ttnn.device
    // CHECK: emitc.call_opaque "ttnn::conv3d"
    %0 = "ttnn.conv3d"(%arg0, %arg1, %arg2, %device)
      <{
        in_channels = 4: i32,
        out_channels = 16: i32,
        batch_size = 1: i32,
        input_depth = 8: i32,
        input_height = 28: i32,
        input_width = 28: i32,
        kernel_size = array<i32: 3, 3, 3>,
        stride = array<i32: 1, 1, 1>,
        padding = array<i32: 0, 0, 0>,
        padding_mode = "zeros",
        groups = 1: i32
      }> : (tensor<1x8x28x28x4xbf16>, tensor<1x1x108x16xbf16>, tensor<1x1x1x32x16xbf16>, !ttnn.device) -> tensor<1x6x26x26x16xbf16>
    return %0 : tensor<1x6x26x26x16xbf16>
  }

  // Test Conv3d with stride and padding EmitC conversion
  func.func @test_conv3d_stride_padding_emitc(%arg0: tensor<1x8x28x28x16xbf16>, %arg1: tensor<1x1x432x32xbf16>) -> tensor<1x4x14x14x32xbf16> {
    %device = "ttnn.get_device"() <{mesh_shape = #ttnn<mesh_shape 1x1>}> : () -> !ttnn.device
    // CHECK: emitc.call_opaque "ttnn::conv3d"
    %0 = "ttnn.conv3d"(%arg0, %arg1, %device)
      <{
        in_channels = 16: i32,
        out_channels = 32: i32,
        batch_size = 1: i32,
        input_depth = 8: i32,
        input_height = 28: i32,
        input_width = 28: i32,
        kernel_size = array<i32: 3, 3, 3>,
        stride = array<i32: 2, 2, 2>,
        padding = array<i32: 1, 1, 1>,
        padding_mode = "zeros",
        groups = 1: i32
      }> : (tensor<1x8x28x28x16xbf16>, tensor<1x1x432x32xbf16>, !ttnn.device) -> tensor<1x4x14x14x32xbf16>
    return %0 : tensor<1x4x14x14x32xbf16>
  }

  // Test Conv3d with replicate padding EmitC conversion
  func.func @test_conv3d_replicate_padding_emitc(%arg0: tensor<1x8x28x28x4xbf16>, %arg1: tensor<1x1x108x16xbf16>) -> tensor<1x8x28x28x16xbf16> {
    %device = "ttnn.get_device"() <{mesh_shape = #ttnn<mesh_shape 1x1>}> : () -> !ttnn.device
    // CHECK: emitc.call_opaque "ttnn::conv3d"
    %0 = "ttnn.conv3d"(%arg0, %arg1, %device)
      <{
        in_channels = 4: i32,
        out_channels = 16: i32,
        batch_size = 1: i32,
        input_depth = 8: i32,
        input_height = 28: i32,
        input_width = 28: i32,
        kernel_size = array<i32: 3, 3, 3>,
        stride = array<i32: 1, 1, 1>,
        padding = array<i32: 1, 1, 1>,
        padding_mode = "replicate",
        groups = 1: i32
      }> : (tensor<1x8x28x28x4xbf16>, tensor<1x1x108x16xbf16>, !ttnn.device) -> tensor<1x8x28x28x16xbf16>
    return %0 : tensor<1x8x28x28x16xbf16>
  }
}
