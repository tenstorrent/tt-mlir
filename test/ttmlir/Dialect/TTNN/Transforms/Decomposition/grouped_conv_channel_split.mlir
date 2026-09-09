// RUN: ttmlir-opt --ttcore-register-device --ttnn-layout --ttnn-decomposition -o %t %s
// RUN: FileCheck %s --input-file=%t

module attributes {} {
  // A depthwise conv2d (groups == in_channels == out_channels) too wide to fit
  // L1 is split into two 4096-channel convs concatenated on the channel dim.
  func.func @depthwise_conv2d_8192_channels(%arg0: tensor<1x1x64x8192xbf16>, %arg1: tensor<8192x1x3x3xbf16>) -> tensor<1x1x64x8192xbf16> {
    // CHECK-LABEL: func.func @depthwise_conv2d_8192_channels
    // Input is cut on the channel dim (last), weight on the output-channel dim (first).
    // CHECK: "ttnn.slice_static"
    // CHECK-SAME: ends = [1 : i32, 1 : i32, 64 : i32, 4096 : i32]
    // CHECK: "ttnn.slice_static"
    // CHECK-SAME: ends = [4096 : i32, 1 : i32, 3 : i32, 3 : i32]
    // CHECK: "ttnn.conv2d"
    // CHECK-SAME: groups = 4096 : i32
    // CHECK-SAME: in_channels = 4096 : i32
    // CHECK-SAME: out_channels = 4096 : i32
    // CHECK: "ttnn.conv2d"
    // CHECK-SAME: groups = 4096 : i32
    // CHECK-NOT: "ttnn.conv2d"
    // CHECK: "ttnn.concat"
    // CHECK-SAME: dim = 3 : si32

    %0 = "ttnn.get_device"() <{mesh_shape = #ttnn<mesh_shape 1x1>}> : () -> !ttnn.device
    %1 = "ttnn.conv2d"(%arg0, %arg1, %0)
          <{
            batch_size = 1 : i32,
            dilation = array<i32: 1, 1>,
            groups = 8192 : i32,
            in_channels = 8192 : i32,
            input_height = 8 : i32,
            input_width = 8 : i32,
            kernel_size = array<i32: 3, 3>,
            out_channels = 8192 : i32,
            padding = array<i32: 1, 1>,
            stride = array<i32: 1, 1>
          }> : (tensor<1x1x64x8192xbf16>, tensor<8192x1x3x3xbf16>, !ttnn.device) -> tensor<1x1x64x8192xbf16>
    return %1 : tensor<1x1x64x8192xbf16>
  }

  // A grouped (non-depthwise) conv2d splits on group boundaries too: 64 groups
  // of 128 channels become two convs of 32 groups each.
  func.func @grouped_conv2d_with_bias(%arg0: tensor<1x1x64x8192xbf16>, %arg1: tensor<8192x128x3x3xbf16>, %arg2: tensor<1x1x1x8192xbf16>) -> tensor<1x1x64x8192xbf16> {
    // CHECK-LABEL: func.func @grouped_conv2d_with_bias
    // Input, weight and bias are each cut over the same group range; the bias is
    // (1, 1, 1, O), so it follows the weight's output-channel range.
    // CHECK: "ttnn.slice_static"
    // CHECK-SAME: ends = [1 : i32, 1 : i32, 64 : i32, 4096 : i32]
    // CHECK: "ttnn.slice_static"
    // CHECK-SAME: ends = [4096 : i32, 128 : i32, 3 : i32, 3 : i32]
    // CHECK: "ttnn.slice_static"
    // CHECK-SAME: ends = [1 : i32, 1 : i32, 1 : i32, 4096 : i32]
    // CHECK: "ttnn.conv2d"
    // CHECK-SAME: groups = 32 : i32
    // CHECK-SAME: in_channels = 4096 : i32
    // CHECK-SAME: out_channels = 4096 : i32
    // CHECK: "ttnn.conv2d"
    // CHECK-SAME: groups = 32 : i32
    // CHECK-NOT: "ttnn.conv2d"
    // CHECK: "ttnn.concat"
    %0 = "ttnn.get_device"() <{mesh_shape = #ttnn<mesh_shape 1x1>}> : () -> !ttnn.device
    %1 = "ttnn.conv2d"(%arg0, %arg1, %arg2, %0)
          <{
            batch_size = 1 : i32,
            dilation = array<i32: 1, 1>,
            groups = 64 : i32,
            in_channels = 8192 : i32,
            input_height = 8 : i32,
            input_width = 8 : i32,
            kernel_size = array<i32: 3, 3>,
            out_channels = 8192 : i32,
            padding = array<i32: 1, 1>,
            stride = array<i32: 1, 1>
          }> : (tensor<1x1x64x8192xbf16>, tensor<8192x128x3x3xbf16>, tensor<1x1x1x8192xbf16>, !ttnn.device) -> tensor<1x1x64x8192xbf16>
    return %1 : tensor<1x1x64x8192xbf16>
  }

  // Depthwise conv1d takes the same path; channels are the last dim of its
  // (N, L, C) input and of its flattened (1, 1, N * L_out, O) result.
  func.func @depthwise_conv1d_8192_channels(%arg0: tensor<1x512x8192xbf16>, %arg1: tensor<8192x1x3xbf16>) -> tensor<1x1x512x8192xbf16> {
    // CHECK-LABEL: func.func @depthwise_conv1d_8192_channels
    // CHECK: "ttnn.slice_static"
    // CHECK-SAME: ends = [1 : i32, 512 : i32, 4096 : i32]
    // CHECK: "ttnn.conv1d"
    // CHECK-SAME: groups = 4096 : i32
    // CHECK-SAME: in_channels = 4096 : i32
    // CHECK: "ttnn.conv1d"
    // CHECK-SAME: groups = 4096 : i32
    // CHECK-NOT: "ttnn.conv1d"
    // CHECK: "ttnn.concat"
    %0 = "ttnn.get_device"() <{mesh_shape = #ttnn<mesh_shape 1x1>}> : () -> !ttnn.device
    %1 = "ttnn.conv1d"(%arg0, %arg1, %0)
          <{
            batch_size = 1 : i32,
            dilation = 1 : i32,
            groups = 8192 : i32,
            in_channels = 8192 : i32,
            input_length = 512 : i32,
            kernel_size = 3 : i32,
            out_channels = 8192 : i32,
            padding = array<i32: 1, 1>,
            stride = 1 : i32
          }> : (tensor<1x512x8192xbf16>, tensor<8192x1x3xbf16>, !ttnn.device) -> tensor<1x1x512x8192xbf16>
    return %1 : tensor<1x1x512x8192xbf16>
  }

  // groups == 1 is left alone: every output channel reads every input channel,
  // so there is no channel partition to cut along.
  func.func @dense_conv2d_is_not_split(%arg0: tensor<1x1x64x8192xbf16>, %arg1: tensor<8192x8192x1x1xbf16>) -> tensor<1x1x64x8192xbf16> {
    // CHECK-LABEL: func.func @dense_conv2d_is_not_split
    // CHECK-NOT: "ttnn.slice_static"
    // CHECK-NOT: "ttnn.concat"
    // CHECK: "ttnn.conv2d"
    // CHECK-SAME: in_channels = 8192 : i32
    %0 = "ttnn.get_device"() <{mesh_shape = #ttnn<mesh_shape 1x1>}> : () -> !ttnn.device
    %1 = "ttnn.conv2d"(%arg0, %arg1, %0)
          <{
            batch_size = 1 : i32,
            dilation = array<i32: 1, 1>,
            groups = 1 : i32,
            in_channels = 8192 : i32,
            input_height = 8 : i32,
            input_width = 8 : i32,
            kernel_size = array<i32: 1, 1>,
            out_channels = 8192 : i32,
            padding = array<i32: 0, 0>,
            stride = array<i32: 1, 1>
          }> : (tensor<1x1x64x8192xbf16>, tensor<8192x8192x1x1xbf16>, !ttnn.device) -> tensor<1x1x64x8192xbf16>
    return %1 : tensor<1x1x64x8192xbf16>
  }

  // A grouped conv already inside the channel budget is left alone.
  func.func @narrow_depthwise_conv2d_is_not_split(%arg0: tensor<1x1x64x512xbf16>, %arg1: tensor<512x1x3x3xbf16>) -> tensor<1x1x64x512xbf16> {
    // CHECK-LABEL: func.func @narrow_depthwise_conv2d_is_not_split
    // CHECK-NOT: "ttnn.slice_static"
    // CHECK-NOT: "ttnn.concat"
    // CHECK: "ttnn.conv2d"
    // CHECK-SAME: in_channels = 512 : i32
    %0 = "ttnn.get_device"() <{mesh_shape = #ttnn<mesh_shape 1x1>}> : () -> !ttnn.device
    %1 = "ttnn.conv2d"(%arg0, %arg1, %0)
          <{
            batch_size = 1 : i32,
            dilation = array<i32: 1, 1>,
            groups = 512 : i32,
            in_channels = 512 : i32,
            input_height = 8 : i32,
            input_width = 8 : i32,
            kernel_size = array<i32: 3, 3>,
            out_channels = 512 : i32,
            padding = array<i32: 1, 1>,
            stride = array<i32: 1, 1>
          }> : (tensor<1x1x64x512xbf16>, tensor<512x1x3x3xbf16>, !ttnn.device) -> tensor<1x1x64x512xbf16>
    return %1 : tensor<1x1x64x512xbf16>
  }
}
