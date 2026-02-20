// RUN: ttmlir-opt --ttir-to-ttnn-backend-pipeline -o %t %s
// RUN: FileCheck %s --input-file=%t
module {
  // Test simple 3D convolution
  func.func @conv3d_simple(%arg0: tensor<1x8x28x28x4xbf16>, %arg1: tensor<16x4x3x3x3xbf16>) -> tensor<1x6x26x26x16xbf16> {
    // CHECK: "ttnn.reshape"
    // CHECK: "ttnn.conv3d"
    %0 = "ttir.conv3d"(%arg0, %arg1)
            <{
              stride = array<i32: 1, 1, 1>,
              padding = array<i32: 0, 0, 0>,
              groups = 1 : i32,
              padding_mode = "zeros"
            }> : (tensor<1x8x28x28x4xbf16>, tensor<16x4x3x3x3xbf16>) -> tensor<1x6x26x26x16xbf16>
    return %0 : tensor<1x6x26x26x16xbf16>
  }

  // Test 3D convolution with stride
  func.func @conv3d_with_stride(%arg0: tensor<1x8x28x28x16xbf16>, %arg1: tensor<32x16x3x3x3xbf16>) -> tensor<1x4x14x14x32xbf16> {
    // CHECK: "ttnn.reshape"
    // CHECK: "ttnn.conv3d"
    %0 = "ttir.conv3d"(%arg0, %arg1)
            <{
              stride = array<i32: 2, 2, 2>,
              padding = array<i32: 1, 1, 1>,
              groups = 1 : i32,
              padding_mode = "zeros"
            }> : (tensor<1x8x28x28x16xbf16>, tensor<32x16x3x3x3xbf16>) -> tensor<1x4x14x14x32xbf16>
    return %0 : tensor<1x4x14x14x32xbf16>
  }

  // Test 3D convolution with bias
  func.func @conv3d_with_bias(%arg0: tensor<1x8x28x28x4xbf16>, %arg1: tensor<16x4x3x3x3xbf16>, %arg2: tensor<1x1x1x1x16xbf16>) -> tensor<1x6x26x26x16xbf16> {
    // CHECK: "ttnn.reshape"
    // CHECK: "ttnn.conv3d"
    %0 = "ttir.conv3d"(%arg0, %arg1, %arg2)
            <{
              stride = array<i32: 1, 1, 1>,
              padding = array<i32: 0, 0, 0>,
              groups = 1 : i32,
              padding_mode = "zeros"
            }> : (tensor<1x8x28x28x4xbf16>, tensor<16x4x3x3x3xbf16>, tensor<1x1x1x1x16xbf16>) -> tensor<1x6x26x26x16xbf16>
    return %0 : tensor<1x6x26x26x16xbf16>
  }

  // Test 3D convolution with different padding mode
  func.func @conv3d_padding_replicate(%arg0: tensor<1x8x28x28x4xbf16>, %arg1: tensor<16x4x3x3x3xbf16>) -> tensor<1x8x28x28x16xbf16> {
    // CHECK: "ttnn.reshape"
    // CHECK: "ttnn.conv3d"
    // CHECK-SAME: padding_mode = "replicate"
    %0 = "ttir.conv3d"(%arg0, %arg1)
            <{
              stride = array<i32: 1, 1, 1>,
              padding = array<i32: 1, 1, 1>,
              groups = 1 : i32,
              padding_mode = "replicate"
            }> : (tensor<1x8x28x28x4xbf16>, tensor<16x4x3x3x3xbf16>) -> tensor<1x8x28x28x16xbf16>
    return %0 : tensor<1x8x28x28x16xbf16>
  }

  // Test 3D convolution with larger kernel
  func.func @conv3d_large_kernel(%arg0: tensor<1x16x32x32x8xbf16>, %arg1: tensor<32x8x5x5x5xbf16>) -> tensor<1x12x28x28x32xbf16> {
    // CHECK: "ttnn.reshape"
    // CHECK: "ttnn.conv3d"
    %0 = "ttir.conv3d"(%arg0, %arg1)
            <{
              stride = array<i32: 1, 1, 1>,
              padding = array<i32: 0, 0, 0>,
              groups = 1 : i32,
              padding_mode = "zeros"
            }> : (tensor<1x16x32x32x8xbf16>, tensor<32x8x5x5x5xbf16>) -> tensor<1x12x28x28x32xbf16>
    return %0 : tensor<1x12x28x28x32xbf16>
  }
  // Test 3D convolution with different layout (NCDHW -> NDHWC conversion)
  func.func @conv3d_layout_conversion(%arg0: tensor<1x4x8x28x28xbf16>, %arg1: tensor<16x4x3x3x3xbf16>) -> tensor<1x16x6x26x26xbf16> {
    // CHECK: "ttnn.permute"
    // CHECK: "ttnn.conv3d"
    // CHECK: "ttnn.permute"
    %0 = "ttir.conv3d"(%arg0, %arg1)
            <{
              stride = array<i32: 1, 1, 1>,
              padding = array<i32: 0, 0, 0>,
              groups = 1 : i32,
              batch_dim = 0 : i64,
              channel_dim = 1 : i64,
              depth_dim = 2 : i64,
              height_dim = 3 : i64,
              width_dim = 4 : i64,
              padding_mode = "zeros"
            }> : (tensor<1x4x8x28x28xbf16>, tensor<16x4x3x3x3xbf16>) -> tensor<1x16x6x26x26xbf16>
    return %0 : tensor<1x16x6x26x26xbf16>
  }

  // Test 3D convolution with non-tile-aligned output channels (NCDHW layout)
  func.func @conv3d_non_tile_aligned_out_channels(%arg0: tensor<1x128x11x34x34xbf16>, %arg1: tensor<48x128x3x3x3xbf16>) -> tensor<1x48x9x32x32xbf16> {
    // CHECK: "ttnn.permute"
    // CHECK: "ttnn.pad"
    // CHECK: "ttnn.conv3d"
    // CHECK: "ttnn.slice_static"
    // CHECK: "ttnn.permute"
    %0 = "ttir.conv3d"(%arg0, %arg1)
            <{
              stride = array<i32: 1, 1, 1>,
              padding = array<i32: 0, 0, 0>,
              groups = 1 : i32,
              batch_dim = 0 : i64,
              channel_dim = 1 : i64,
              depth_dim = 2 : i64,
              height_dim = 3 : i64,
              width_dim = 4 : i64,
              padding_mode = "zeros"
            }> : (tensor<1x128x11x34x34xbf16>, tensor<48x128x3x3x3xbf16>) -> tensor<1x48x9x32x32xbf16>
    return %0 : tensor<1x48x9x32x32xbf16>
  }

  // Test 3D convolution with depth padding workaround (NCDHW layout).
  func.func @conv3d_depth_padding_workaround(%arg0: tensor<1x128x2x4x4xbf16>, %arg1: tensor<1024x128x3x3x3xbf16>) -> tensor<1x1024x2x4x4xbf16> {
    // CHECK: "ttnn.permute"
    // CHECK: "ttnn.permute"
    // CHECK: "ttnn.pad"
    // CHECK: "ttnn.permute"
    // CHECK: "ttnn.conv3d"
    // CHECK: "ttnn.permute"
    %0 = "ttir.conv3d"(%arg0, %arg1)
            <{
              stride = array<i32: 1, 1, 1>,
              padding = array<i32: 1, 1, 1>,
              groups = 1 : i32,
              batch_dim = 0 : i64,
              channel_dim = 1 : i64,
              depth_dim = 2 : i64,
              height_dim = 3 : i64,
              width_dim = 4 : i64,
              padding_mode = "zeros"
            }> : (tensor<1x128x2x4x4xbf16>, tensor<1024x128x3x3x3xbf16>) -> tensor<1x1024x2x4x4xbf16>
    return %0 : tensor<1x1024x2x4x4xbf16>
  }

}
