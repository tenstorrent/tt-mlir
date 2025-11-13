// RUN: ttmlir-opt --ttir-to-ttnn-backend-pipeline="system-desc-path=%system_desc_path%" -o %t.mlir %s
// RUN: ttmlir-translate --ttnn-to-flatbuffer -o %t.ttnn %t.mlir

module {
  func.func @conv3d_simple(%arg0: tensor<1x8x28x28x32xbf16>, %arg1: tensor<32x32x3x3x3xbf16>) -> tensor<1x6x26x26x32xbf16> {
    %0 = ttir.empty() : tensor<1x6x26x26x32xbf16>
    %1 = "ttir.conv3d"(%arg0, %arg1, %0)
            <{
              stride = array<i32: 1, 1, 1>,
              padding = array<i32: 0, 0, 0>,
              padding_mode = "zeros",
              groups = 1: i32
            }> : (tensor<1x8x28x28x32xbf16>, tensor<32x32x3x3x3xbf16>, tensor<1x6x26x26x32xbf16>) -> tensor<1x6x26x26x32xbf16>
    return %1 : tensor<1x6x26x26x32xbf16>
  }

  func.func @conv3d_with_bias(%arg0: tensor<1x8x28x28x32xbf16>, %arg1: tensor<32x32x3x3x3xbf16>, %arg2: tensor<1x1x1x1x32xbf16>) -> tensor<1x6x26x26x32xbf16> {
    %0 = ttir.empty() : tensor<1x6x26x26x32xbf16>
    %1 = "ttir.conv3d"(%arg0, %arg1, %arg2, %0)
            <{
              stride = array<i32: 1, 1, 1>,
              padding = array<i32: 0, 0, 0>,
              padding_mode = "zeros",
              groups = 1: i32
            }> : (tensor<1x8x28x28x32xbf16>, tensor<32x32x3x3x3xbf16>, tensor<1x1x1x1x32xbf16>, tensor<1x6x26x26x32xbf16>) -> tensor<1x6x26x26x32xbf16>
    return %1 : tensor<1x6x26x26x32xbf16>
  }

  func.func @conv3d_with_stride(%arg0: tensor<1x8x28x28x32xbf16>, %arg1: tensor<64x32x3x3x3xbf16>) -> tensor<1x3x13x13x64xbf16> {
    %0 = ttir.empty() : tensor<1x3x13x13x64xbf16>
    %1 = "ttir.conv3d"(%arg0, %arg1, %0)
            <{
              stride = array<i32: 2, 2, 2>,
              padding = array<i32: 0, 0, 0>,
              padding_mode = "zeros",
              groups = 1: i32
            }> : (tensor<1x8x28x28x32xbf16>, tensor<64x32x3x3x3xbf16>, tensor<1x3x13x13x64xbf16>) -> tensor<1x3x13x13x64xbf16>
    return %1 : tensor<1x3x13x13x64xbf16>
  }

}
