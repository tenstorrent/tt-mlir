// RUN: ttmlir-opt --ttir-to-ttnn-backend-pipeline -o %t %s
// RUN: FileCheck %s --input-file=%t
module {
  func.func @conv1d_simple(%arg0: tensor<1x32x64xbf16>, %arg1: tensor<64x64x3xbf16>, %arg2: tensor<1x1x64xbf16>) -> tensor<1x30x64xbf16> {
    // CHECK: "ttnn.conv1d"
    %0 = "ttir.conv1d"(%arg0, %arg1, %arg2)
            <{
              stride = 1: i32,
              padding = 0: i32,
              dilation = 1: i32,
              groups = 1: i32
            }> : (tensor<1x32x64xbf16>, tensor<64x64x3xbf16>, tensor<1x1x64xbf16>) -> tensor<1x30x64xbf16>
    return %0 : tensor<1x30x64xbf16>
  }

  func.func @conv1d_no_bias(%arg0: tensor<1x32x64xbf16>, %arg1: tensor<64x64x3xbf16>) -> tensor<1x32x64xbf16> {
    // CHECK: "ttnn.conv1d"
    %0 = "ttir.conv1d"(%arg0, %arg1)
            <{
              stride = 1: i32,
              padding = 1: i32,
              dilation = 1: i32,
              groups = 1: i32
            }> : (tensor<1x32x64xbf16>, tensor<64x64x3xbf16>) -> tensor<1x32x64xbf16>
    return %0 : tensor<1x32x64xbf16>
  }

  func.func @conv1d_stride(%arg0: tensor<3x32x8xbf16>, %arg1: tensor<16x8x3xbf16>, %arg2: tensor<1x1x16xbf16>) -> tensor<3x15x16xbf16> {
    // CHECK: "ttnn.conv1d"
    %0 = "ttir.conv1d"(%arg0, %arg1, %arg2)
            <{
              stride = 2: i32,
              padding = 0: i32,
              dilation = 1: i32,
              groups = 1: i32
            }> : (tensor<3x32x8xbf16>, tensor<16x8x3xbf16>, tensor<1x1x16xbf16>) -> tensor<3x15x16xbf16>
    return %0 : tensor<3x15x16xbf16>
  }
}
