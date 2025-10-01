// RUN: ttmlir-opt --convert-ttir-to-linalg -o %t %s
// RUN: FileCheck %s --input-file=%t

module {
  func.func @conv2d_simple(%arg0: tensor<1x32x32x64xbf16>, %arg1: tensor<64x64x3x3xbf16>, %arg2: tensor<1x1x1x64xbf16>) -> tensor<1x30x30x64xbf16> {
    %0 = ttir.empty() : tensor<1x30x30x64xbf16>
    // CHECK: tosa.transpose
    // CHECK: tosa.reshape
    // CHECK: tosa.conv2d
    // CHECK-NOT: tensor.extract_slice
    %1 = "ttir.conv2d"(%arg0, %arg1, %arg2, %0)
            <{
              stride = 1: i32,
              padding = 0: i32,
              dilation = 1: i32,
              groups = 1: i32
            }> : (tensor<1x32x32x64xbf16>, tensor<64x64x3x3xbf16>, tensor<1x1x1x64xbf16>, tensor<1x30x30x64xbf16>) -> tensor<1x30x30x64xbf16>
    return %1 : tensor<1x30x30x64xbf16>
  }

  func.func @conv2d_stride_1(%arg0: tensor<3x32x32x8xbf16>, %arg1: tensor<16x8x3x3xbf16>, %arg2: tensor<1x1x1x16xbf16>) -> tensor<3x15x15x16xbf16> {
    %0 = ttir.empty() : tensor<3x15x15x16xbf16>
    // CHECK: tosa.transpose
    // CHECK: tosa.reshape
    // CHECK: tosa.conv2d
    // CHECK: tensor.extract_slice
    %1 = "ttir.conv2d"(%arg0, %arg1, %arg2, %0)
            <{
              stride = 2: i32,
              padding = 0: i32,
              dilation = 1: i32,
              groups = 1: i32
            }> : (tensor<3x32x32x8xbf16>, tensor<16x8x3x3xbf16>, tensor<1x1x1x16xbf16>, tensor<3x15x15x16xbf16>) -> tensor<3x15x15x16xbf16>
    return %1 : tensor<3x15x15x16xbf16>
  }

  func.func @conv2d_stride_2(%arg0: tensor<4x32x32x16xbf16>, %arg1: tensor<8x16x3x3xbf16>, %arg2: tensor<1x1x1x8xbf16>) -> tensor<4x8x5x8xbf16> {
    %0 = ttir.empty() : tensor<4x8x5x8xbf16>
    // CHECK: tosa.transpose
    // CHECK: tosa.reshape
    // CHECK: tosa.conv2d
    // CHECK: tensor.extract_slice
    %1 = "ttir.conv2d"(%arg0, %arg1, %arg2, %0)
            <{
              stride = array<i32: 4, 6>,
              padding = 0: i32,
              dilation = 1: i32,
              groups = 1: i32
            }> : (tensor<4x32x32x16xbf16>, tensor<8x16x3x3xbf16>, tensor<1x1x1x8xbf16>, tensor<4x8x5x8xbf16>) -> tensor<4x8x5x8xbf16>
    return %1 : tensor<4x8x5x8xbf16>
  }

  func.func @conv2d_padding_1(%arg0: tensor<32x32x32x4xbf16>, %arg1: tensor<8x4x3x3xbf16>, %arg2: tensor<1x1x1x8xbf16>) -> tensor<32x38x38x8xbf16> {
    %0 = ttir.empty() : tensor<32x38x38x8xbf16>
    // CHECK: tosa.transpose
    // CHECK: tosa.reshape
    // CHECK: tosa.conv2d
    // CHECK-NOT: tensor.extract_slice
    %1 = "ttir.conv2d"(%arg0, %arg1, %arg2, %0)
            <{
              stride = 1: i32,
              padding = 4: i32,
              dilation = 1: i32,
              groups = 1: i32
            }> : (tensor<32x32x32x4xbf16>, tensor<8x4x3x3xbf16>, tensor<1x1x1x8xbf16>, tensor<32x38x38x8xbf16>) -> tensor<32x38x38x8xbf16>
    return %1 : tensor<32x38x38x8xbf16>
  }

  func.func @conv2d_padding_2(%arg0: tensor<16x32x32x32xbf16>, %arg1: tensor<128x32x3x3xbf16>, %arg2: tensor<1x1x1x128xbf16>) -> tensor<16x54x46x128xbf16> {
    %0 = ttir.empty() : tensor<16x54x46x128xbf16>
    // CHECK: tosa.transpose
    // CHECK: tosa.reshape
    // CHECK: tosa.conv2d
    // CHECK-NOT: tensor.extract_slice
    %1 = "ttir.conv2d"(%arg0, %arg1, %arg2, %0)
            <{
              stride = 1: i32,
              padding = array<i32: 12, 8>,
              dilation = 1: i32,
              groups = 1: i32
            }> : (tensor<16x32x32x32xbf16>, tensor<128x32x3x3xbf16>, tensor<1x1x1x128xbf16>, tensor<16x54x46x128xbf16>) -> tensor<16x54x46x128xbf16>
    return %1 : tensor<16x54x46x128xbf16>
  }

  func.func @conv2d_padding_3(%arg0: tensor<8x32x32x64xbf16>, %arg1: tensor<256x64x3x3xbf16>, %arg2: tensor<1x1x1x256xbf16>) -> tensor<8x48x42x256xbf16> {
    %0 = ttir.empty() : tensor<8x48x42x256xbf16>
    // CHECK: tosa.transpose
    // CHECK: tosa.reshape
    // CHECK: tosa.conv2d
    // CHECK-NOT: tensor.extract_slice
    %1 = "ttir.conv2d"(%arg0, %arg1, %arg2, %0)
            <{
              stride = 1: i32,
              padding = array<i32: 12, 8, 6, 4>,
              dilation = 1: i32,
              groups = 1: i32
            }> : (tensor<8x32x32x64xbf16>, tensor<256x64x3x3xbf16>, tensor<1x1x1x256xbf16>, tensor<8x48x42x256xbf16>) -> tensor<8x48x42x256xbf16>
    return %1 : tensor<8x48x42x256xbf16>
  }

  func.func @conv2d_dilation_1(%arg0: tensor<16x32x32x128xbf16>, %arg1: tensor<64x128x3x3xbf16>, %arg2: tensor<1x1x1x64xbf16>) -> tensor<16x24x24x64xbf16> {
    %0 = ttir.empty() : tensor<16x24x24x64xbf16>
    // CHECK: tosa.transpose
    // CHECK: tosa.reshape
    // CHECK: tosa.conv2d
    // CHECK-NOT: tensor.extract_slice
    %1 = "ttir.conv2d"(%arg0, %arg1, %arg2, %0)
            <{
              stride = 1: i32,
              padding = 0: i32,
              dilation = 4: i32,
              groups = 1: i32
            }> : (tensor<16x32x32x128xbf16>, tensor<64x128x3x3xbf16>, tensor<1x1x1x64xbf16>, tensor<16x24x24x64xbf16>) -> tensor<16x24x24x64xbf16>
    return %1 : tensor<16x24x24x64xbf16>
  }

  func.func @conv2d_dilation_2(%arg0: tensor<32x32x32x16xbf16>, %arg1: tensor<64x16x3x3xbf16>, %arg2: tensor<1x1x1x64xbf16>) -> tensor<32x20x28x64xbf16> {
    %0 = ttir.empty() : tensor<32x20x28x64xbf16>
    // CHECK: tosa.transpose
    // CHECK: tosa.reshape
    // CHECK: tosa.conv2d
    // CHECK-NOT: tensor.extract_slice
    %1 = "ttir.conv2d"(%arg0, %arg1, %arg2, %0)
            <{
              stride = 1: i32,
              padding = 0: i32,
              dilation = array<i32: 6, 2>,
              groups = 1: i32
            }> : (tensor<32x32x32x16xbf16>, tensor<64x16x3x3xbf16>, tensor<1x1x1x64xbf16>, tensor<32x20x28x64xbf16>) -> tensor<32x20x28x64xbf16>
    return %1 : tensor<32x20x28x64xbf16>
  }



}
