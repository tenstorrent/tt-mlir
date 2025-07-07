// RUN: ttmlir-opt --ttir-to-ttnn-backend-pipeline="system-desc-path=%system_desc_path%" %s > %t.mlir
// RUN: ttmlir-translate --ttnn-to-flatbuffer %t.mlir > %t.ttnn

module {
  func.func @conv2d_bf16(%arg0: tensor<16x32x32x64xbf16>, %arg1: tensor<64x64x3x3xbf16>, %arg2: tensor<1x1x1x64xbf16>) -> tensor<16x30x30x64xbf16> {
    %0 = ttir.empty() : tensor<16x30x30x64xbf16>
    %1 = "ttir.conv2d"(%arg0, %arg1, %arg2, %0)
            <{
              stride = 1: i32,
              padding = 0: i32,
              dilation = 1: i32,
              groups = 1: i32
            }> : (tensor<16x32x32x64xbf16>, tensor<64x64x3x3xbf16>, tensor<1x1x1x64xbf16>, tensor<16x30x30x64xbf16>) -> tensor<16x30x30x64xbf16>
    return %1 : tensor<16x30x30x64xbf16>
  }

  // func.func @conv2d_f32(%arg0: tensor<16x32x32x64xf32>, %arg1: tensor<64x64x3x3xf32>, %arg2: tensor<1x1x1x64xf32>) -> tensor<16x30x30x64xf32> {
  //   %0 = ttir.empty() : tensor<16x30x30x64xf32>
  //   %1 = "ttir.conv2d"(%arg0, %arg1, %arg2, %0)
  //           <{
  //             stride = 1: i32,
  //             padding = 0: i32,
  //             dilation = 1: i32,
  //             groups = 1: i32
  //           }> : (tensor<16x32x32x64xf32>, tensor<64x64x3x3xf32>, tensor<1x1x1x64xf32>, tensor<16x30x30x64xf32>) -> tensor<16x30x30x64xf32>
  //   return %1 : tensor<16x30x30x64xf32>
  // }

  // func.func @conv2d_asymmetric_padding_bf16(%arg0: tensor<1x32x32x64xbf16>, %arg1: tensor<128x64x3x3xbf16>, %arg2: tensor<1x1x1x128xbf16>) -> tensor<1x33x37x128xbf16> {
  //   %0 = ttir.empty() : tensor<1x33x37x128xbf16>
  //   %1 = "ttir.conv2d"(%arg0, %arg1, %arg2, %0)
  //           <{
  //             stride = 1: i32,
  //             padding = array<i32: 1, 3, 2, 4>,
  //             dilation = 1: i32,
  //             groups = 1: i32
  //           }> : (tensor<1x32x32x64xbf16>, tensor<128x64x3x3xbf16>, tensor<1x1x1x128xbf16>, tensor<1x33x37x128xbf16>) -> tensor<1x33x37x128xbf16>
  //   return %1 : tensor<1x33x37x128xbf16>
  // }

  // func.func @conv2d_asymmetric_padding_f32(%arg0: tensor<1x16x16x32xf32>, %arg1: tensor<64x32x5x5xf32>, %arg2: tensor<1x1x1x64xf32>) -> tensor<1x15x21x64xf32> {
  //   %0 = ttir.empty() : tensor<1x15x21x64xf32>
  //   %1 = "ttir.conv2d"(%arg0, %arg1, %arg2, %0)
  //           <{
  //             stride = 1: i32,
  //             padding = array<i32: 2, 4, 1, 5>,
  //             dilation = 1: i32,
  //             groups = 1: i32
  //           }> : (tensor<1x16x16x32xf32>, tensor<64x32x5x5xf32>, tensor<1x1x1x64xf32>, tensor<1x15x21x64xf32>) -> tensor<1x15x21x64xf32>
  //   return %1 : tensor<1x15x21x64xf32>
  // }
}
