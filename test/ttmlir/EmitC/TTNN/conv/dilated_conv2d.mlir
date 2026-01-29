// TODO(dmilinkovic): re-enable CPU-hoisted const-eval once EmitC support for CPU-hoisted ops lands - issue #6100.
// RUN: ttmlir-opt --ttir-to-ttnn-backend-pipeline="enable-cpu-hoisted-const-eval=false system-desc-path=%system_desc_path%" -o %t.mlir %s
// RUN: ttmlir-translate --ttnn-to-flatbuffer -o %basename_t.ttnn %t.mlir
// RUN: ttmlir-opt --ttnn-to-emitc-device-pipeline -o %t2.mlir %t.mlir
// RUN: ttmlir-translate --mlir-to-cpp -o %basename_t.cpp %t2.mlir

module {
  func.func @dilated_even_conv2d_bf16(%arg0: tensor<16x32x32x64xbf16>, %arg1: tensor<64x64x3x3xbf16>, %arg2: tensor<1x1x1x64xbf16>) -> tensor<16x28x28x64xbf16> {
    %0 = "ttir.conv2d"(%arg0, %arg1, %arg2)
            <{
              stride = 1: i32,
              padding = 0: i32,
              dilation = 2: i32,
              groups = 1: i32
            }> : (tensor<16x32x32x64xbf16>, tensor<64x64x3x3xbf16>, tensor<1x1x1x64xbf16>) -> tensor<16x28x28x64xbf16>
    return %0 : tensor<16x28x28x64xbf16>
  }

  func.func @dilated_even_conv2d_f32(%arg0: tensor<16x32x32x64xf32>, %arg1: tensor<64x64x3x3xf32>, %arg2: tensor<1x1x1x64xf32>) -> tensor<16x28x28x64xf32> {
    %0 = "ttir.conv2d"(%arg0, %arg1, %arg2)
            <{
              stride = 1: i32,
              padding = 0: i32,
              dilation = 2: i32,
              groups = 1: i32
            }> : (tensor<16x32x32x64xf32>, tensor<64x64x3x3xf32>, tensor<1x1x1x64xf32>) -> tensor<16x28x28x64xf32>
    return %0 : tensor<16x28x28x64xf32>
  }

  func.func @dilated_uneven_conv2d_bf16(%arg0: tensor<16x32x32x64xbf16>, %arg1: tensor<64x64x3x3xbf16>, %arg2: tensor<1x1x1x64xbf16>) -> tensor<16x26x28x64xbf16> {
    %0 = "ttir.conv2d"(%arg0, %arg1, %arg2)
            <{
              stride = 1: i32,
              padding = 0: i32,
              dilation = array<i32: 3, 2>,
              groups = 1: i32
            }> : (tensor<16x32x32x64xbf16>, tensor<64x64x3x3xbf16>, tensor<1x1x1x64xbf16>) -> tensor<16x26x28x64xbf16>
    return %0 : tensor<16x26x28x64xbf16>
  }

  func.func @dilated_uneven_conv2d_f32(%arg0: tensor<16x32x32x64xbf16>, %arg1: tensor<64x64x3x3xbf16>, %arg2: tensor<1x1x1x64xbf16>) -> tensor<16x26x28x64xbf16> {
    %0 = "ttir.conv2d"(%arg0, %arg1, %arg2)
            <{
              stride = 1: i32,
              padding = 0: i32,
              dilation = array<i32: 3, 2>,
              groups = 1: i32
            }> : (tensor<16x32x32x64xbf16>, tensor<64x64x3x3xbf16>, tensor<1x1x1x64xbf16>) -> tensor<16x26x28x64xbf16>
    return %0 : tensor<16x26x28x64xbf16>
  }
}
