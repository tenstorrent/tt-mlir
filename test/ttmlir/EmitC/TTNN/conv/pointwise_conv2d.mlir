// RUN: ttmlir-opt --ttir-to-ttnn-backend-pipeline="system-desc-path=%system_desc_path%" -o %t.mlir %s
// RUN: ttmlir-translate --ttnn-to-flatbuffer -o %basename_t.ttnn %t.mlir
// RUN: ttmlir-opt --ttnn-backend-to-emitc-pipeline -o %t2.mlir %t.mlir
// RUN: ttmlir-translate --mlir-to-cpp -o %basename_t.cpp %t2.mlir

module {
  func.func @pointwise_conv2d_bf16(%arg0: tensor<16x32x32x64xbf16>, %arg1: tensor<32x64x1x1xbf16>, %arg2: tensor<1x1x1x32xbf16>) -> tensor<16x32x32x32xbf16> {
    %0 = ttir.empty() : tensor<16x32x32x32xbf16>
    %1 = "ttir.conv2d"(%arg0, %arg1, %arg2, %0)
            <{
              stride = 1: i32,
              padding = 0: i32,
              dilation = 1: i32,
              groups = 1: i32
            }> : (tensor<16x32x32x64xbf16>, tensor<32x64x1x1xbf16>, tensor<1x1x1x32xbf16>, tensor<16x32x32x32xbf16>) -> tensor<16x32x32x32xbf16>
    return %1 : tensor<16x32x32x32xbf16>
  }

  func.func @pointwise_conv2d_1x1_f32(%arg0: tensor<16x32x32x64xf32>, %arg1: tensor<32x64x1x1xf32>, %arg2: tensor<1x1x1x32xf32>) -> tensor<16x32x32x32xf32> {
    %0 = ttir.empty() : tensor<16x32x32x32xf32>
    %1 = "ttir.conv2d"(%arg0, %arg1, %arg2, %0)
            <{
              stride = 1: i32,
              padding = 0: i32,
              dilation = 1: i32,
              groups = 1: i32
            }> : (tensor<16x32x32x64xf32>, tensor<32x64x1x1xf32>, tensor<1x1x1x32xf32>, tensor<16x32x32x32xf32>) -> tensor<16x32x32x32xf32>
    return %1 : tensor<16x32x32x32xf32>
  }
}
