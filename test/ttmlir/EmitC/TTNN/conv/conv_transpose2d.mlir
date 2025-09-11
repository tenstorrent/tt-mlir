// RUN: ttmlir-opt --ttir-to-ttnn-backend-pipeline="system-desc-path=%system_desc_path%" -o %t.mlir %s
// RUN: ttmlir-translate --ttnn-to-flatbuffer -o %basename_t.ttnn %t.mlir
// RUN: ttmlir-opt --ttnn-backend-to-emitc-pipeline -o %t2.mlir %t.mlir
// RUN: ttmlir-translate --mlir-to-cpp -o %basename_t.cpp %t2.mlir

module {
  func.func @conv_transpose2d_bf16(%arg0: tensor<3x8x8x256xbf16>, %arg1: tensor<256x256x3x3xbf16>, %arg2: tensor<1x1x1x256xbf16>) -> tensor<3x10x10x256xbf16> {
    %0 = ttir.empty() : tensor<3x10x10x256xbf16>
    %1 = "ttir.conv_transpose2d"(%arg0, %arg1, %arg2, %0)
            <{
              stride = 1: i32,
              padding = 0: i32,
              output_padding = 0: i32,
              dilation = 1: i32,
              groups = 1: i32}
            > : (tensor<3x8x8x256xbf16>, tensor<256x256x3x3xbf16>, tensor<1x1x1x256xbf16>, tensor<3x10x10x256xbf16>) -> tensor<3x10x10x256xbf16>
    return %1 : tensor<3x10x10x256xbf16>
  }

  func.func @conv_transpose2d_f32(%arg0: tensor<3x8x8x256xf32>, %arg1: tensor<256x256x3x3xf32>, %arg2: tensor<1x1x1x256xf32>) -> tensor<3x10x10x256xf32> {
    %0 = ttir.empty() : tensor<3x10x10x256xf32>
    %1 = "ttir.conv_transpose2d"(%arg0, %arg1, %arg2, %0)
            <{
              stride = 1: i32,
              padding = 0: i32,
              output_padding = 0: i32,
              dilation = 1: i32,
              groups = 1: i32}
            > : (tensor<3x8x8x256xf32>, tensor<256x256x3x3xf32>, tensor<1x1x1x256xf32>, tensor<3x10x10x256xf32>) -> tensor<3x10x10x256xf32>
    return %1 : tensor<3x10x10x256xf32>
  }
}
