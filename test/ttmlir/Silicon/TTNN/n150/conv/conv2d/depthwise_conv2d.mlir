// RUN: ttmlir-opt --ttir-to-ttnn-backend-pipeline="system-desc-path=%system_desc_path%" %s > %t.mlir
// RUN: FileCheck %s --input-file=%t.mlir
// RUN: ttmlir-translate --ttnn-to-flatbuffer %t.mlir > %t.ttnn

module {
  func.func @depthwise_conv2d_bf16(%arg0: tensor<32x64x64x3xbf16>, %arg1: tensor<3x1x3x3xbf16>, %arg2: tensor<1x1x1x3xbf16>) -> tensor<32x64x64x3xbf16> {
    %0 = ttir.empty() : tensor<32x64x64x3xbf16>
    // CHECK: %[[C:.*]] = "ttnn.conv2d"[[C:.*]]
    %1 = "ttir.conv2d"(%arg0, %arg1, %arg2, %0)
            <{
              stride = 1: i32,
              padding = 1: i32,
              dilation = 1: i32,
              groups = 3: i32
            }> : (tensor<32x64x64x3xbf16>, tensor<3x1x3x3xbf16>, tensor<1x1x1x3xbf16>, tensor<32x64x64x3xbf16>) -> tensor<32x64x64x3xbf16>
    return %1 : tensor<32x64x64x3xbf16>
  }

  func.func @depthwise_conv2d_f32(%arg0: tensor<32x64x64x3xf32>, %arg1: tensor<3x1x3x3xf32>, %arg2: tensor<1x1x1x3xf32>) -> tensor<32x64x64x3xf32> {
    %0 = ttir.empty() : tensor<32x64x64x3xf32>
    // CHECK: %[[C:.*]] = "ttnn.conv2d"[[C:.*]]
    %1 = "ttir.conv2d"(%arg0, %arg1, %arg2, %0)
            <{
              stride = 1: i32,
              padding = 1: i32,
              dilation = 1: i32,
              groups = 3: i32
            }> : (tensor<32x64x64x3xf32>, tensor<3x1x3x3xf32>, tensor<1x1x1x3xf32>, tensor<32x64x64x3xf32>) -> tensor<32x64x64x3xf32>
    return %1 : tensor<32x64x64x3xf32>
  }
}
