// RUN: ttmlir-opt --ttir-to-ttnn-backend-pipeline="system-desc-path=%system_desc_path%" -o %t.mlir %s
// RUN: ttmlir-translate --ttnn-to-flatbuffer -o %t.ttnn %t.mlir
module {
  func.func @linear(%arg0: tensor<2x34x1024xf32>, %arg1: tensor<1024x1024xf32>, %bias: tensor<2x34x1024xf32>) -> tensor<2x34x1024xf32> {
    %0 = ttir.empty() : tensor<2x34x1024xf32>
    %1 = "ttir.linear"(%arg0, %arg1, %bias, %0) : (tensor<2x34x1024xf32>, tensor<1024x1024xf32>, tensor<2x34x1024xf32>, tensor<2x34x1024xf32>) -> tensor<2x34x1024xf32>
    return %1 : tensor<2x34x1024xf32>
  }
  func.func @linear_with_implicit_broadcast(%arg0: tensor<2x34x1024xf32>, %arg1: tensor<1024x1024xf32>, %bias: tensor<1024xf32>) -> tensor<2x34x1024xf32> {
    %0 = ttir.empty() : tensor<2x34x1024xf32>
    %1 = "ttir.linear"(%arg0, %arg1, %bias, %0) : (tensor<2x34x1024xf32>, tensor<1024x1024xf32>, tensor<1024xf32>, tensor<2x34x1024xf32>) -> tensor<2x34x1024xf32>
    return %1 : tensor<2x34x1024xf32>
  }
}
