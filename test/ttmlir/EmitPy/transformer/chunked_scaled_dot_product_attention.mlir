// RUN: ttmlir-opt --ttcore-register-device --ttnn-layout --ttnn-workaround --convert-ttnn-to-emitpy -o %t.mlir %s
// RUN: ttmlir-translate --mlir-to-python %t.mlir | FileCheck %s

func.func @chunked_sdpa(%arg0: tensor<1x12x64x64xbf16>, %arg1: tensor<128x12x32x64xbf16>, %arg2: tensor<128x12x32x64xbf16>, %arg3: tensor<1x8xi32>, %arg4: tensor<1xi32>) -> tensor<1x12x64x64xbf16> {
  // CHECK: ttnn.transformer.chunked_scaled_dot_product_attention(
  %0 = "ttnn.chunked_scaled_dot_product_attention"(%arg0, %arg1, %arg2, %arg3, %arg4) <{scale = 1.250000e-01 : f32}> : (tensor<1x12x64x64xbf16>, tensor<128x12x32x64xbf16>, tensor<128x12x32x64xbf16>, tensor<1x8xi32>, tensor<1xi32>) -> tensor<1x12x64x64xbf16>
  return %0 : tensor<1x12x64x64xbf16>
}
