// RUN: not ttmlir-opt --split-input-file %s 2>&1 | FileCheck %s
// Negative test for clamp operation

// Verify that the parsing fails if input and output shapes do not match.
module attributes {} {
  func.func @clamp(%arg0: tensor<64x64xbf16>) -> tensor<64x128xbf16> {
    %0 = tensor.empty() : tensor<64x128xbf16>
    // CHECK: error: 'ttir.clamp' op input and output must have same shape.
    %1 = "ttir.clamp"(%arg0, %0) <{max = 3.000000e+00 : f32, min = 2.000000e+00 : f32}> : (tensor<64x64xbf16>, tensor<64x128xbf16>) -> tensor<64x128xbf16>
    return %1 : tensor<64x128xbf16>
  }
}
