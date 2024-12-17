// RUN: not ttmlir-opt --split-input-file %s 2>&1 | FileCheck %s
// Verify that test fails if dim is out of bounds

module {
func.func @neg_dim_six(%arg0: tensor<4x2x32x32xbf16>) -> tensor<1x4x2x32x32xbf16> {
  // CHECK: error: 'ttir.unsqueeze' op Dimension attribute must be within the bounds of the input tensor
  %0 = tensor.empty() : tensor<1x4x2x32x32xbf16>
  %1 = "ttir.unsqueeze"(%arg0, %0) <{dim = -6 : si32}> : (tensor<4x2x32x32xbf16>, tensor<1x4x2x32x32xbf16>) -> tensor<1x4x2x32x32xbf16>
  return %1 : tensor<1x4x2x32x32xbf16>
}
}

// -----
module {
func.func @pos_dim_five(%arg0: tensor<4x2x32x32xbf16>) -> tensor<1x4x2x32x32xbf16> {
  // CHECK: error: 'ttir.unsqueeze' op Dimension attribute must be within the bounds of the input tensor
  %0 = tensor.empty() : tensor<1x4x2x32x32xbf16>
  %1 = "ttir.unsqueeze"(%arg0, %0) <{dim = 5 : si32}> : (tensor<4x2x32x32xbf16>, tensor<1x4x2x32x32xbf16>) -> tensor<1x4x2x32x32xbf16>
  return %1 : tensor<1x4x2x32x32xbf16>
}
}
