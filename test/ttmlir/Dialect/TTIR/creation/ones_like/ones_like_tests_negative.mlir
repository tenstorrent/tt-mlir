// RUN: not ttmlir-opt %s 2>&1 | FileCheck %s
module {
  func.func @forward(%arg0: tensor<1x4x100x100xbf16>) -> tensor<1x3x100x100xbf16> {
    %0 = ttir.empty() : tensor<1x3x100x100xbf16>
    // CHECK: error: 'ttir.ones_like' op requires the same type for all operands and results
    %1 = "ttir.ones_like"(%arg0, %0) : (tensor<1x4x100x100xbf16>, tensor<1x3x100x100xbf16>) -> tensor<1x3x100x100xbf16>
    return %1 : tensor<1x3x100x100xbf16>
  }
}
