// RUN: not ttmlir-opt --ttir-load-system-desc --ttir-layout --convert-ttir-to-ttnn %s 2>&1 | FileCheck %s
// CHECK: error: 'ttir.softmax' op Dimension attribute must be within the bounds of the input tensor
module attributes {} {
  func.func @forward(%arg0: tensor<512x1024xbf16>) -> tensor<512x1024xbf16> {
    %0 = tensor.empty() : tensor<512x1024xbf16>
    %1 = "ttir.softmax"(%arg0, %0) <{dimension = 2 : si32}> : (tensor<512x1024xbf16>, tensor<512x1024xbf16>) -> tensor<512x1024xbf16>
    return %1 : tensor<512x1024xbf16>
  }
}
