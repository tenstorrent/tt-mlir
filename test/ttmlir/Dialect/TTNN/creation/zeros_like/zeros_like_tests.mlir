// RUN: ttmlir-opt --ttir-to-ttnn-backend-pipeline -o %t %s
// RUN: FileCheck %s --input-file=%t
module {
  func.func @forward(%arg0: tensor<1x3x100x100xbf16>) -> tensor<1x3x100x100xbf16> {
    %0 = ttir.empty() : tensor<1x3x100x100xbf16>
    // CHECK: "ttnn.zeros_like"
    %1 = "ttir.zeros_like"(%arg0, %0) : (tensor<1x3x100x100xbf16>, tensor<1x3x100x100xbf16>) -> tensor<1x3x100x100xbf16>
    return %1 : tensor<1x3x100x100xbf16>
  }
}
