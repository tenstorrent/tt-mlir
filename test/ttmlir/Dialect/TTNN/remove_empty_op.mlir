// RUN: ttmlir-opt --ttir-to-ttnn-backend-pipeline %s | FileCheck %s
module attributes {} {
  func.func @forward(%arg0: tensor<4x2x32x32xbf16>) -> tensor<2x4x32x32xbf16> {
    // CHECK-NOT: "ttnn.empty"
    %0 = tensor.empty() : tensor<2x4x32x32xbf16>
    // CHECK: %[[C:.*]] = "ttnn.reshape"[[C:.*]]
    %1 = "ttir.reshape"(%arg0, %0) <{shape = array<i64: 2, 4, 32, 32>}> : (tensor<4x2x32x32xbf16>, tensor<2x4x32x32xbf16>) -> tensor<2x4x32x32xbf16>
    return %1 : tensor<2x4x32x32xbf16>
  }
}
