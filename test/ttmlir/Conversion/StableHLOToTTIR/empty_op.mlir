// REQUIRES: stablehlo
// RUN: ttmlir-opt --stablehlo-to-ttir-pipeline %s | FileCheck %s
module @module_empty {
  func.func @test_empty_boolean() -> tensor<1xi1> {
    // CHECK: %[[EMPTY:[0-9]+]] = tensor.empty() : tensor<1xbf16>
    %0 = tensor.empty() : tensor<1xi1>
    // CHECK:     return %[[EMPTY]] : tensor<1xbf16>
    return %0 : tensor<1xi1>
  }
}
