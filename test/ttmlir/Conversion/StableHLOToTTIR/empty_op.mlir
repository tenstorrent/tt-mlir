// REQUIRES: stablehlo
// RUN: ttmlir-opt --stablehlo-to-ttir-pipeline -o %t %s
// RUN: FileCheck %s --input-file=%t
module @module_empty {
  func.func @test_empty_boolean() -> tensor<1xi1> {
    // CHECK-LABEL: @test_empty_boolean
    // CHECK: %[[EMPTY:[0-9]+]] = ttir.empty() : tensor<1xi1>
    %0 = tensor.empty() : tensor<1xi1>
    // CHECK: return %[[EMPTY]] : tensor<1xi1>
    return %0 : tensor<1xi1>
  }

  func.func @test_empty_float() -> tensor<4x64xf32> {
    // CHECK-LABEL: @test_empty_float
    // CHECK: %[[EMPTY:[0-9]+]] = ttir.empty() : tensor<4x64xf32>
    %0 = tensor.empty() : tensor<4x64xf32>
    // CHECK: return %[[EMPTY]] : tensor<4x64xf32>
    return %0 : tensor<4x64xf32>
  }
}
