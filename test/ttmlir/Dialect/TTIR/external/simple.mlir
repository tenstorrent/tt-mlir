// RUN: ttmlir-opt --ttir-link-external-functions -o %t %s
// RUN: FileCheck %s --input-file=%t
//
// Check that a simple external function is linked correctly.

module {
  func.func @test_simple(%arg0: tensor<32x32xf32>) -> tensor<64x64xf32> {
    // CHECK: call @injected(%arg0)
    %0 = "ttir.invoke_external"(%arg0) {path = "simple.mlir.ext", entry = "injected"}
      : (tensor<32x32xf32>) -> tensor<64x64xf32>
    return %0 : tensor<64x64xf32>
  }
  // CHECK: func.func private @injected
}
