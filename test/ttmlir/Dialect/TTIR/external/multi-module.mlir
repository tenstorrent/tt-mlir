// RUN: ttmlir-opt --ttir-link-external-functions -o %t %s
// RUN: FileCheck %s --input-file=%t
//
// Check that we can link a function from two different external modules.

module {
  func.func @test_multi_module() -> (tensor<i32>, tensor<32xbf16>) {
    // CHECK: call @foo()
    %a = "ttir.invoke_external"() {path = "multi-module-1.mlir.ext", entry = "foo"}
      : () -> tensor<i32>
    // CHECK: call @foo_0()
    %b = "ttir.invoke_external"() {path = "multi-module-2.mlir.ext", entry = "foo"}
      : () -> tensor<32xbf16>
    return %a, %b : tensor<i32>, tensor<32xbf16>
  }

  // CHECK: func.func private @foo() -> tensor<i32>
  // CHECK: func.func private @foo_0() -> tensor<32xbf16>
}
