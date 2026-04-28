// RUN: ttmlir-opt --ttir-inline-external-functions -o %t %s
// RUN: FileCheck %s --input-file=%t
//
// Verify that invoking the same external entry point twice from the same module
// only inserts the function definition once.

module {
  func.func @test_multi_invoke() -> (tensor<2x3x4xi16>, tensor<2x3x4xi16>) {
    // CHECK: call @foo()
    %a = "ttir.invoke_external"() {path = "multi-invoke.mlir.ext", entry = "foo"}
      : () -> tensor<2x3x4xi16>
    // CHECK: call @foo()
    %b = "ttir.invoke_external"() {path = "multi-invoke.mlir.ext", entry = "foo"}
      : () -> tensor<2x3x4xi16>
    return %a, %b : tensor<2x3x4xi16>, tensor<2x3x4xi16>
  }

  // @foo must appear exactly once in the output.
  // CHECK:      func.func private @foo() -> tensor<2x3x4xi16>
  // CHECK-NOT:  @foo
}
