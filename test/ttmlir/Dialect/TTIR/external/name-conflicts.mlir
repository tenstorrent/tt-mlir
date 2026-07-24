// RUN: ttmlir-opt --ttir-link-external-functions -o %t %s
// RUN: FileCheck %s --input-file=%t
//
// Check that the link pass renames the injected `foo` to avoid a conflict
// with the caller module's own `foo`.

module {
  // CHECK: func.func @foo()
  func.func @foo() -> tensor<2x2xf32> {
    %0 = "ttir.constant"() <{value = dense<0.0> : tensor<2x2xf32>}> : () -> tensor<2x2xf32>
    return %0 : tensor<2x2xf32>
  }

  func.func @test_name_conflict() -> (tensor<2x2xf32>, tensor<2x2xf32>) {
    // CHECK: call @foo()
    %zeros = func.call @foo() : () -> tensor<2x2xf32>
    // NOTE: Invoke the external foo (all ones); the pass must rename it to avoid
    // clobbering the local @foo.
    // CHECK: call @foo_0()
    %ones = "ttir.invoke_external"() {path = "name-conflicts.mlir.ext", entry = "foo"}
      : () -> tensor<2x2xf32>
    return %zeros, %ones : tensor<2x2xf32>, tensor<2x2xf32>
  }

  // The injected foo must appear in the output under its renamed symbol.
  // CHECK: func.func private @foo_0()
}
