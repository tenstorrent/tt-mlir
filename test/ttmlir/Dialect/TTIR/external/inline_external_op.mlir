// RUN: ttmlir-opt --ttir-inline-external-functions -o %t %s
// RUN: FileCheck %s --input-file=%t

module {
  func.func @test_inline_external(%arg0: tensor<32x32xf32>) -> tensor<64x64xf32> {
    // NOTE: the only way to pass scalars is as 0D tensors.
    // CHECK: "ttir.invoke_external"(%arg0)
    // CHECK-SAME: entry = "injected"
    // CHECK-SAME: path = "injected.mlir"
    %0 = "ttir.invoke_external"(%arg0)
      {path = "test/ttmlir/Dialect/TTIR/external/injected.mlir", entry = "injected"}
      : (tensor<32x32xf32>) -> tensor<64x64xf32>
    return %0 : tensor<64x64xf32>
  }
}
