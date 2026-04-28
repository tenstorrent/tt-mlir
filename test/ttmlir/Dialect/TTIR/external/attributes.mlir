// RUN: ttmlir-opt --ttir-link-external-functions -o %t %s
// RUN: FileCheck %s --input-file=%t
//
// Check that the `#ttnn.ttnn_layout` attribute defined only in the external
// module is preserved in the destination module after linking.
//
// CHECK: ttnn.ttnn_layout
// CHECK-LABEL: func.func @test_attrs
// CHECK: call @compute(%arg0)
// CHECK-LABEL: func.func private @compute_with_layout
// CHECK-LABEL: func.func private @compute

module {
  func.func @test_attrs(%arg0: tensor<64x128xbf16>) -> tensor<64x128xbf16> {
    %0 = "ttir.invoke_external"(%arg0) {path = "attributes.mlir.ext", entry = "compute"}
      : (tensor<64x128xbf16>) -> tensor<64x128xbf16>
    return %0 : tensor<64x128xbf16>
  }
}
