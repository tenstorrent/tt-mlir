// RUN: ttmlir-opt --ttir-propagate-weight-dtype %s | FileCheck %s

// Test that matmul does NOT get a weight_dtype attribute when the arg has none.

module {
  // CHECK-LABEL: func.func @no_propagate_without_annotation
  func.func @no_propagate_without_annotation(
    %arg0: tensor<32x128xbf16>,
    %arg1: tensor<128x256xbf16> {ttcore.argument_type = #ttcore.argument_type<parameter>}
  ) -> tensor<32x256xbf16> {
    // CHECK: "ttir.matmul"(%arg0, %arg1)
    // CHECK-NOT: ttcore.weight_dtype
    %0 = "ttir.matmul"(%arg0, %arg1) : (tensor<32x128xbf16>, tensor<128x256xbf16>) -> tensor<32x256xbf16>
    return %0 : tensor<32x256xbf16>
  }
}
