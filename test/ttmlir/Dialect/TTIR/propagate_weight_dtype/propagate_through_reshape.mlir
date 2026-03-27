// RUN: ttmlir-opt --ttir-propagate-weight-dtype %s | FileCheck %s

// Test that weight_dtype propagates through a reshape op to the matmul.

module {
  // CHECK-LABEL: func.func @propagate_through_reshape
  func.func @propagate_through_reshape(
    %arg0: tensor<32x128xbf16>,
    %arg1: tensor<1x128x256xbf16> {ttcore.argument_type = #ttcore.argument_type<parameter>, ttcore.weight_dtype = "bfp_bf8"}
  ) -> tensor<32x256xbf16> {
    %0 = "ttir.reshape"(%arg1) <{shape = [128 : i32, 256 : i32]}> : (tensor<1x128x256xbf16>) -> tensor<128x256xbf16>
    // CHECK: "ttir.matmul"
    // CHECK-SAME: ttcore.weight_dtype = "bfp_bf8"
    %1 = "ttir.matmul"(%arg0, %0) : (tensor<32x128xbf16>, tensor<128x256xbf16>) -> tensor<32x256xbf16>
    return %1 : tensor<32x256xbf16>
  }
}
