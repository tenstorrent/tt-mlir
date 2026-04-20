// RUN: ttmlir-opt --ttir-propagate-weight-dtype %s | FileCheck %s

// Test that weight_dtype propagates through a chain of TM ops to the matmul.

module {
  // CHECK-LABEL: func.func @propagate_through_chain
  func.func @propagate_through_chain(
    %arg0: tensor<32x256xbf16>,
    %arg1: tensor<1x128x256xbf16> {ttcore.argument_type = #ttcore.argument_type<parameter>, ttcore.weight_dtype = "bfp_bf8"}
  ) -> tensor<32x128xbf16> {
    %0 = "ttir.reshape"(%arg1) <{shape = [128 : i32, 256 : i32]}> : (tensor<1x128x256xbf16>) -> tensor<128x256xbf16>
    %1 = "ttir.permute"(%0) <{permutation = array<i64: 1, 0>}> : (tensor<128x256xbf16>) -> tensor<256x128xbf16>
    // CHECK: "ttir.matmul"
    // CHECK-SAME: ttcore.weight_dtype = "bfp_bf8"
    %2 = "ttir.matmul"(%arg0, %1) : (tensor<32x256xbf16>, tensor<256x128xbf16>) -> tensor<32x128xbf16>
    return %2 : tensor<32x128xbf16>
  }
}
