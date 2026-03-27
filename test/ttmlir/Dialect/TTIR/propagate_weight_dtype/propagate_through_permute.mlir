// RUN: ttmlir-opt --ttir-propagate-weight-dtype %s | FileCheck %s

// Test that weight_dtype propagates through a permute op to the matmul.

module {
  // CHECK-LABEL: func.func @propagate_through_permute
  func.func @propagate_through_permute(
    %arg0: tensor<32x256xbf16>,
    %arg1: tensor<128x256xbf16> {ttcore.argument_type = #ttcore.argument_type<parameter>, ttcore.weight_dtype = "bfp_bf4"}
  ) -> tensor<32x128xbf16> {
    %0 = "ttir.permute"(%arg1) <{permutation = array<i64: 1, 0>}> : (tensor<128x256xbf16>) -> tensor<256x128xbf16>
    // CHECK: "ttir.matmul"
    // CHECK-SAME: ttcore.weight_dtype = "bfp_bf4"
    %1 = "ttir.matmul"(%arg0, %0) : (tensor<32x256xbf16>, tensor<256x128xbf16>) -> tensor<32x128xbf16>
    return %1 : tensor<32x128xbf16>
  }
}
