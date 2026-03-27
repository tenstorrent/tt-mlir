// RUN: ttmlir-opt --ttir-propagate-weight-dtype %s | FileCheck %s

// Test that weight_dtype propagates through a broadcast op to the matmul.
// BroadcastOp does not have the TensorManipulation trait, but should still
// be traced through.

module {
  // CHECK-LABEL: func.func @propagate_through_broadcast
  func.func @propagate_through_broadcast(
    %arg0: tensor<32x128xbf16>,
    %arg1: tensor<1x256xbf16> {ttcore.argument_type = #ttcore.argument_type<parameter>, ttcore.weight_dtype = "bfp_bf8"}
  ) -> tensor<32x256xbf16> {
    %0 = "ttir.broadcast"(%arg1) <{broadcast_dimensions = array<i64: 128, 1>}> : (tensor<1x256xbf16>) -> tensor<128x256xbf16>
    // CHECK: "ttir.matmul"
    // CHECK-SAME: ttcore.weight_dtype = "bfp_bf8"
    %1 = "ttir.matmul"(%arg0, %0) : (tensor<32x128xbf16>, tensor<128x256xbf16>) -> tensor<32x256xbf16>
    return %1 : tensor<32x256xbf16>
  }
}
