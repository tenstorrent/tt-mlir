// REQUIRES: opmodel
// RUN: ttmlir-opt --ttir-to-ttnn-backend-pipeline="optimization-level=1 enable-greedy-optimizer=false" -o %t %s
// RUN: FileCheck %s --input-file=%t

// Smoke test for Conv3dConfigSearchSpace integration. With the optimizer
// enabled, the lowered ttnn.conv3d op must carry a non-null conv3d_config
// attribute. Without this wiring, conv3d_config stays at the TTIRToTTNN
// default nullptr (which yields the bad tt-metal default blocking of all-1).
//
// Output shape (1, 6, 26, 26, 32) so T_out=6, H_out=W_out=26, c_out=32.
// The optimizer's structural pre-filter picks the top-K candidates by
// block volume and aspect balance; OpModel then validates legality and
// re-ranks by runtime cycles. The expected values below are the OpModel
// winner for this shape — a regression that collapses the search back
// to (1,1,1,32,32) or significantly changes the OpModel-best pick will
// be caught here.
module {
  func.func @conv3d_smoke(
      %arg0: tensor<1x8x28x28x128xbf16>,
      %arg1: tensor<32x128x3x3x3xbf16>)
      -> tensor<1x6x26x26x32xbf16> {
    // CHECK: "ttnn.conv3d"
    // CHECK-SAME: conv3d_config = #ttnn.conv3d_config<
    // CHECK-SAME: t_out_block = 6
    // CHECK-SAME: w_out_block = 4
    // CHECK-SAME: h_out_block = 8
    // CHECK-SAME: c_out_block = 32
    // CHECK-SAME: c_in_block = 128
    %0 = "ttir.conv3d"(%arg0, %arg1) <{
        stride = array<i32: 1, 1, 1>,
        padding = array<i32: 0, 0, 0>,
        groups = 1 : i32,
        padding_mode = "zeros"
      }> : (tensor<1x8x28x28x128xbf16>, tensor<32x128x3x3x3xbf16>)
        -> tensor<1x6x26x26x32xbf16>
    return %0 : tensor<1x6x26x26x32xbf16>
  }
}
