// REQUIRES: opmodel
// RUN: ttmlir-opt --ttir-to-ttnn-backend-pipeline="optimization-level=1 enable-greedy-optimizer=false override-conv3d-config=conv3d_full=weights_dtype#bf16:t_out_block#1:w_out_block#1:h_out_block#1:c_out_block#32:c_in_block#64,conv3d_partial=c_in_block#64" -o %t %s
// RUN: FileCheck %s --input-file=%t

// Verifies --override-conv3d-config pins config fields on Conv3dOps matched
// by NameLoc.
module {
  // Full override: every field pinned. The emitted conv3d_config must reflect
  // exactly the override values, ignoring the empirical scoring.
  func.func @full(
      %arg0: tensor<1x8x28x28x128xbf16>,
      %arg1: tensor<32x128x3x3x3xbf16>)
      -> tensor<1x6x26x26x32xbf16> {
    // CHECK: "ttnn.conv3d"
    // CHECK-SAME: conv3d_config = #ttnn.conv3d_config<
    // CHECK-SAME: weights_dtype = bf16
    // CHECK-SAME: t_out_block = 1
    // CHECK-SAME: w_out_block = 1
    // CHECK-SAME: h_out_block = 1
    // CHECK-SAME: c_out_block = 32
    // CHECK-SAME: c_in_block = 64
    %0 = "ttir.conv3d"(%arg0, %arg1) <{
        stride = array<i32: 1, 1, 1>,
        padding = array<i32: 0, 0, 0>,
        groups = 1 : i32,
        padding_mode = "zeros"
      }> : (tensor<1x8x28x28x128xbf16>, tensor<32x128x3x3x3xbf16>)
        -> tensor<1x6x26x26x32xbf16> loc(#loc_full)
    return %0 : tensor<1x6x26x26x32xbf16>
  }

  // Partial override: only c_in_block pinned at 64. The remaining fields
  // are chosen by the structural pre-filter + OpModel-driven ranking; this
  // test proves the override forces c_in_block=64 (the smoke test, with
  // no override, has c_in_block=64 anyway for this shape) while leaving
  // t/h/w/c_out to the optimizer's selection.
  func.func @partial(
      %arg0: tensor<1x8x28x28x128xbf16>,
      %arg1: tensor<32x128x3x3x3xbf16>)
      -> tensor<1x6x26x26x32xbf16> {
    // CHECK: "ttnn.conv3d"
    // CHECK-SAME: conv3d_config = #ttnn.conv3d_config<
    // CHECK-SAME: t_out_block = 6
    // CHECK-SAME: w_out_block = 4
    // CHECK-SAME: h_out_block = 8
    // CHECK-SAME: c_out_block = 32
    // CHECK-SAME: c_in_block = 64
    %0 = "ttir.conv3d"(%arg0, %arg1) <{
        stride = array<i32: 1, 1, 1>,
        padding = array<i32: 0, 0, 0>,
        groups = 1 : i32,
        padding_mode = "zeros"
      }> : (tensor<1x8x28x28x128xbf16>, tensor<32x128x3x3x3xbf16>)
        -> tensor<1x6x26x26x32xbf16> loc(#loc_partial)
    return %0 : tensor<1x6x26x26x32xbf16>
  }
}
#loc_full = loc("conv3d_full")
#loc_partial = loc("conv3d_partial")
