// REQUIRES: opmodel
// RUN: ttmlir-opt --ttir-to-ttnn-backend-pipeline="optimization-level=1 enable-greedy-optimizer=true override-conv3d-config=conv3d_full=weights_dtype#bf16:t_out_block#1:w_out_block#1:h_out_block#1:c_out_block#32:c_in_block#64,conv3d_partial=c_in_block#64" -o %t %s
// RUN: FileCheck %s --input-file=%t

// Same as conv3d_config_override.mlir but with the greedy memory-layout
// optimizer (enable-greedy-optimizer=true, the default at optimization-level>=1)
// instead of the chain TTNNOptimizer. Regression test for Conv3dRuleBook
// getOutputHints: without it the greedy path's base getOutputHints emits a null
// hint that drops the Conv3dConfig, so --override-conv3d-config was silently
// ignored under greedy. Both paths must pin the same fields.
module {
  // Full override: every field pinned. The emitted conv3d_config must reflect
  // exactly the override values. TTNNPrepareConv3dWeights additionally fills
  // compute_with_storage_grid_size from the device worker grid (the override
  // syntax has no field for it), so a grid must also appear.
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
    // CHECK-SAME: compute_with_storage_grid_size = #ttcore.grid<
    %0 = "ttir.conv3d"(%arg0, %arg1) <{
        stride = array<i32: 1, 1, 1>,
        padding = array<i32: 0, 0, 0>,
        groups = 1 : i32,
        padding_mode = "zeros"
      }> : (tensor<1x8x28x28x128xbf16>, tensor<32x128x3x3x3xbf16>)
        -> tensor<1x6x26x26x32xbf16> loc(#loc_full)
    return %0 : tensor<1x6x26x26x32xbf16>
  }

  // Partial override: only c_in_block pinned at 64. The greedy path must still
  // carry the override config through; TTNNPrepareConv3dWeights then completes
  // the unset fields with tt-metal's defaults (t/w/h_out_block = 1,
  // c_out_block = 32) plus the device worker grid.
  func.func @partial(
      %arg0: tensor<1x8x28x28x128xbf16>,
      %arg1: tensor<32x128x3x3x3xbf16>)
      -> tensor<1x6x26x26x32xbf16> {
    // CHECK: "ttnn.conv3d"
    // CHECK-SAME: conv3d_config = #ttnn.conv3d_config<
    // CHECK-SAME: t_out_block = 1
    // CHECK-SAME: w_out_block = 1
    // CHECK-SAME: h_out_block = 1
    // CHECK-SAME: c_out_block = 32
    // CHECK-SAME: c_in_block = 64
    // CHECK-SAME: compute_with_storage_grid_size = #ttcore.grid<
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
