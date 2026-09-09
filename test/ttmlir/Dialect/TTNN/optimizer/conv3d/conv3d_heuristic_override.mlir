// REQUIRES: opmodel
// RUN: ttmlir-opt --ttir-to-ttnn-backend-pipeline="optimization-level=1 override-conv3d-config=conv3d_match=c_out_block#96" -o %t %s
// RUN: FileCheck %s --input-file=%t

// Verifies override/heuristic precedence: a --override-conv3d-config field wins
// over the tt-metal _DEFAULT_BLOCKINGS heuristic, while the fields the override
// leaves unset come from the heuristic (not the constant conversion default).
module {
  // in=192, out=384, kernel=(3,3,3) matches the table entry
  //   (192, 384, (3, 3, 3)): (C_in=64, C_out=128, T=1, H=8, W=4).
  // The override pins c_out_block=96, so the emitted config must show
  // c_out_block=96 (override wins over the heuristic's 128) while
  // w_out_block=4, h_out_block=8, c_in_block=64 come from the heuristic --
  // values that differ from both the override and the constant conversion
  // default (w=h=1, c_in=32), proving the heuristic, not the default, supplied
  // the unset fields. (c_out_block=96 also divides out_channels=384, so it
  // survives the post-optimizer config validation.)
  func.func @match(
      %arg0: tensor<1x8x28x28x192xbf16>,
      %arg1: tensor<384x192x3x3x3xbf16>)
      -> tensor<1x6x26x26x384xbf16> {
    // CHECK: "ttnn.conv3d"
    // CHECK-SAME: conv3d_config = #ttnn.conv3d_config<
    // CHECK-SAME: t_out_block = 1
    // CHECK-SAME: w_out_block = 4
    // CHECK-SAME: h_out_block = 8
    // CHECK-SAME: c_out_block = 96
    // CHECK-SAME: c_in_block = 64
    // CHECK-SAME: compute_with_storage_grid_size = #ttcore.grid<
    %0 = "ttir.conv3d"(%arg0, %arg1) <{
        stride = array<i32: 1, 1, 1>,
        padding = array<i32: 0, 0, 0>,
        groups = 1 : i32,
        padding_mode = "zeros"
      }> : (tensor<1x8x28x28x192xbf16>, tensor<384x192x3x3x3xbf16>)
        -> tensor<1x6x26x26x384xbf16> loc(#loc_match)
    return %0 : tensor<1x6x26x26x384xbf16>
  }
}
#loc_match = loc("conv3d_match")
