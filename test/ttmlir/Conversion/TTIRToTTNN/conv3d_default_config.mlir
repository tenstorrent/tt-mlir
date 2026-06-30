// RUN: ttmlir-opt --ttir-to-ttnn-backend-pipeline %s | FileCheck %s

// Every emitted Conv3dOp must carry a *complete* Conv3dConfigAttr. tt-metal
// only auto-derives a full default config when conv3d_config is absent; given
// any partial config it leaves the unset fields at their struct defaults
// (C_out_block = 0, grid = 1x1), which break the kernel's blocking invariants
// for some shapes. TTIRToTTNN pins c_in_block = TILE_WIDTH (32) and
// TTNNPrepareConv3dWeights completes the config with tt-metal's own defaults
// (t/w/h_out_block = 1, c_out_block = 32) plus the device worker grid, so the
// op's config is the single source of truth on every backend.
module {
  func.func @conv3d_default_c_in_block(%arg0: tensor<1x8x28x28x32xbf16>, %arg1: tensor<32x32x3x3x3xbf16>) -> tensor<1x6x26x26x32xbf16> {
    // CHECK: "ttnn.conv3d"
    // CHECK-SAME: conv3d_config = #ttnn.conv3d_config<
    // CHECK-SAME: t_out_block = 1
    // CHECK-SAME: w_out_block = 1
    // CHECK-SAME: h_out_block = 1
    // CHECK-SAME: c_out_block = 32
    // CHECK-SAME: c_in_block = 32
    // CHECK-SAME: compute_with_storage_grid_size = #ttcore.grid<
    %0 = "ttir.conv3d"(%arg0, %arg1)
            <{
              stride = array<i32: 1, 1, 1>,
              padding = array<i32: 0, 0, 0>,
              padding_mode = "zeros",
              groups = 1: i32
            }> : (tensor<1x8x28x28x32xbf16>, tensor<32x32x3x3x3xbf16>) -> tensor<1x6x26x26x32xbf16>
    return %0 : tensor<1x6x26x26x32xbf16>
  }
}
