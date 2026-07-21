// REQUIRES: opmodel
// RUN: ttmlir-opt --ttir-to-ttnn-backend-pipeline="optimization-level=1" -o %t %s
// RUN: FileCheck %s --input-file=%t

// Verifies the conv3d config heuristic (tt-metal _DEFAULT_BLOCKINGS lookup in
// LegalOpConfigAnalysis). When (in_channels, out_channels, kernel) matches a
// table entry the optimizer pins the table's blocking; otherwise it keeps the
// conversion default.
module {
  // in=192, out=384, kernel=(3,3,3) matches _DEFAULT_BLOCKINGS:
  //   (192, 384, (3, 3, 3)): (C_in=64, C_out=128, T=1, H=8, W=4)
  // Remapped to Conv3dConfig fields: t=1, w=4, h=8, c_out=128, c_in=64. Note
  // c_out_block=128 and w/h_out_block != 1 distinguish this from the constant
  // conversion default (t=w=h=1, c_out=32, c_in=32).
  func.func @match(
      %arg0: tensor<1x8x28x28x192xbf16>,
      %arg1: tensor<384x192x3x3x3xbf16>)
      -> tensor<1x6x26x26x384xbf16> {
    // CHECK: "ttnn.conv3d"
    // CHECK-SAME: conv3d_config = #ttnn.conv3d_config<
    // CHECK-SAME: t_out_block = 1
    // CHECK-SAME: w_out_block = 4
    // CHECK-SAME: h_out_block = 8
    // CHECK-SAME: c_out_block = 128
    // CHECK-SAME: c_in_block = 64
    // CHECK-SAME: compute_with_storage_grid_size = #ttcore.grid<
    %0 = "ttir.conv3d"(%arg0, %arg1) <{
        stride = array<i32: 1, 1, 1>,
        padding = array<i32: 0, 0, 0>,
        groups = 1 : i32,
        padding_mode = "zeros"
      }> : (tensor<1x8x28x28x192xbf16>, tensor<384x192x3x3x3xbf16>)
        -> tensor<1x6x26x26x384xbf16>
    return %0 : tensor<1x6x26x26x384xbf16>
  }

  // in=128, out=32, kernel=(3,3,3) is absent from the table, so the optimizer
  // keeps the constant conversion default: t=w=h=1, c_out=32, c_in=32.
  func.func @nomatch(
      %arg0: tensor<1x8x28x28x128xbf16>,
      %arg1: tensor<32x128x3x3x3xbf16>)
      -> tensor<1x6x26x26x32xbf16> {
    // CHECK: "ttnn.conv3d"
    // CHECK-SAME: conv3d_config = #ttnn.conv3d_config<
    // CHECK-SAME: t_out_block = 1
    // CHECK-SAME: w_out_block = 1
    // CHECK-SAME: h_out_block = 1
    // CHECK-SAME: c_out_block = 32
    // CHECK-SAME: c_in_block = 32
    // CHECK-SAME: compute_with_storage_grid_size = #ttcore.grid<
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
