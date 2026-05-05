// RUN: ttmlir-opt --ttir-to-ttnn-backend-pipeline -o %t %s
// RUN: FileCheck %s --input-file=%t
module {
  // tt-metal known-good shape: C=32 (TILE_WIDTH), small spatial.
  func.func @grid_sample_small(
      %arg0: tensor<1x32x8x8xbf16>,
      %arg1: tensor<1x6x6x2xbf16>) -> tensor<1x32x6x6xbf16> {
    // CHECK: "ttnn.grid_sample"
    %0 = "ttir.grid_sample"(%arg0, %arg1) <{
      mode = "bilinear",
      padding_mode = "zeros",
      align_corners = false
    }> : (tensor<1x32x8x8xbf16>, tensor<1x6x6x2xbf16>) -> tensor<1x32x6x6xbf16>
    return %0 : tensor<1x32x6x6xbf16>
  }

  // Deformable DETR shape — the original DRAM-OOM use case.
  func.func @grid_sample_deformable_detr(
      %arg0: tensor<8x32x100x134xbf16>,
      %arg1: tensor<8x17821x4x2xbf16>) -> tensor<8x32x17821x4xbf16> {
    // CHECK: "ttnn.grid_sample"
    %0 = "ttir.grid_sample"(%arg0, %arg1) <{
      mode = "bilinear",
      padding_mode = "zeros",
      align_corners = false
    }> : (tensor<8x32x100x134xbf16>, tensor<8x17821x4x2xbf16>) -> tensor<8x32x17821x4xbf16>
    return %0 : tensor<8x32x17821x4xbf16>
  }

  // align_corners=true variant.
  func.func @grid_sample_align_corners(
      %arg0: tensor<2x64x50x50xbf16>,
      %arg1: tensor<2x25x25x2xbf16>) -> tensor<2x64x25x25xbf16> {
    // CHECK: "ttnn.grid_sample"
    %0 = "ttir.grid_sample"(%arg0, %arg1) <{
      mode = "bilinear",
      padding_mode = "zeros",
      align_corners = true
    }> : (tensor<2x64x50x50xbf16>, tensor<2x25x25x2xbf16>) -> tensor<2x64x25x25xbf16>
    return %0 : tensor<2x64x25x25xbf16>
  }
}
