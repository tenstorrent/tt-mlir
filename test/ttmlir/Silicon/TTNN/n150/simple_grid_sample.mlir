// RUN: rm -rf %t.ttnn
// RUN: rm -rf %t.mlir
// RUN: ttmlir-opt --ttir-to-ttnn-backend-pipeline="system-desc-path=%system_desc_path%" -o %t.mlir %s
// RUN: FileCheck %s --input-file=%t.mlir
// RUN: ttmlir-translate --ttnn-to-flatbuffer -o %t.ttnn %t.mlir

module attributes {} {
  // tt-metal known-good shape: C=32 (TILE_WIDTH), small spatial.
  func.func public @grid_sample_small(
      %arg0: tensor<1x32x8x8xbf16>,
      %arg1: tensor<1x6x6x2xbf16>) -> tensor<1x32x6x6xbf16> {
    // CHECK-LABEL: func.func public @grid_sample_small(
    // CHECK: "ttnn.grid_sample"
    %0 = "ttir.grid_sample"(%arg0, %arg1) <{
      mode = "bilinear",
      padding_mode = "zeros",
      align_corners = false
    }> : (tensor<1x32x8x8xbf16>, tensor<1x6x6x2xbf16>) -> tensor<1x32x6x6xbf16>
    return %0 : tensor<1x32x6x6xbf16>
  }

  // align_corners=true variant.
  func.func public @grid_sample_align_corners(
      %arg0: tensor<2x64x50x50xbf16>,
      %arg1: tensor<2x25x25x2xbf16>) -> tensor<2x64x25x25xbf16> {
    // CHECK-LABEL: func.func public @grid_sample_align_corners(
    // CHECK: "ttnn.grid_sample"
    %0 = "ttir.grid_sample"(%arg0, %arg1) <{
      mode = "bilinear",
      padding_mode = "zeros",
      align_corners = true
    }> : (tensor<2x64x50x50xbf16>, tensor<2x25x25x2xbf16>) -> tensor<2x64x25x25xbf16>
    return %0 : tensor<2x64x25x25xbf16>
  }
}
