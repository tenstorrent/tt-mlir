// RUN: ttmlir-opt --ttcore-register-device --ttnn-workaround -o %t %s
// RUN: FileCheck %s --input-file=%t
// Unit tests for layer_norm_pre_all_gather zero-pad workaround.

// Verify that when input width is not a multiple of TILE_WIDTH (32), a
// ttnn.pad with value=0.0 is inserted before layer_norm_pre_all_gather to
// explicitly zero the tile padding and satisfy the kernel's contract.

#dram = #ttnn.buffer_type<dram>
// Input layout: width=72 (not tile-multiple, pads to 96)
#ttnn_layout_72 = #ttnn.ttnn_layout<(d0, d1, d2, d3) -> (d0 * 32 + d1, d2, d3), <1x1>, memref<32x1x!ttcore.tile<32x32, bf16>, #dram>, <interleaved>>
// Padded input layout: width=96 (3 tiles)
#ttnn_layout_96 = #ttnn.ttnn_layout<(d0, d1, d2, d3) -> (d0 * 32 + d1, d2, d3), <1x1>, memref<32x3x!ttcore.tile<32x32, bf16>, #dram>, <interleaved>>
// Output layout: always width=64 (2 tiles of stats)
#ttnn_layout_out = #ttnn.ttnn_layout<(d0, d1, d2, d3) -> (d0 * 32 + d1, d2, d3), <1x1>, memref<32x2x!ttcore.tile<32x32, bf16>, #dram>, <interleaved>>
// Tile-aligned input layout: width=64
#ttnn_layout_64 = #ttnn.ttnn_layout<(d0, d1, d2, d3) -> (d0 * 32 + d1, d2, d3), <1x1>, memref<32x2x!ttcore.tile<32x32, bf16>, #dram>, <interleaved>>

module attributes {} {
  // CHECK-LABEL: layer_norm_pre_all_gather_non_tile_aligned_width
  func.func @layer_norm_pre_all_gather_non_tile_aligned_width(
      %arg0: tensor<1x1x32x72xbf16, #ttnn_layout_72>)
      -> tensor<1x1x32x64xbf16, #ttnn_layout_out> {
    // CHECK: %[[PAD:.*]] = "ttnn.pad"(%arg0)
    // CHECK-SAME: padding = array<i32: 0, 0, 0, 0, 0, 0, 0, 24>
    // CHECK-SAME: value = 0.000000e+00
    // CHECK: "ttnn.layer_norm_pre_all_gather"(%[[PAD]])
    // CHECK-SAME: operandSegmentSizes = array<i32: 1, 0, 0>
    %0 = "ttnn.layer_norm_pre_all_gather"(%arg0) <{dtype = #ttcore.supportedDataTypes<bf16>, operandSegmentSizes = array<i32: 1, 0, 0>}> : (tensor<1x1x32x72xbf16, #ttnn_layout_72>) -> tensor<1x1x32x64xbf16, #ttnn_layout_out>
    return %0 : tensor<1x1x32x64xbf16, #ttnn_layout_out>
  }

  // Verify that tile-aligned width is left unchanged (no pad inserted).
  // CHECK-LABEL: layer_norm_pre_all_gather_tile_aligned_width
  func.func @layer_norm_pre_all_gather_tile_aligned_width(
      %arg0: tensor<1x1x32x64xbf16, #ttnn_layout_64>)
      -> tensor<1x1x32x64xbf16, #ttnn_layout_out> {
    // CHECK-NOT: "ttnn.pad"
    // CHECK: "ttnn.layer_norm_pre_all_gather"(%arg0)
    %0 = "ttnn.layer_norm_pre_all_gather"(%arg0) <{dtype = #ttcore.supportedDataTypes<bf16>, operandSegmentSizes = array<i32: 1, 0, 0>}> : (tensor<1x1x32x64xbf16, #ttnn_layout_64>) -> tensor<1x1x32x64xbf16, #ttnn_layout_out>
    return %0 : tensor<1x1x32x64xbf16, #ttnn_layout_out>
  }
}
