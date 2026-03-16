// RUN: ttmlir-opt --ttcore-register-device="system-desc-path=%system_desc_path%" --ttnn-workaround="ttnn-enable-noc-dma-hang-workarounds=true" -o %t %s
// RUN: FileCheck %s --input-file=%t

// Unit tests for ReshapeNarrowTiledRewritePattern.
// The pattern decomposes a tiled reshape into TILE->RM + reshape + RM->TILE
// when the output last-dim partial tile would produce a NOC DMA write < 32
// bytes (Wormhole B0 hang).

#dram = #ttnn.buffer_type<dram>

// Test 1: Workaround SHOULD apply.
// bf16 (2 bytes), output last-dim 8 => partial 8 elems => 16 bytes < 32.
// Input 32x32 -> Output 64x8: last dim changes, small partial tile.
#layout_32x32 = #ttnn.ttnn_layout<(d0, d1) -> (d0, d1), <1x1>, memref<1x1x!ttcore.tile<32x32, bf16>, #dram>, <interleaved>>
#layout_64x8  = #ttnn.ttnn_layout<(d0, d1) -> (d0, d1), <1x1>, memref<2x1x!ttcore.tile<32x32, bf16>, #dram>, <interleaved>>

module attributes {} {
  func.func @reshape_narrow_should_apply(
      %arg0: tensor<32x32xbf16, #layout_32x32>
  ) -> tensor<64x8xbf16, #layout_64x8> {
    // CHECK-LABEL: func.func @reshape_narrow_should_apply
    // Step 1: to_layout TILE -> ROW_MAJOR
    // CHECK: "ttnn.to_layout"
    // CHECK-SAME: layout = #ttnn.layout<row_major>
    // Step 2: reshape in ROW_MAJOR
    // CHECK: "ttnn.reshape"
    // Step 3: to_layout ROW_MAJOR -> TILE
    // CHECK: "ttnn.to_layout"
    // CHECK-SAME: layout = #ttnn.layout<tile>
    %0 = "ttnn.reshape"(%arg0) <{shape = [64 : i32, 8 : i32]}> : (tensor<32x32xbf16, #layout_32x32>) -> tensor<64x8xbf16, #layout_64x8>
    return %0 : tensor<64x8xbf16, #layout_64x8>
  }
}

// -----

// Test 2: Workaround should NOT apply - output last dim is tile-aligned.
// 32x32 -> 16x64: last dim 64 is tile-aligned, no partial tile.
#layout_16x64 = #ttnn.ttnn_layout<(d0, d1) -> (d0, d1), <1x1>, memref<1x2x!ttcore.tile<32x32, bf16>, #dram>, <interleaved>>

module attributes {} {
  func.func @reshape_narrow_aligned_output(
      %arg0: tensor<32x32xbf16, #layout_32x32>
  ) -> tensor<16x64xbf16, #layout_16x64> {
    // CHECK-LABEL: func.func @reshape_narrow_aligned_output
    // CHECK: "ttnn.reshape"
    // CHECK-NOT: layout = #ttnn.layout<row_major>
    %0 = "ttnn.reshape"(%arg0) <{shape = [16 : i32, 64 : i32]}> : (tensor<32x32xbf16, #layout_32x32>) -> tensor<16x64xbf16, #layout_16x64>
    return %0 : tensor<16x64xbf16, #layout_16x64>
  }
}

// -----

// Test 3: Workaround should NOT apply - segment bytes >= 32.
// f32 (4 bytes), output last-dim 16 => partial 16 elems => 64 bytes >= 32.
#layout_f32_32x32 = #ttnn.ttnn_layout<(d0, d1) -> (d0, d1), <1x1>, memref<1x1x!ttcore.tile<32x32, f32>, #dram>, <interleaved>>
#layout_f32_64x16 = #ttnn.ttnn_layout<(d0, d1) -> (d0, d1), <1x1>, memref<2x1x!ttcore.tile<32x32, f32>, #dram>, <interleaved>>

module attributes {} {
  func.func @reshape_narrow_large_segment(
      %arg0: tensor<32x32xf32, #layout_f32_32x32>
  ) -> tensor<64x16xf32, #layout_f32_64x16> {
    // CHECK-LABEL: func.func @reshape_narrow_large_segment
    // CHECK: "ttnn.reshape"
    // CHECK-NOT: layout = #ttnn.layout<row_major>
    %0 = "ttnn.reshape"(%arg0) <{shape = [64 : i32, 16 : i32]}> : (tensor<32x32xf32, #layout_f32_32x32>) -> tensor<64x16xf32, #layout_f32_64x16>
    return %0 : tensor<64x16xf32, #layout_f32_64x16>
  }
}

// -----

// Test 4: Workaround should NOT apply - view reshape (last dim unchanged).
// 1x32x8 -> 32x8: last dim stays 8 (same), second-to-last also same => view.
#layout_1x32x8 = #ttnn.ttnn_layout<(d0, d1, d2) -> (d0 * 32 + d1, d2), <1x1>, memref<1x1x!ttcore.tile<32x32, bf16>, #dram>, <interleaved>>
#layout_32x8   = #ttnn.ttnn_layout<(d0, d1) -> (d0, d1), <1x1>, memref<1x1x!ttcore.tile<32x32, bf16>, #dram>, <interleaved>>

module attributes {} {
  func.func @reshape_narrow_view_skip(
      %arg0: tensor<1x32x8xbf16, #layout_1x32x8>
  ) -> tensor<32x8xbf16, #layout_32x8> {
    // CHECK-LABEL: func.func @reshape_narrow_view_skip
    // CHECK: "ttnn.reshape"
    // The view-skip logic should prevent the pattern from applying because
    // last dim is unchanged (8 == 8) and second-to-last is unchanged (32 == 32).
    %0 = "ttnn.reshape"(%arg0) <{shape = [32 : i32, 8 : i32]}> : (tensor<1x32x8xbf16, #layout_1x32x8>) -> tensor<32x8xbf16, #layout_32x8>
    return %0 : tensor<32x8xbf16, #layout_32x8>
  }
}
