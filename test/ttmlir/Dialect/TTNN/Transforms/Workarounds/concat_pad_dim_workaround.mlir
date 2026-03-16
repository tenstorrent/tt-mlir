// RUN: ttmlir-opt --ttcore-register-device="system-desc-path=%system_desc_path%" --ttnn-workaround="ttnn-enable-noc-dma-hang-workarounds=true" -o %t %s
// RUN: FileCheck %s --input-file=%t

// Unit tests for ConcatOpPadDimRewritePattern.
// The pattern pads the last (unaligned) concat input to tile boundary, then
// slices the result back to the original shape. It only fires on Wormhole B0
// when a partial-tile column would produce a NOC DMA write < 32 bytes.

#dram = #ttnn.buffer_type<dram>

// bf16 element = 2 bytes.  last-dim 8 => partial 8 elements => 16 bytes < 32.
// Two inputs: first is tile-aligned (dim=3, size 32), last is unaligned (size 8).
// Pattern should apply: pad last input dim 32->32 (noop) and 8->32, concat, slice.
#layout_32 = #ttnn.ttnn_layout<(d0, d1) -> (d0, d1), <1x1>, memref<1x1x!ttcore.tile<32x32, bf16>, #dram>, <interleaved>>
#layout_8  = #ttnn.ttnn_layout<(d0, d1) -> (d0, d1), <1x1>, memref<1x1x!ttcore.tile<32x32, bf16>, #dram>, <interleaved>>
#layout_40 = #ttnn.ttnn_layout<(d0, d1) -> (d0, d1), <1x1>, memref<1x2x!ttcore.tile<32x32, bf16>, #dram>, <interleaved>>

module attributes {} {
  // Test 1: Workaround SHOULD apply - last input unaligned with small NOC write
  func.func @concat_pad_dim_should_apply(
      %arg0: tensor<32x32xbf16, #layout_32>,
      %arg1: tensor<32x8xbf16, #layout_8>
  ) -> tensor<32x40xbf16, #layout_40> {
    // CHECK-LABEL: func.func @concat_pad_dim_should_apply
    // CHECK: "ttnn.pad"
    // CHECK: "ttnn.concat"
    // CHECK: "ttnn.slice_static"
    %0 = "ttnn.concat"(%arg0, %arg1) <{dim = 1 : si32, memory_config = #ttnn.memory_config<#dram, <interleaved>>}> : (tensor<32x32xbf16, #layout_32>, tensor<32x8xbf16, #layout_8>) -> tensor<32x40xbf16, #layout_40>
    return %0 : tensor<32x40xbf16, #layout_40>
  }
}

// -----

// Test 2: Workaround should NOT apply - all inputs are tile-aligned
#layout_64 = #ttnn.ttnn_layout<(d0, d1) -> (d0, d1), <1x1>, memref<1x2x!ttcore.tile<32x32, bf16>, #dram>, <interleaved>>

module attributes {} {
  func.func @concat_pad_dim_all_aligned(
      %arg0: tensor<32x32xbf16, #layout_32>,
      %arg1: tensor<32x32xbf16, #layout_32>
  ) -> tensor<32x64xbf16, #layout_64> {
    // CHECK-LABEL: func.func @concat_pad_dim_all_aligned
    // CHECK: "ttnn.concat"
    // CHECK-NOT: "ttnn.pad"
    // CHECK-NOT: "ttnn.slice_static"
    %0 = "ttnn.concat"(%arg0, %arg1) <{dim = 1 : si32, memory_config = #ttnn.memory_config<#dram, <interleaved>>}> : (tensor<32x32xbf16, #layout_32>, tensor<32x32xbf16, #layout_32>) -> tensor<32x64xbf16, #layout_64>
    return %0 : tensor<32x64xbf16, #layout_64>
  }
}

// -----

// Test 3: Workaround should NOT apply - non-final input is unaligned
// (both inputs unaligned; pattern refuses to rewrite)
#layout_16 = #ttnn.ttnn_layout<(d0, d1) -> (d0, d1), <1x1>, memref<1x1x!ttcore.tile<32x32, bf16>, #dram>, <interleaved>>

module attributes {} {
  func.func @concat_pad_dim_non_final_unaligned(
      %arg0: tensor<32x8xbf16, #layout_8>,
      %arg1: tensor<32x8xbf16, #layout_8>
  ) -> tensor<32x16xbf16, #layout_16> {
    // CHECK-LABEL: func.func @concat_pad_dim_non_final_unaligned
    // CHECK: "ttnn.concat"
    // CHECK-NOT: "ttnn.pad"
    // CHECK-NOT: "ttnn.slice_static"
    %0 = "ttnn.concat"(%arg0, %arg1) <{dim = 1 : si32, memory_config = #ttnn.memory_config<#dram, <interleaved>>}> : (tensor<32x8xbf16, #layout_8>, tensor<32x8xbf16, #layout_8>) -> tensor<32x16xbf16, #layout_16>
    return %0 : tensor<32x16xbf16, #layout_16>
  }
}

// -----

// Test 4: Workaround should NOT apply - partial width is large enough
// (f32 element = 4 bytes, last-dim 16 => partial 16 elems => 64 bytes >= 32)
#layout_f32_16 = #ttnn.ttnn_layout<(d0, d1) -> (d0, d1), <1x1>, memref<1x1x!ttcore.tile<32x32, f32>, #dram>, <interleaved>>
#layout_f32_32 = #ttnn.ttnn_layout<(d0, d1) -> (d0, d1), <1x1>, memref<1x1x!ttcore.tile<32x32, f32>, #dram>, <interleaved>>
#layout_f32_48 = #ttnn.ttnn_layout<(d0, d1) -> (d0, d1), <1x1>, memref<1x2x!ttcore.tile<32x32, f32>, #dram>, <interleaved>>

module attributes {} {
  func.func @concat_pad_dim_large_partial_width(
      %arg0: tensor<32x32xf32, #layout_f32_32>,
      %arg1: tensor<32x16xf32, #layout_f32_16>
  ) -> tensor<32x48xf32, #layout_f32_48> {
    // CHECK-LABEL: func.func @concat_pad_dim_large_partial_width
    // CHECK: "ttnn.concat"
    // CHECK-NOT: "ttnn.pad"
    // CHECK-NOT: "ttnn.slice_static"
    %0 = "ttnn.concat"(%arg0, %arg1) <{dim = 1 : si32, memory_config = #ttnn.memory_config<#dram, <interleaved>>}> : (tensor<32x32xf32, #layout_f32_32>, tensor<32x16xf32, #layout_f32_16>) -> tensor<32x48xf32, #layout_f32_48>
    return %0 : tensor<32x48xf32, #layout_f32_48>
  }
}
