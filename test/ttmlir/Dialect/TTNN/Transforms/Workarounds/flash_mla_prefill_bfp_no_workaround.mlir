// RUN: ttmlir-opt --split-input-file --ttcore-register-device --ttnn-workaround -o %t %s
// RUN: FileCheck %s --input-file=%t

// Verify the flash_mla_prefill workaround pass leaves inputs untouched when
// they are already in a tt-metal SDPA-supported dtype (bf16/bfp_bf8/bfp_bf4).

#dram = #ttnn.buffer_type<dram>
#l_qk_bfp8 = #ttnn.ttnn_layout<(d0, d1, d2, d3) -> (d0 * 512 + d1 * 32 + d2, d3), <1x1>, memref<16x4x!ttcore.tile<32x32, bfp_bf8>, #dram>, <interleaved>>
#l_out_bfp8 = #ttnn.ttnn_layout<(d0, d1, d2, d3) -> (d0 * 512 + d1 * 32 + d2, d3), <1x1>, memref<16x2x!ttcore.tile<32x32, bfp_bf8>, #dram>, <interleaved>>

// Q/K/output already bfp_bf8: pass must not insert any to_layout/typecast.
func.func @flash_mla_prefill_bfp8_no_workaround(
    %query: tensor<1x16x32x128xbf16, #l_qk_bfp8>,
    %key:   tensor<1x1x32x128xbf16, #l_qk_bfp8>)
    -> tensor<1x16x32x64xbf16, #l_out_bfp8> {
  // CHECK-LABEL: func.func @flash_mla_prefill_bfp8_no_workaround
  // CHECK-NOT: ttnn.to_layout
  // CHECK-NOT: ttnn.typecast
  // CHECK: "ttnn.flash_mla_prefill"
  %0 = "ttnn.flash_mla_prefill"(%query, %key) <{operandSegmentSizes = array<i32: 1, 1, 0, 0>, head_dim_v = 64 : ui32, is_causal = true}>
      : (tensor<1x16x32x128xbf16, #l_qk_bfp8>,
         tensor<1x1x32x128xbf16, #l_qk_bfp8>)
        -> tensor<1x16x32x64xbf16, #l_out_bfp8>
  return %0 : tensor<1x16x32x64xbf16, #l_out_bfp8>
}

// -----

#dram = #ttnn.buffer_type<dram>
#l_qk_bfp4 = #ttnn.ttnn_layout<(d0, d1, d2, d3) -> (d0 * 512 + d1 * 32 + d2, d3), <1x1>, memref<16x4x!ttcore.tile<32x32, bfp_bf4>, #dram>, <interleaved>>
#l_out_bfp4 = #ttnn.ttnn_layout<(d0, d1, d2, d3) -> (d0 * 512 + d1 * 32 + d2, d3), <1x1>, memref<16x2x!ttcore.tile<32x32, bfp_bf4>, #dram>, <interleaved>>

// Q/K/output already bfp_bf4: pass must not insert any to_layout/typecast.
func.func @flash_mla_prefill_bfp4_no_workaround(
    %query: tensor<1x16x32x128xbf16, #l_qk_bfp4>,
    %key:   tensor<1x1x32x128xbf16, #l_qk_bfp4>)
    -> tensor<1x16x32x64xbf16, #l_out_bfp4> {
  // CHECK-LABEL: func.func @flash_mla_prefill_bfp4_no_workaround
  // CHECK-NOT: ttnn.to_layout
  // CHECK-NOT: ttnn.typecast
  // CHECK: "ttnn.flash_mla_prefill"
  %0 = "ttnn.flash_mla_prefill"(%query, %key) <{operandSegmentSizes = array<i32: 1, 1, 0, 0>, head_dim_v = 64 : ui32, is_causal = true}>
      : (tensor<1x16x32x128xbf16, #l_qk_bfp4>,
         tensor<1x1x32x128xbf16, #l_qk_bfp4>)
        -> tensor<1x16x32x64xbf16, #l_out_bfp4>
  return %0 : tensor<1x16x32x64xbf16, #l_out_bfp4>
}

// -----

#dram = #ttnn.buffer_type<dram>
#l_qk_bf16 = #ttnn.ttnn_layout<(d0, d1, d2, d3) -> (d0 * 512 + d1 * 32 + d2, d3), <1x1>, memref<16x4x!ttcore.tile<32x32, bf16>, #dram>, <interleaved>>
#l_out_bf16 = #ttnn.ttnn_layout<(d0, d1, d2, d3) -> (d0 * 512 + d1 * 32 + d2, d3), <1x1>, memref<16x2x!ttcore.tile<32x32, bf16>, #dram>, <interleaved>>
#l_mask_bfp8 = #ttnn.ttnn_layout<(d0, d1, d2, d3) -> (d0 * 32 + d1 * 32 + d2, d3), <1x1>, memref<1x1x!ttcore.tile<32x32, bfp_bf8>, #dram>, <interleaved>>

// Q/K bf16 but mask bfp_bf8: workaround must not touch Q/K/output OR mask
// (all three are SDPA-supported dtypes).
func.func @flash_mla_prefill_bf16_qk_bfp8_mask(
    %query: tensor<1x16x32x128xbf16, #l_qk_bf16>,
    %key:   tensor<1x1x32x128xbf16, #l_qk_bf16>,
    %mask:  tensor<1x1x32x32xbf16, #l_mask_bfp8>)
    -> tensor<1x16x32x64xbf16, #l_out_bf16> {
  // CHECK-LABEL: func.func @flash_mla_prefill_bf16_qk_bfp8_mask
  // CHECK-NOT: ttnn.to_layout
  // CHECK-NOT: ttnn.typecast
  // CHECK: "ttnn.flash_mla_prefill"
  %0 = "ttnn.flash_mla_prefill"(%query, %key, %mask) <{operandSegmentSizes = array<i32: 1, 1, 0, 1>, head_dim_v = 64 : ui32, is_causal = false}>
      : (tensor<1x16x32x128xbf16, #l_qk_bf16>,
         tensor<1x1x32x128xbf16, #l_qk_bf16>,
         tensor<1x1x32x32xbf16, #l_mask_bfp8>)
        -> tensor<1x16x32x64xbf16, #l_out_bf16>
  return %0 : tensor<1x16x32x64xbf16, #l_out_bf16>
}
