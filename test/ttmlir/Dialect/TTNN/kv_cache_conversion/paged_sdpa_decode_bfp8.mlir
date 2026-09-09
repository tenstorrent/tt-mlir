// RUN: ttmlir-opt --ttnn-kv-cache-dtype-conversion="target-dtype=bfp_bf8" %s | FileCheck %s

#dram = #ttnn.buffer_type<dram>

// The paged SDPA decode op reads its key/value from the KV cache. After the
// cache is converted to BFP8 the key/value become BFP8 while the query stays
// bf16. The tt-metal decode kernel supports exactly this (BF16 query over a
// BFP8 paged cache), so the pass must leave the query untouched and the op must
// still verify.

#cache = #ttnn.ttnn_layout<(d0, d1, d2, d3) -> (d0 * 512 + d1 * 64 + d2, d3), <1x1>, memref<16x4x!ttcore.tile<32x32, bf16>, #dram>, <interleaved>>
#query = #ttnn.ttnn_layout<(d0, d1, d2, d3) -> (d0 * 8 + d1 * 8 + d2, d3), <1x1>, memref<1x4x!ttcore.tile<32x32, bf16>, #dram>, <interleaved>>
#pagetable = #ttnn.ttnn_layout<(d0, d1) -> (d0, d1), <1x1>, memref<1x1xsi32, #dram>, <interleaved>>
#curpos = #ttnn.ttnn_layout<(d0) -> (0, d0), <1x1>, memref<1x1xsi32, #dram>, <interleaved>>

// CHECK-LABEL: func.func @test_paged_sdpa_decode_bfp8
// Key/value cache args are converted to BFP8.
// CHECK-SAME: %arg0: tensor<1x8x64x128x!ttcore.tile<32x32, bfp_bf8>
// CHECK-SAME: %arg1: tensor<1x8x64x128x!ttcore.tile<32x32, bfp_bf8>
// Query stays bf16.
// CHECK-SAME: %arg2: tensor<1x1x8x128xbf16
module attributes {} {
  func.func @test_paged_sdpa_decode_bfp8(
      %key: tensor<1x8x64x128xbf16, #cache> {ttcore.kv_cache},
      %value: tensor<1x8x64x128xbf16, #cache> {ttcore.kv_cache},
      %query: tensor<1x1x8x128xbf16, #query>,
      %page_table: tensor<1x1xsi32, #pagetable>,
      %cur_pos: tensor<1xsi32, #curpos>
  ) -> tensor<1x1x8x128xbf16, #query> attributes {tt.function_type = "forward_device"} {
    // No typecast is inserted on the query; the op runs with a BF16 query and
    // BFP8 key/value.
    // CHECK-NOT: "ttnn.typecast"
    // CHECK: "ttnn.paged_scaled_dot_product_attention_decode"(%arg2, %arg0, %arg1, %arg3, %arg4)
    %0 = "ttnn.paged_scaled_dot_product_attention_decode"(%query, %key, %value, %page_table, %cur_pos) <{is_causal = true, operandSegmentSizes = array<i32: 1, 1, 1, 1, 0, 1, 0>}> : (
        tensor<1x1x8x128xbf16, #query>,
        tensor<1x8x64x128xbf16, #cache>,
        tensor<1x8x64x128xbf16, #cache>,
        tensor<1x1xsi32, #pagetable>,
        tensor<1xsi32, #curpos>
    ) -> tensor<1x1x8x128xbf16, #query>
    return %0 : tensor<1x1x8x128xbf16, #query>
  }
}
