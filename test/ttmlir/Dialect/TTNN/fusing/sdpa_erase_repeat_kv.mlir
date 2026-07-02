// End-to-end pipeline test for the GQA SDPA fusion. A frontend emits an
// explicit repeat_interleave (repeat_kv) that expands the 8 KV heads to match
// the 32 query heads before scaled_dot_product_attention. With the
// enable-sdpa-erase-repeat-kv pipeline option, the expansion is removed and the
// un-expanded K/V are handed straight to the TTNN SDPA op (GQA broadcast).

// RUN: ttmlir-opt --ttir-to-ttnn-backend-pipeline="system-desc-path=%system_desc_path% enable-ttnn-decomposition-pass=false enable-sdpa-erase-repeat-kv=true" -o %t.on %s
// RUN: FileCheck %s --check-prefix=ON --input-file=%t.on
//
// RUN: ttmlir-opt --ttir-to-ttnn-backend-pipeline="system-desc-path=%system_desc_path% enable-ttnn-decomposition-pass=false" -o %t.off %s
// RUN: FileCheck %s --check-prefix=OFF --input-file=%t.off

module {
  // ON-LABEL:  func.func @gqa_sdpa
  // OFF-LABEL: func.func @gqa_sdpa
  func.func @gqa_sdpa(%query: tensor<1x32x128x64xbf16>,
                      %key: tensor<1x8x128x64xbf16>,
                      %value: tensor<1x8x128x64xbf16>,
                      %mask: tensor<1x1x128x128xbf16>)
                      -> tensor<1x32x128x64xbf16> {
    // With the flag on, the repeat_interleave expansion is folded away and
    // SDPA consumes the un-expanded K/V directly.
    // ON-NOT: ttnn.repeat_interleave
    // ON: ttnn.scaled_dot_product_attention
    //
    // With the flag off (default), the expansion survives to TTNN.
    // OFF: ttnn.repeat_interleave
    // OFF: ttnn.scaled_dot_product_attention
    %ke = "ttir.repeat_interleave"(%key) <{repeats = 4 : ui32, dim = 1 : si32}> : (tensor<1x8x128x64xbf16>) -> tensor<1x32x128x64xbf16>
    %ve = "ttir.repeat_interleave"(%value) <{repeats = 4 : ui32, dim = 1 : si32}> : (tensor<1x8x128x64xbf16>) -> tensor<1x32x128x64xbf16>
    %out = "ttir.scaled_dot_product_attention"(%query, %ke, %ve, %mask) <{operandSegmentSizes = array<i32: 1, 1, 1, 1, 0>, is_causal = false, scale = 0.0883883461 : f32}> : (tensor<1x32x128x64xbf16>, tensor<1x32x128x64xbf16>, tensor<1x32x128x64xbf16>, tensor<1x1x128x128xbf16>) -> tensor<1x32x128x64xbf16>
    return %out : tensor<1x32x128x64xbf16>
  }
}
