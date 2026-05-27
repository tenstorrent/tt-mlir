// RUN: ttmlir-opt --ttcore-register-device --ttnn-decomposition --split-input-file -o %t %s
// RUN: FileCheck %s --input-file=%t

#dram = #ttnn.buffer_type<dram>
#full = #ttnn.ttnn_layout<(d0, d1, d2, d3) -> (d0 * 4 + d1 * 4 + d2, d3), <1x1>, memref<1x4x!ttcore.tile<32x32, bf16>, #dram>, <interleaved>>
#half = #ttnn.ttnn_layout<(d0, d1, d2, d3) -> (d0 * 4 + d1 * 4 + d2, d3), <1x1>, memref<1x2x!ttcore.tile<32x32, bf16>, #dram>, <interleaved>>

// When both cos and sin caches arrive as `ttnn.concat(x, x, dim=-1)`
// duplications (half-D → full-D), the decomposition should pick the
// complex-rotation form: four half-D multiplies + one half-D subtract +
// one half-D add + a single half→full concat. No `ttnn.neg`, and the
// duplication concats fall away as dead code.
module {
  func.func @rope_decomp_complex_rotation_from_self_concat(
      %x: tensor<1x1x4x128xbf16, #full>,
      %cos_half: tensor<1x1x4x64xbf16, #half>,
      %sin_half: tensor<1x1x4x64xbf16, #half>)
      -> tensor<1x1x4x128xbf16, #full> {
    // CHECK-LABEL: func.func @rope_decomp_complex_rotation_from_self_concat
    // No rope op, no neg, and no `ttnn.concat` *before* the half->full
    // reassembly — proves the two input self-concat duplications were
    // dead-code-eliminated when the rope op got replaced.
    // CHECK-NOT: "ttnn.concat"
    // CHECK-NOT: "ttnn.rotary_embedding"
    // CHECK-NOT: "ttnn.neg"
    // CHECK: "ttnn.slice_static"
    // CHECK: "ttnn.slice_static"
    // CHECK: "ttnn.multiply"
    // CHECK: "ttnn.multiply"
    // CHECK: "ttnn.subtract"
    // CHECK: "ttnn.multiply"
    // CHECK: "ttnn.multiply"
    // CHECK: "ttnn.add"
    // CHECK: "ttnn.concat"
    // CHECK-SAME: dim = 3
    // No further concat after the reassembly — exactly one concat total.
    // CHECK-NOT: "ttnn.concat"
    // CHECK-NOT: "ttnn.neg"
    %cos = "ttnn.concat"(%cos_half, %cos_half) <{dim = 3 : si32}> :
      (tensor<1x1x4x64xbf16, #half>, tensor<1x1x4x64xbf16, #half>) ->
       tensor<1x1x4x128xbf16, #full>
    %sin = "ttnn.concat"(%sin_half, %sin_half) <{dim = 3 : si32}> :
      (tensor<1x1x4x64xbf16, #half>, tensor<1x1x4x64xbf16, #half>) ->
       tensor<1x1x4x128xbf16, #full>
    %r = "ttnn.rotary_embedding"(%x, %cos, %sin) :
      (tensor<1x1x4x128xbf16, #full>,
       tensor<1x1x4x128xbf16, #full>,
       tensor<1x1x4x128xbf16, #full>) -> tensor<1x1x4x128xbf16, #full>
    return %r : tensor<1x1x4x128xbf16, #full>
  }
}

// -----

#dram = #ttnn.buffer_type<dram>
#full = #ttnn.ttnn_layout<(d0, d1, d2, d3) -> (d0 * 4 + d1 * 4 + d2, d3), <1x1>, memref<1x4x!ttcore.tile<32x32, bf16>, #dram>, <interleaved>>

// When cos/sin do not match the self-concat pattern (e.g. they arrive
// already full-D from elsewhere), the existing rotate_half decomposition
// should still trigger: a single `ttnn.neg` and a `ttnn.concat` reassembling
// neg(x_hi) and x_lo.
module {
  func.func @rope_decomp_rotate_half_fallback(
      %x: tensor<1x1x4x128xbf16, #full>,
      %cos: tensor<1x1x4x128xbf16, #full>,
      %sin: tensor<1x1x4x128xbf16, #full>)
      -> tensor<1x1x4x128xbf16, #full> {
    // CHECK-LABEL: func.func @rope_decomp_rotate_half_fallback
    // CHECK-NOT: "ttnn.rotary_embedding"
    // CHECK: "ttnn.slice_static"
    // CHECK: "ttnn.slice_static"
    // CHECK: "ttnn.neg"
    // CHECK: "ttnn.concat"
    // CHECK: "ttnn.multiply"
    // CHECK: "ttnn.multiply"
    // CHECK: "ttnn.add"
    %r = "ttnn.rotary_embedding"(%x, %cos, %sin) :
      (tensor<1x1x4x128xbf16, #full>,
       tensor<1x1x4x128xbf16, #full>,
       tensor<1x1x4x128xbf16, #full>) -> tensor<1x1x4x128xbf16, #full>
    return %r : tensor<1x1x4x128xbf16, #full>
  }
}
