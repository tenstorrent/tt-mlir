// Attention mask subgraph from GPT-OSS-20B (decode shape: 1x1x1x128) with an
// integer-indexed compare front-end. This shape exercises the cross-type
// comparison path - the comparison op takes si32 indices and produces a bf16
// mask. Previously this lowering failed in `d2m-insert-dst-register-access`
// because the `tile_bcast` from si32 -> bf16 inside the generic region left a
// si32-typed DST slot for a bf16-typed compute tail. See issue #8193.
// 3 ops: gt(si32, si32) -> bf16, then two logical_ands.

module {
  func.func @attention_mask_compare_decode(
      %indices_a : tensor<1x1x1x128xsi32>,
      %indices_b : tensor<1x1x1x1xsi32>,
      %mask      : tensor<1x1x1x1xbf16>,
      %input_b   : tensor<1x1x1x128xbf16>)
      -> tensor<1x1x1x128xbf16> {
    %0 = "ttir.gt"(%indices_a, %indices_b) : (tensor<1x1x1x128xsi32>, tensor<1x1x1x1xsi32>) -> tensor<1x1x1x128xbf16>
    %1 = "ttir.logical_and"(%mask, %0) : (tensor<1x1x1x1xbf16>, tensor<1x1x1x128xbf16>) -> tensor<1x1x1x128xbf16>
    %2 = "ttir.logical_and"(%1, %input_b) : (tensor<1x1x1x128xbf16>, tensor<1x1x1x128xbf16>) -> tensor<1x1x1x128xbf16>
    return %2 : tensor<1x1x1x128xbf16>
  }
}
