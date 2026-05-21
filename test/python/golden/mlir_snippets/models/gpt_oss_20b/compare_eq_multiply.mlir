// Cross-type compare + multiply pattern observed in GPT-OSS-20B (variant
// reported by Brandon on issue #8193). The compare op takes si32 indices and
// produces a bf16 result, which then feeds a bf16 multiply. Same failure mode
// as the gt-based attention mask: the si32 -> bf16 broadcast inside the
// generic region creates a type-mismatched DST slot.

module {
  func.func @compare_eq_multiply(
      %indices_a : tensor<1x4x544xsi32>,
      %indices_b : tensor<4x1x1xsi32>,
      %input     : tensor<1x4x544xbf16>)
      -> tensor<4x4x544xbf16> {
    %0 = "ttir.eq"(%indices_a, %indices_b) : (tensor<1x4x544xsi32>, tensor<4x1x1xsi32>) -> tensor<4x4x544xbf16>
    %1 = "ttir.multiply"(%0, %input) : (tensor<4x4x544xbf16>, tensor<1x4x544xbf16>) -> tensor<4x4x544xbf16>
    return %1 : tensor<4x4x544xbf16>
  }
}
