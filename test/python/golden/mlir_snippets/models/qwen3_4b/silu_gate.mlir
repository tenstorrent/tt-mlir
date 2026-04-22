// SiLU gate activation from Qwen3 4B FFN: typecast -> sigmoid -> multiply -> typecast -> multiply
// Corresponds to the gated SiLU used in the MLP block

module {
  func.func @silu_gate(%arg0: tensor<32x18x9728xbf16>, %arg1: tensor<32x18x9728xbf16>) -> tensor<32x18x9728xbf16> {
    %0 = "ttir.typecast"(%arg0) <{conservative_folding = false}> : (tensor<32x18x9728xbf16>) -> tensor<32x18x9728xf32>
    %1 = "ttir.sigmoid"(%0) : (tensor<32x18x9728xf32>) -> tensor<32x18x9728xf32>
    %2 = "ttir.multiply"(%0, %1) : (tensor<32x18x9728xf32>, tensor<32x18x9728xf32>) -> tensor<32x18x9728xf32>
    %3 = "ttir.typecast"(%2) <{conservative_folding = false}> : (tensor<32x18x9728xf32>) -> tensor<32x18x9728xbf16>
    %4 = "ttir.multiply"(%3, %arg1) : (tensor<32x18x9728xbf16>, tensor<32x18x9728xbf16>) -> tensor<32x18x9728xbf16>
    return %4 : tensor<32x18x9728xbf16>
  }
}
