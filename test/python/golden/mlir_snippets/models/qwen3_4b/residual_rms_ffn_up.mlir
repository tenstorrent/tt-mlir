// Residual add + RMS norm + FFN up projection from Qwen3 4B
// add -> rms_norm -> reshape -> permute -> dot_general

module {
  func.func @residual_rms_ffn_up(%arg0: tensor<32x18x2560xbf16>, %arg1: tensor<32x18x2560xbf16>, %arg2: tensor<2560xbf16>, %arg3: tensor<9728x2560xbf16>) -> tensor<576x9728xbf16> {
    %0 = "ttir.add"(%arg0, %arg1) : (tensor<32x18x2560xbf16>, tensor<32x18x2560xbf16>) -> tensor<32x18x2560xbf16>
    %1 = "ttir.rms_norm"(%0, %arg2) <{epsilon = 9.99999997E-7 : f32, normalized_shape = array<i64: 2560>, operandSegmentSizes = array<i32: 1, 1, 0>}> : (tensor<32x18x2560xbf16>, tensor<2560xbf16>) -> tensor<32x18x2560xbf16>
    %2 = "ttir.reshape"(%1) <{shape = [576 : i32, 2560 : i32]}> : (tensor<32x18x2560xbf16>) -> tensor<576x2560xbf16>
    %3 = "ttir.permute"(%arg3) <{permutation = array<i64: 1, 0>}> : (tensor<9728x2560xbf16>) -> tensor<2560x9728xbf16>
    %4 = "ttir.dot_general"(%2, %3) <{batch_dims_lhs = array<i64>, batch_dims_rhs = array<i64>, contract_dims_lhs = array<i64: 1>, contract_dims_rhs = array<i64: 0>}> : (tensor<576x2560xbf16>, tensor<2560x9728xbf16>) -> tensor<576x9728xbf16>
    return %4 : tensor<576x9728xbf16>
  }
}
