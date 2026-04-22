// FFN gate multiply + down projection from Qwen3 4B MLP
// multiply -> reshape -> permute -> dot_general -> reshape

module {
  func.func @ffn_gate_down(%arg0: tensor<32x18x9728xbf16>, %arg1: tensor<32x18x9728xbf16>, %arg2: tensor<2560x9728xbf16>) -> tensor<32x18x2560xbf16> {
    %0 = "ttir.multiply"(%arg0, %arg1) : (tensor<32x18x9728xbf16>, tensor<32x18x9728xbf16>) -> tensor<32x18x9728xbf16>
    %1 = "ttir.reshape"(%0) <{shape = [576 : i32, 9728 : i32]}> : (tensor<32x18x9728xbf16>) -> tensor<576x9728xbf16>
    %2 = "ttir.permute"(%arg2) <{permutation = array<i64: 1, 0>}> : (tensor<2560x9728xbf16>) -> tensor<9728x2560xbf16>
    %3 = "ttir.dot_general"(%1, %2) <{batch_dims_lhs = array<i64>, batch_dims_rhs = array<i64>, contract_dims_lhs = array<i64: 1>, contract_dims_rhs = array<i64: 0>}> : (tensor<576x9728xbf16>, tensor<9728x2560xbf16>) -> tensor<576x2560xbf16>
    %4 = "ttir.reshape"(%3) <{shape = [32 : i32, 18 : i32, 2560 : i32]}> : (tensor<576x2560xbf16>) -> tensor<32x18x2560xbf16>
    return %4 : tensor<32x18x2560xbf16>
  }
}
