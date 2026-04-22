// Attention output: softmax @ V -> typecast -> permute -> reshape -> output projection
// dot_general -> typecast -> permute -> reshape -> dot_general

module {
  func.func @attn_output(%arg0: tensor<32x32x18x18xf32>, %arg1: tensor<32x32x18x128xf32>, %arg2: tensor<4096x2560xbf16>) -> tensor<576x2560xbf16> {
    %0 = "ttir.dot_general"(%arg0, %arg1) <{batch_dims_lhs = array<i64: 0, 1>, batch_dims_rhs = array<i64: 0, 1>, contract_dims_lhs = array<i64: 3>, contract_dims_rhs = array<i64: 2>}> : (tensor<32x32x18x18xf32>, tensor<32x32x18x128xf32>) -> tensor<32x32x18x128xf32>
    %1 = "ttir.typecast"(%0) <{conservative_folding = false}> : (tensor<32x32x18x128xf32>) -> tensor<32x32x18x128xbf16>
    %2 = "ttir.permute"(%1) <{permutation = array<i64: 0, 2, 1, 3>}> : (tensor<32x32x18x128xbf16>) -> tensor<32x18x32x128xbf16>
    %3 = "ttir.reshape"(%2) <{shape = [576 : i32, 4096 : i32]}> : (tensor<32x18x32x128xbf16>) -> tensor<576x4096xbf16>
    %4 = "ttir.dot_general"(%3, %arg2) <{batch_dims_lhs = array<i64>, batch_dims_rhs = array<i64>, contract_dims_lhs = array<i64: 1>, contract_dims_rhs = array<i64: 0>}> : (tensor<576x4096xbf16>, tensor<4096x2560xbf16>) -> tensor<576x2560xbf16>
    return %4 : tensor<576x2560xbf16>
  }
}
