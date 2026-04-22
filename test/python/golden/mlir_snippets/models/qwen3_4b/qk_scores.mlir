// QK attention score computation from Qwen3 4B with scaling and mask add
// typecast -> multiply(scale=1/sqrt(128)) -> dot_general -> add -> typecast

module {
  func.func @qk_scores(%arg0: tensor<32x32x18x128xbf16>, %arg1: tensor<32x32x128x18xf32>, %arg2: tensor<32x32x18x18xf32>) -> tensor<32x32x18x18xf64> {
    %cst = "ttir.full"() <{fill_value = 0.297301769 : f32, shape = array<i32: 32, 32, 18, 128>}> : () -> tensor<32x32x18x128xf32>
    %0 = "ttir.typecast"(%arg0) <{conservative_folding = false}> : (tensor<32x32x18x128xbf16>) -> tensor<32x32x18x128xf32>
    %1 = "ttir.multiply"(%0, %cst) : (tensor<32x32x18x128xf32>, tensor<32x32x18x128xf32>) -> tensor<32x32x18x128xf32>
    %2 = "ttir.dot_general"(%1, %arg1) <{batch_dims_lhs = array<i64: 0, 1>, batch_dims_rhs = array<i64: 0, 1>, contract_dims_lhs = array<i64: 3>, contract_dims_rhs = array<i64: 2>}> : (tensor<32x32x18x128xf32>, tensor<32x32x128x18xf32>) -> tensor<32x32x18x18xf32>
    %3 = "ttir.add"(%2, %arg2) : (tensor<32x32x18x18xf32>, tensor<32x32x18x18xf32>) -> tensor<32x32x18x18xf32>
    %4 = "ttir.typecast"(%3) <{conservative_folding = false}> : (tensor<32x32x18x18xf32>) -> tensor<32x32x18x18xf64>
    return %4 : tensor<32x32x18x18xf64>
  }
}
