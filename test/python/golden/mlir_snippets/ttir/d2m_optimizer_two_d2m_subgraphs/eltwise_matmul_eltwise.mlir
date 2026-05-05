// Two D2M subgraphs separated by a matmul.
// Chain 1 (exp, multiply, add) -> matmul -> Chain 2 (sigmoid, multiply, add)

module {
  func.func @eltwise_matmul_eltwise(
      %arg0: tensor<4x64x128xbf16>,
      %arg1: tensor<4x64x128xbf16>,
      %arg2: tensor<4x64x128xbf16>,
      %arg3: tensor<4x128x64xbf16>,
      %arg4: tensor<4x64x64xbf16>,
      %arg5: tensor<4x64x64xbf16>)
      -> tensor<4x64x64xbf16> {
    %0 = "ttir.exp"(%arg0) : (tensor<4x64x128xbf16>) -> tensor<4x64x128xbf16>
    %1 = "ttir.multiply"(%0, %arg1) : (tensor<4x64x128xbf16>, tensor<4x64x128xbf16>) -> tensor<4x64x128xbf16>
    %2 = "ttir.add"(%1, %arg2) : (tensor<4x64x128xbf16>, tensor<4x64x128xbf16>) -> tensor<4x64x128xbf16>
    %3 = "ttir.dot_general"(%2, %arg3) <{batch_dims_lhs = array<i64: 0>, batch_dims_rhs = array<i64: 0>, contract_dims_lhs = array<i64: 2>, contract_dims_rhs = array<i64: 1>}> : (tensor<4x64x128xbf16>, tensor<4x128x64xbf16>) -> tensor<4x64x64xbf16>
    %4 = "ttir.sigmoid"(%3) : (tensor<4x64x64xbf16>) -> tensor<4x64x64xbf16>
    %5 = "ttir.multiply"(%4, %arg4) : (tensor<4x64x64xbf16>, tensor<4x64x64xbf16>) -> tensor<4x64x64xbf16>
    %6 = "ttir.add"(%5, %arg5) : (tensor<4x64x64xbf16>, tensor<4x64x64xbf16>) -> tensor<4x64x64xbf16>
    return %6 : tensor<4x64x64xbf16>
  }
}
