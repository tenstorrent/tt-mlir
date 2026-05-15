// Two D2M subgraphs (unary-only eltwise chains) separated by a matmul.
// Chain 1 (abs, neg, exp) -> matmul -> Chain 2 (sigmoid, neg, abs)

module {
  func.func @unary_matmul_unary(
      %arg0: tensor<64x128xbf16>,
      %arg1: tensor<128x64xbf16>)
      -> tensor<64x64xbf16> {
    %0 = "ttir.abs"(%arg0) : (tensor<64x128xbf16>) -> tensor<64x128xbf16>
    %1 = "ttir.neg"(%0) : (tensor<64x128xbf16>) -> tensor<64x128xbf16>
    %2 = "ttir.exp"(%1) : (tensor<64x128xbf16>) -> tensor<64x128xbf16>
    %3 = "ttir.matmul"(%2, %arg1) <{transpose_a = false, transpose_b = false}> : (tensor<64x128xbf16>, tensor<128x64xbf16>) -> tensor<64x64xbf16>
    %4 = "ttir.sigmoid"(%3) : (tensor<64x64xbf16>) -> tensor<64x64xbf16>
    %5 = "ttir.neg"(%4) : (tensor<64x64xbf16>) -> tensor<64x64xbf16>
    %6 = "ttir.abs"(%5) : (tensor<64x64xbf16>) -> tensor<64x64xbf16>
    return %6 : tensor<64x64xbf16>
  }
}
