module {
  func.func @silu_gate_matmul_64x4608_to_64x3584_bf16(%arg0: tensor<64x4608xbf16>, %arg1: tensor<64x4608xbf16>, %arg2: tensor<4608x3584xbf16>) -> tensor<64x3584xbf16> {
    %0 = "ttir.typecast"(%arg0) <{conservative_folding = false}> : (tensor<64x4608xbf16>) -> tensor<64x4608xf32>
    %1 = "ttir.sigmoid"(%0) : (tensor<64x4608xf32>) -> tensor<64x4608xf32>
    %2 = "ttir.multiply"(%0, %1) : (tensor<64x4608xf32>, tensor<64x4608xf32>) -> tensor<64x4608xf32>
    %3 = "ttir.typecast"(%2) <{conservative_folding = false}> : (tensor<64x4608xf32>) -> tensor<64x4608xbf16>
    %4 = "ttir.multiply"(%3, %arg1) : (tensor<64x4608xbf16>, tensor<64x4608xbf16>) -> tensor<64x4608xbf16>
    %5 = "ttir.matmul"(%4, %arg2) <{transpose_a = false, transpose_b = false}> : (tensor<64x4608xbf16>, tensor<4608x3584xbf16>) -> tensor<64x3584xbf16>
    return %5 : tensor<64x3584xbf16>
  }
}
