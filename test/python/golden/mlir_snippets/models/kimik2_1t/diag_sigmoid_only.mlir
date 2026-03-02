module {
  func.func @sigmoid_only(%arg0: tensor<64x4608xbf16>) -> tensor<64x4608xbf16> {
    %0 = "ttir.typecast"(%arg0) <{conservative_folding = false}> : (tensor<64x4608xbf16>) -> tensor<64x4608xf32>
    %1 = "ttir.sigmoid"(%0) : (tensor<64x4608xf32>) -> tensor<64x4608xf32>
    %2 = "ttir.typecast"(%1) <{conservative_folding = false}> : (tensor<64x4608xf32>) -> tensor<64x4608xbf16>
    return %2 : tensor<64x4608xbf16>
  }
}
