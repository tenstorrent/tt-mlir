module {
  func.func @to_layout(%arg0: tensor<32x32xbf16>) -> tensor<32x32xf32> {
    %0 = ttir.empty() : tensor<32x32xf32>
    %1 = ttir.to_layout %arg0, %0 : tensor<32x32xbf16> into tensor<32x32xf32> -> tensor<32x32xf32>
    return %1 : tensor<32x32xf32>
  }
}
