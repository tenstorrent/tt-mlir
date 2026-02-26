module {
  func.func @where_0(%arg0: tensor<32x1x17x128xi1>, %arg1: tensor<32x1x17x128xbf16>, %arg2: tensor<32x1x17x128xbf16>) -> tensor<32x1x17x128xbf16> {
    %0 = "ttir.where"(%arg0, %arg1, %arg2) : (tensor<32x1x17x128xi1>, tensor<32x1x17x128xbf16>, tensor<32x1x17x128xbf16>) -> tensor<32x1x17x128xbf16>
    return %0 : tensor<32x1x17x128xbf16>
  }

  func.func @where_1(%arg0: tensor<32x8x17x128xi1>, %arg1: tensor<32x8x17x128xf32>, %arg2: tensor<32x8x17x128xf32>) -> tensor<32x8x17x128xf32> {
    %0 = "ttir.where"(%arg0, %arg1, %arg2) : (tensor<32x8x17x128xi1>, tensor<32x8x17x128xf32>, tensor<32x8x17x128xf32>) -> tensor<32x8x17x128xf32>
    return %0 : tensor<32x8x17x128xf32>
  }
}
