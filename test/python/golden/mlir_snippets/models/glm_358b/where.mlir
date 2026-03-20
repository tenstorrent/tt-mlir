module {
  func.func @where_0(%arg0: tensor<1x1x32x32xi1>, %arg1: tensor<1x1x32x32xbf16>, %arg2: tensor<1x1x32x32xbf16>) -> tensor<1x1x32x32xbf16> {
    %0 = "ttir.where"(%arg0, %arg1, %arg2) : (tensor<1x1x32x32xi1>, tensor<1x1x32x32xbf16>, tensor<1x1x32x32xbf16>) -> tensor<1x1x32x32xbf16>
    return %0 : tensor<1x1x32x32xbf16>
  }

  func.func @where_1(%arg0: tensor<1x96x32x1xi1>, %arg1: tensor<1x96x32x32xf32>, %arg2: tensor<1x96x32x32xf32>) -> tensor<1x96x32x32xf32> {
    %0 = "ttir.where"(%arg0, %arg1, %arg2) : (tensor<1x96x32x1xi1>, tensor<1x96x32x32xf32>, tensor<1x96x32x32xf32>) -> tensor<1x96x32x32xf32>
    return %0 : tensor<1x96x32x32xf32>
  }
}
