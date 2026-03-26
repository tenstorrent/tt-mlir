module {
  func.func @concat_0(%arg0: tensor<1x1x32x32xf32>, %arg1: tensor<1x1x32x32xf32>) -> tensor<1x1x32x64xf32> {
    %0 = "ttir.concat"(%arg0, %arg1) <{dim = 3 : si32}> : (tensor<1x1x32x32xf32>, tensor<1x1x32x32xf32>) -> tensor<1x1x32x64xf32>
    return %0 : tensor<1x1x32x64xf32>
  }

  func.func @concat_1(%arg0: tensor<1x8x32x32xbf16>, %arg1: tensor<1x8x32x32xbf16>) -> tensor<1x8x32x64xbf16> {
    %0 = "ttir.concat"(%arg0, %arg1) <{dim = 3 : si32}> : (tensor<1x8x32x32xbf16>, tensor<1x8x32x32xbf16>) -> tensor<1x8x32x64xbf16>
    return %0 : tensor<1x8x32x64xbf16>
  }

  func.func @concat_2(%arg0: tensor<1x8x32x64xbf16>, %arg1: tensor<1x8x32x64xbf16>) -> tensor<1x8x32x128xbf16> {
    %0 = "ttir.concat"(%arg0, %arg1) <{dim = 3 : si32}> : (tensor<1x8x32x64xbf16>, tensor<1x8x32x64xbf16>) -> tensor<1x8x32x128xbf16>
    return %0 : tensor<1x8x32x128xbf16>
  }

  func.func @concat_3(%arg0: tensor<1x96x32x32xbf16>, %arg1: tensor<1x96x32x32xbf16>) -> tensor<1x96x32x64xbf16> {
    %0 = "ttir.concat"(%arg0, %arg1) <{dim = 3 : si32}> : (tensor<1x96x32x32xbf16>, tensor<1x96x32x32xbf16>) -> tensor<1x96x32x64xbf16>
    return %0 : tensor<1x96x32x64xbf16>
  }

  func.func @concat_4(%arg0: tensor<1x96x32x64xbf16>, %arg1: tensor<1x96x32x64xbf16>) -> tensor<1x96x32x128xbf16> {
    %0 = "ttir.concat"(%arg0, %arg1) <{dim = 3 : si32}> : (tensor<1x96x32x64xbf16>, tensor<1x96x32x64xbf16>) -> tensor<1x96x32x128xbf16>
    return %0 : tensor<1x96x32x128xbf16>
  }
}
