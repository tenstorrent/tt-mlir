module {
  func.func @concat_0(%arg0: tensor<1x1x17x64xf32>, %arg1: tensor<1x1x17x64xf32>) -> tensor<1x1x17x128xf32> {
    %0 = "ttir.concat"(%arg0, %arg1) <{dim = 3 : si32}> : (tensor<1x1x17x64xf32>, tensor<1x1x17x64xf32>) -> tensor<1x1x17x128xf32>
    return %0 : tensor<1x1x17x128xf32>
  }

  func.func @concat_1(%arg0: tensor<32x1x17x64xbf16>, %arg1: tensor<32x1x17x64xbf16>) -> tensor<32x1x17x128xbf16> {
    %0 = "ttir.concat"(%arg0, %arg1) <{dim = 3 : si32}> : (tensor<32x1x17x64xbf16>, tensor<32x1x17x64xbf16>) -> tensor<32x1x17x128xbf16>
    return %0 : tensor<32x1x17x128xbf16>
  }

  func.func @concat_2(%arg0: tensor<32x8x17x64xbf16>, %arg1: tensor<32x8x17x64xbf16>) -> tensor<32x8x17x128xbf16> {
    %0 = "ttir.concat"(%arg0, %arg1) <{dim = 3 : si32}> : (tensor<32x8x17x64xbf16>, tensor<32x8x17x64xbf16>) -> tensor<32x8x17x128xbf16>
    return %0 : tensor<32x8x17x128xbf16>
  }
}
