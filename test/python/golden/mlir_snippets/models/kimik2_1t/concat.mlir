module {
  func.func @concat_0(%arg0: tensor<32x32xbf16>, %arg1: tensor<32x32xbf16>) -> tensor<32x64xbf16> {
    %0 = "ttir.concat"(%arg0, %arg1) <{dim = 1 : si32}> : (tensor<32x32xbf16>, tensor<32x32xbf16>) -> tensor<32x64xbf16>
    return %0 : tensor<32x64xbf16>
  }

  func.func @concat_1(%arg0: tensor<1x128x32x32xbf16>, %arg1: tensor<1x128x32x32xbf16>) -> tensor<1x128x32x64xbf16> {
    %0 = "ttir.concat"(%arg0, %arg1) <{dim = 3 : si32}> : (tensor<1x128x32x32xbf16>, tensor<1x128x32x32xbf16>) -> tensor<1x128x32x64xbf16>
    return %0 : tensor<1x128x32x64xbf16>
  }

  func.func @concat_2(%arg0: tensor<1x1x32x32xbf16>, %arg1: tensor<1x1x32x32xbf16>) -> tensor<1x1x32x64xbf16> {
    %0 = "ttir.concat"(%arg0, %arg1) <{dim = 3 : si32}> : (tensor<1x1x32x32xbf16>, tensor<1x1x32x32xbf16>) -> tensor<1x1x32x64xbf16>
    return %0 : tensor<1x1x32x64xbf16>
  }
}
