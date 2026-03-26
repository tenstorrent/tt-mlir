module {
  func.func @concat_0(%arg0: tensor<1x16x128x32xbf16>, %arg1: tensor<1x16x128x32xbf16>) -> tensor<1x16x128x64xbf16> {
    %0 = "ttir.concat"(%arg0, %arg1) <{dim = 3 : si32}> : (tensor<1x16x128x32xbf16>, tensor<1x16x128x32xbf16>) -> tensor<1x16x128x64xbf16>
    return %0 : tensor<1x16x128x64xbf16>
  }

  func.func @concat_1(%arg0: tensor<1x2x1x128x32xbf16>, %arg1: tensor<1x2x1x128x32xbf16>) -> tensor<1x2x1x128x64xbf16> {
    %0 = "ttir.concat"(%arg0, %arg1) <{dim = 4 : si32}> : (tensor<1x2x1x128x32xbf16>, tensor<1x2x1x128x32xbf16>) -> tensor<1x2x1x128x64xbf16>
    return %0 : tensor<1x2x1x128x64xbf16>
  }

  func.func @concat_2(%arg0: tensor<1x16x128x128xbf16>, %arg1: tensor<1x16x128x1xbf16>) -> tensor<1x16x128x129xbf16> {
    %0 = "ttir.concat"(%arg0, %arg1) <{dim = 3 : si32}> : (tensor<1x16x128x128xbf16>, tensor<1x16x128x1xbf16>) -> tensor<1x16x128x129xbf16>
    return %0 : tensor<1x16x128x129xbf16>
  }

  func.func @concat_3(%arg0: tensor<128x4x1xi64>, %arg1: tensor<128x4x1xi64>) -> tensor<128x4x2xi64> {
    %0 = "ttir.concat"(%arg0, %arg1) <{dim = 2 : si32}> : (tensor<128x4x1xi64>, tensor<128x4x1xi64>) -> tensor<128x4x2xi64>
    return %0 : tensor<128x4x2xi64>
  }
}
