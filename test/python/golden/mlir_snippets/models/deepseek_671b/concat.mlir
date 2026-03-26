module {
  func.func @concat_0(%arg0: tensor<1x32x1x32x1xf32>, %arg1: tensor<1x32x1x32x1xf32>) -> tensor<1x32x1x32x2xf32> {
    %0 = "ttir.concat"(%arg0, %arg1) <{dim = 4 : si32}> : (tensor<1x32x1x32x1xf32>, tensor<1x32x1x32x1xf32>) -> tensor<1x32x1x32x2xf32>
    return %0 : tensor<1x32x1x32x2xf32>
  }

  func.func @concat_1(%arg0: tensor<1x32x16x32x1xf32>, %arg1: tensor<1x32x16x32x1xf32>) -> tensor<1x32x16x32x2xf32> {
    %0 = "ttir.concat"(%arg0, %arg1) <{dim = 4 : si32}> : (tensor<1x32x16x32x1xf32>, tensor<1x32x16x32x1xf32>) -> tensor<1x32x16x32x2xf32>
    return %0 : tensor<1x32x16x32x2xf32>
  }

  func.func @concat_2(%arg0: tensor<1x32x16x128xbf16>, %arg1: tensor<1x32x16x64xbf16>) -> tensor<1x32x16x192xbf16> {
    %0 = "ttir.concat"(%arg0, %arg1) <{dim = 3 : si32}> : (tensor<1x32x16x128xbf16>, tensor<1x32x16x64xbf16>) -> tensor<1x32x16x192xbf16>
    return %0 : tensor<1x32x16x192xbf16>
  }

  func.func @concat_3(%arg0: tensor<1024x1xi64>, %arg1: tensor<1024x1xi64>, %arg2: tensor<1024x1xi64>) -> tensor<1024x3xi64> {
    %0 = "ttir.concat"(%arg0, %arg1, %arg2) <{dim = 1 : si32}> : (tensor<1024x1xi64>, tensor<1024x1xi64>, tensor<1024x1xi64>) -> tensor<1024x3xi64>
    return %0 : tensor<1024x3xi64>
  }
}
