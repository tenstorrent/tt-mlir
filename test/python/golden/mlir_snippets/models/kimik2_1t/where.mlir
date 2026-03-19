module {
  func.func @where_0(%arg0: tensor<192xi1>, %arg1: tensor<192xi64>, %arg2: tensor<192xi64>) -> tensor<192xi64> {
    %0 = "ttir.where"(%arg0, %arg1, %arg2) : (tensor<192xi1>, tensor<192xi64>, tensor<192xi64>) -> tensor<192xi64>
    return %0 : tensor<192xi64>
  }

  func.func @where_1(%arg0: tensor<1x1x1x192xi1>, %arg1: tensor<1x128x32x192xbf16>, %arg2: tensor<1x128x32x192xbf16>) -> tensor<1x128x32x192xbf16> {
    %0 = "ttir.where"(%arg0, %arg1, %arg2) : (tensor<1x1x1x192xi1>, tensor<1x128x32x192xbf16>, tensor<1x128x32x192xbf16>) -> tensor<1x128x32x192xbf16>
    return %0 : tensor<1x128x32x192xbf16>
  }

  func.func @where_2(%arg0: tensor<32x32xi1>, %arg1: tensor<32x32xf32>, %arg2: tensor<32x32xf32>) -> tensor<32x32xf32> {
    %0 = "ttir.where"(%arg0, %arg1, %arg2) : (tensor<32x32xi1>, tensor<32x32xf32>, tensor<32x32xf32>) -> tensor<32x32xf32>
    return %0 : tensor<32x32xf32>
  }
}
