module {
  func.func @where_0(%arg0: tensor<16384xi1>, %arg1: tensor<16384xi64>, %arg2: tensor<16384xi64>) -> tensor<16384xi64> {
    %0 = "ttir.where"(%arg0, %arg1, %arg2) : (tensor<16384xi1>, tensor<16384xi64>, tensor<16384xi64>) -> tensor<16384xi64>
    return %0 : tensor<16384xi64>
  }

  func.func @where_1(%arg0: tensor<1x16384x1xi1>, %arg1: tensor<1x16384x512xbf16>, %arg2: tensor<1x16384x512xbf16>) -> tensor<1x16384x512xbf16> {
    %0 = "ttir.where"(%arg0, %arg1, %arg2) : (tensor<1x16384x1xi1>, tensor<1x16384x512xbf16>, tensor<1x16384x512xbf16>) -> tensor<1x16384x512xbf16>
    return %0 : tensor<1x16384x512xbf16>
  }

  func.func @where_2(%arg0: tensor<8xi1>, %arg1: tensor<8xi64>, %arg2: tensor<8xi64>) -> tensor<8xi64> {
    %0 = "ttir.where"(%arg0, %arg1, %arg2) : (tensor<8xi1>, tensor<8xi64>, tensor<8xi64>) -> tensor<8xi64>
    return %0 : tensor<8xi64>
  }

  func.func @where_3(%arg0: tensor<8x1x1xi1>, %arg1: tensor<8x16384x512xbf16>, %arg2: tensor<8x16384x512xbf16>) -> tensor<8x16384x512xbf16> {
    %0 = "ttir.where"(%arg0, %arg1, %arg2) : (tensor<8x1x1xi1>, tensor<8x16384x512xbf16>, tensor<8x16384x512xbf16>) -> tensor<8x16384x512xbf16>
    return %0 : tensor<8x16384x512xbf16>
  }

  func.func @where_4(%arg0: tensor<1x16384x1xi1>, %arg1: tensor<1x16384x64xbf16>, %arg2: tensor<1x16384x64xbf16>) -> tensor<1x16384x64xbf16> {
    %0 = "ttir.where"(%arg0, %arg1, %arg2) : (tensor<1x16384x1xi1>, tensor<1x16384x64xbf16>, tensor<1x16384x64xbf16>) -> tensor<1x16384x64xbf16>
    return %0 : tensor<1x16384x64xbf16>
  }

  func.func @where_5(%arg0: tensor<8x1x1xi1>, %arg1: tensor<8x16384x64xbf16>, %arg2: tensor<8x16384x64xbf16>) -> tensor<8x16384x64xbf16> {
    %0 = "ttir.where"(%arg0, %arg1, %arg2) : (tensor<8x1x1xi1>, tensor<8x16384x64xbf16>, tensor<8x16384x64xbf16>) -> tensor<8x16384x64xbf16>
    return %0 : tensor<8x16384x64xbf16>
  }

  func.func @where_6(%arg0: tensor<1x1x32xi1>, %arg1: tensor<1x32x32xi64>, %arg2: tensor<1x32x32xi64>) -> tensor<1x32x32xi64> {
    %0 = "ttir.where"(%arg0, %arg1, %arg2) : (tensor<1x1x32xi1>, tensor<1x32x32xi64>, tensor<1x32x32xi64>) -> tensor<1x32x32xi64>
    return %0 : tensor<1x32x32xi64>
  }

  func.func @where_7(%arg0: tensor<32x32xi1>, %arg1: tensor<32x32xf32>, %arg2: tensor<32x32xf32>) -> tensor<32x32xf32> {
    %0 = "ttir.where"(%arg0, %arg1, %arg2) : (tensor<32x32xi1>, tensor<32x32xf32>, tensor<32x32xf32>) -> tensor<32x32xf32>
    return %0 : tensor<32x32xf32>
  }
}
