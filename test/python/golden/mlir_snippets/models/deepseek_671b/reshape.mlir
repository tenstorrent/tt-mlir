module {
  func.func @reshape_0(%arg0: tensor<2048xbf16>) -> tensor<1x2048xbf16> {
    %0 = "ttir.reshape"(%arg0) <{shape = [1 : i32, 2048 : i32]}> : (tensor<2048xbf16>) -> tensor<1x2048xbf16>
    return %0 : tensor<1x2048xbf16>
  }

  func.func @reshape_1(%arg0: tensor<3072xbf16>) -> tensor<1x3072xbf16> {
    %0 = "ttir.reshape"(%arg0) <{shape = [1 : i32, 3072 : i32]}> : (tensor<3072xbf16>) -> tensor<1x3072xbf16>
    return %0 : tensor<1x3072xbf16>
  }

  func.func @reshape_2(%arg0: tensor<1x32xf32>) -> tensor<32x1xf32> {
    %0 = "ttir.reshape"(%arg0) <{shape = [32 : i32, 1 : i32]}> : (tensor<1x32xf32>) -> tensor<32x1xf32>
    return %0 : tensor<32x1xf32>
  }
}
