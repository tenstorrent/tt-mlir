module {
  func.func @reshape_0(%arg0: tensor<5120xbf16>) -> tensor<1x5120xbf16> {
    %0 = "ttir.reshape"(%arg0) <{shape = [1 : i32, 5120 : i32]}> : (tensor<5120xbf16>) -> tensor<1x5120xbf16>
    return %0 : tensor<1x5120xbf16>
  }

  func.func @reshape_1(%arg0: tensor<1024xbf16>) -> tensor<1x1x8x128xbf16> {
    %0 = "ttir.reshape"(%arg0) <{shape = [1 : i32, 1 : i32, 8 : i32, 128 : i32]}> : (tensor<1024xbf16>) -> tensor<1x1x8x128xbf16>
    return %0 : tensor<1x1x8x128xbf16>
  }

  func.func @reshape_2(%arg0: tensor<1x32xf32>) -> tensor<32x1xf32> {
    %0 = "ttir.reshape"(%arg0) <{shape = [32 : i32, 1 : i32]}> : (tensor<1x32xf32>) -> tensor<32x1xf32>
    return %0 : tensor<32x1xf32>
  }
}
