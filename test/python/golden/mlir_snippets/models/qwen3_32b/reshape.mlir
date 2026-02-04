module {
  func.func @reshape_0(%arg0: tensor<5120xbf16>) -> tensor<1x5120xbf16> {
    %0 = "ttir.reshape"(%arg0) <{shape = [1 : i32, 5120 : i32]}> : (tensor<5120xbf16>) -> tensor<1x5120xbf16>
    return %0 : tensor<1x5120xbf16>
  }

  func.func @reshape_1(%arg0: tensor<544x1xf32>) -> tensor<32x17xf32> {
    %0 = "ttir.reshape"(%arg0) <{shape = [32 : i32, 17 : i32]}> : (tensor<544x1xf32>) -> tensor<32x17xf32>
    return %0 : tensor<32x17xf32>
  }

  func.func @reshape_2(%arg0: tensor<1x17x64xf32>) -> tensor<1x1x17x64xf32> {
    %0 = "ttir.reshape"(%arg0) <{shape = [1 : i32, 1 : i32, 17 : i32, 64 : i32]}> : (tensor<1x17x64xf32>) -> tensor<1x1x17x64xf32>
    return %0 : tensor<1x1x17x64xf32>
  }
}
