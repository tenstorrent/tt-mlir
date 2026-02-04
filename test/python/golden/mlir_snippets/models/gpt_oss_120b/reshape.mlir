module {
  func.func @reshape_0(%arg0: tensor<360xbf16>) -> tensor<1x360xbf16> {
    %0 = "ttir.reshape"(%arg0) <{shape = [1 : i32, 360 : i32]}> : (tensor<360xbf16>) -> tensor<1x360xbf16>
    return %0 : tensor<1x360xbf16>
  }

  func.func @reshape_1(%arg0: tensor<1x128xf32>) -> tensor<128x1xf32> {
    %0 = "ttir.reshape"(%arg0) <{shape = [128 : i32, 1 : i32]}> : (tensor<1x128xf32>) -> tensor<128x1xf32>
    return %0 : tensor<128x1xf32>
  }

  func.func @reshape_2(%arg0: tensor<1024xbf16>) -> tensor<1x1x16x64xbf16> {
    %0 = "ttir.reshape"(%arg0) <{shape = [1 : i32, 1 : i32, 16 : i32, 64 : i32]}> : (tensor<1024xbf16>) -> tensor<1x1x16x64xbf16>
    return %0 : tensor<1x1x16x64xbf16>
  }
}
