module {
  func.func @reshape_0(%arg0: tensor<7168xbf16>) -> tensor<1x7168xbf16> {
    %0 = "ttir.reshape"(%arg0) <{shape = [1 : i32, 7168 : i32]}> : (tensor<7168xbf16>) -> tensor<1x7168xbf16>
    return %0 : tensor<1x7168xbf16>
  }

  func.func @reshape_1(%arg0: tensor<1536xbf16>) -> tensor<1x1536xbf16> {
    %0 = "ttir.reshape"(%arg0) <{shape = [1 : i32, 1536 : i32]}> : (tensor<1536xbf16>) -> tensor<1x1536xbf16>
    return %0 : tensor<1x1536xbf16>
  }

  func.func @reshape_2(%arg0: tensor<1x32xf32>) -> tensor<32x1xf32> {
    %0 = "ttir.reshape"(%arg0) <{shape = [32 : i32, 1 : i32]}> : (tensor<1x32xf32>) -> tensor<32x1xf32>
    return %0 : tensor<32x1xf32>
  }
}
