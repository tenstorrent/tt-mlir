module {
  func.func @typecast_0(%arg0: tensor<1x1x32x32xbf16>) -> tensor<1x1x32x32xf32> {
    %0 = "ttir.typecast"(%arg0) <{conservative_folding = false}> : (tensor<1x1x32x32xbf16>) -> tensor<1x1x32x32xf32>
    return %0 : tensor<1x1x32x32xf32>
  }
  func.func @typecast_1(%arg0: tensor<1x32x1xbf16>) -> tensor<1x32x1xf32> {
    %0 = "ttir.typecast"(%arg0) <{conservative_folding = false}> : (tensor<1x32x1xbf16>) -> tensor<1x32x1xf32>
    return %0 : tensor<1x32x1xf32>
  }
  func.func @typecast_2(%arg0: tensor<1x1x32x64xf32>) -> tensor<1x1x32x64xbf16> {
    %0 = "ttir.typecast"(%arg0) <{conservative_folding = false}> : (tensor<1x1x32x64xf32>) -> tensor<1x1x32x64xbf16>
    return %0 : tensor<1x1x32x64xbf16>
  }
  func.func @typecast_3(%arg0: tensor<1x32xi64>) -> tensor<1x32xui32> {
    %0 = "ttir.typecast"(%arg0) <{conservative_folding = false}> : (tensor<1x32xi64>) -> tensor<1x32xui32>
    return %0 : tensor<1x32xui32>
  }
  func.func @typecast_4(%arg0: tensor<32x5120xbf16>) -> tensor<32x5120xf32> {
    %0 = "ttir.typecast"(%arg0) <{conservative_folding = false}> : (tensor<32x5120xbf16>) -> tensor<32x5120xf32>
    return %0 : tensor<32x5120xf32>
  }
  func.func @typecast_5(%arg0: tensor<32x5120xf32>) -> tensor<32x5120xbf16> {
    %0 = "ttir.typecast"(%arg0) <{conservative_folding = false}> : (tensor<32x5120xf32>) -> tensor<32x5120xbf16>
    return %0 : tensor<32x5120xbf16>
  }
  func.func @typecast_6(%arg0: tensor<32x1024xbf16>) -> tensor<32x1024xf32> {
    %0 = "ttir.typecast"(%arg0) <{conservative_folding = false}> : (tensor<32x1024xbf16>) -> tensor<32x1024xf32>
    return %0 : tensor<32x1024xf32>
  }
  func.func @typecast_7(%arg0: tensor<1x8x32x128xf32>) -> tensor<1x8x32x128xbf16> {
    %0 = "ttir.typecast"(%arg0) <{conservative_folding = false}> : (tensor<1x8x32x128xf32>) -> tensor<1x8x32x128xbf16>
    return %0 : tensor<1x8x32x128xbf16>
  }
  func.func @typecast_8(%arg0: tensor<32x12288xbf16>) -> tensor<32x12288xf32> {
    %0 = "ttir.typecast"(%arg0) <{conservative_folding = false}> : (tensor<32x12288xbf16>) -> tensor<32x12288xf32>
    return %0 : tensor<32x12288xf32>
  }
  func.func @typecast_9(%arg0: tensor<1x96x32x128xf32>) -> tensor<1x96x32x128xbf16> {
    %0 = "ttir.typecast"(%arg0) <{conservative_folding = false}> : (tensor<1x96x32x128xf32>) -> tensor<1x96x32x128xbf16>
    return %0 : tensor<1x96x32x128xbf16>
  }
  func.func @typecast_10(%arg0: tensor<1x96x32x128xbf16>) -> tensor<1x96x32x128xf32> {
    %0 = "ttir.typecast"(%arg0) <{conservative_folding = false}> : (tensor<1x96x32x128xbf16>) -> tensor<1x96x32x128xf32>
    return %0 : tensor<1x96x32x128xf32>
  }
  func.func @typecast_11(%arg0: tensor<1x8x1x32x128xbf16>) -> tensor<1x8x1x32x128xf32> {
    %0 = "ttir.typecast"(%arg0) <{conservative_folding = false}> : (tensor<1x8x1x32x128xbf16>) -> tensor<1x8x1x32x128xf32>
    return %0 : tensor<1x8x1x32x128xf32>
  }
  func.func @typecast_12(%arg0: tensor<1x96x32x32xf32>) -> tensor<1x96x32x32xf64> {
    %0 = "ttir.typecast"(%arg0) <{conservative_folding = false}> : (tensor<1x96x32x32xf32>) -> tensor<1x96x32x32xf64>
    return %0 : tensor<1x96x32x32xf64>
  }
}
