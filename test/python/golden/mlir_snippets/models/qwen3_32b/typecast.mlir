module {
  func.func @typecast_0(%arg0: tensor<1x64x1xbf16>) -> tensor<1x64x1xf32> {
    %0 = "ttir.typecast"(%arg0) <{conservative_folding = false}> : (tensor<1x64x1xbf16>) -> tensor<1x64x1xf32>
    return %0 : tensor<1x64x1xf32>
  }
  func.func @typecast_1(%arg0: tensor<32x17xi64>) -> tensor<32x17xui32> {
    %0 = "ttir.typecast"(%arg0) <{conservative_folding = false}> : (tensor<32x17xi64>) -> tensor<32x17xui32>
    return %0 : tensor<32x17xui32>
  }
  func.func @typecast_2(%arg0: tensor<544x5120xbf16>) -> tensor<544x5120xf32> {
    %0 = "ttir.typecast"(%arg0) <{conservative_folding = false}> : (tensor<544x5120xbf16>) -> tensor<544x5120xf32>
    return %0 : tensor<544x5120xf32>
  }
  func.func @typecast_3(%arg0: tensor<544x5120xf32>) -> tensor<544x5120xbf16> {
    %0 = "ttir.typecast"(%arg0) <{conservative_folding = false}> : (tensor<544x5120xf32>) -> tensor<544x5120xbf16>
    return %0 : tensor<544x5120xbf16>
  }
  func.func @typecast_4(%arg0: tensor<544x128xbf16>) -> tensor<544x128xf32> {
    %0 = "ttir.typecast"(%arg0) <{conservative_folding = false}> : (tensor<544x128xbf16>) -> tensor<544x128xf32>
    return %0 : tensor<544x128xf32>
  }
  func.func @typecast_5(%arg0: tensor<32x1x17x128xf32>) -> tensor<32x1x17x128xbf16> {
    %0 = "ttir.typecast"(%arg0) <{conservative_folding = false}> : (tensor<32x1x17x128xf32>) -> tensor<32x1x17x128xbf16>
    return %0 : tensor<32x1x17x128xbf16>
  }
  func.func @typecast_6(%arg0: tensor<17xi64>) -> tensor<17xf32> {
    %0 = "ttir.typecast"(%arg0) <{conservative_folding = false}> : (tensor<17xi64>) -> tensor<17xf32>
    return %0 : tensor<17xf32>
  }
  func.func @typecast_7(%arg0: tensor<1x1x17x128xf32>) -> tensor<1x1x17x128xbf16> {
    %0 = "ttir.typecast"(%arg0) <{conservative_folding = false}> : (tensor<1x1x17x128xf32>) -> tensor<1x1x17x128xbf16>
    return %0 : tensor<1x1x17x128xbf16>
  }
  func.func @typecast_8(%arg0: tensor<544x1024xbf16>) -> tensor<544x1024xf32> {
    %0 = "ttir.typecast"(%arg0) <{conservative_folding = false}> : (tensor<544x1024xbf16>) -> tensor<544x1024xf32>
    return %0 : tensor<544x1024xf32>
  }
  func.func @typecast_9(%arg0: tensor<32x8x17x128xf32>) -> tensor<32x8x17x128xbf16> {
    %0 = "ttir.typecast"(%arg0) <{conservative_folding = false}> : (tensor<32x8x17x128xf32>) -> tensor<32x8x17x128xbf16>
    return %0 : tensor<32x8x17x128xbf16>
  }
  func.func @typecast_10(%arg0: tensor<32x8x17x128xbf16>) -> tensor<32x8x17x128xf32> {
    %0 = "ttir.typecast"(%arg0) <{conservative_folding = false}> : (tensor<32x8x17x128xbf16>) -> tensor<32x8x17x128xf32>
    return %0 : tensor<32x8x17x128xf32>
  }
  func.func @typecast_11(%arg0: tensor<32x1x128x128xbf16>) -> tensor<32x1x128x128xf32> {
    %0 = "ttir.typecast"(%arg0) <{conservative_folding = false}> : (tensor<32x1x128x128xbf16>) -> tensor<32x1x128x128xf32>
    return %0 : tensor<32x1x128x128xf32>
  }
  func.func @typecast_12(%arg0: tensor<1x1x17x128xbf16>) -> tensor<1x1x17x128xf32> {
    %0 = "ttir.typecast"(%arg0) <{conservative_folding = false}> : (tensor<1x1x17x128xbf16>) -> tensor<1x1x17x128xf32>
    return %0 : tensor<1x1x17x128xf32>
  }
  func.func @typecast_13(%arg0: tensor<32x8x17x128xf32>) -> tensor<32x8x17x128xf64> {
    %0 = "ttir.typecast"(%arg0) <{conservative_folding = false}> : (tensor<32x8x17x128xf32>) -> tensor<32x8x17x128xf64>
    return %0 : tensor<32x8x17x128xf64>
  }
}
