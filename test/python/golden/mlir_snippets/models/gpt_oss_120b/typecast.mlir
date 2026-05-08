module {
  func.func @typecast_0(%arg0: tensor<1x1x360xbf16>) -> tensor<1x1x360xf32> {
    %0 = "ttir.typecast"(%arg0) <{conservative_folding = false}> : (tensor<1x1x360xbf16>) -> tensor<1x1x360xf32>
    return %0 : tensor<1x1x360xf32>
  }
  func.func @typecast_1(%arg0: tensor<1x360xbf16>) -> tensor<1x360xf32> {
    %0 = "ttir.typecast"(%arg0) <{conservative_folding = false}> : (tensor<1x360xbf16>) -> tensor<1x360xf32>
    return %0 : tensor<1x360xf32>
  }
  func.func @typecast_2(%arg0: tensor<1x128x32xf32>) -> tensor<1x128x32xbf16> {
    %0 = "ttir.typecast"(%arg0) <{conservative_folding = false}> : (tensor<1x128x32xf32>) -> tensor<1x128x32xbf16>
    return %0 : tensor<1x128x32xbf16>
  }
  func.func @typecast_3(%arg0: tensor<1x128xi64>) -> tensor<1x128xui32> {
    %0 = "ttir.typecast"(%arg0) <{conservative_folding = false}> : (tensor<1x128xi64>) -> tensor<1x128xui32>
    return %0 : tensor<1x128xui32>
  }
  func.func @typecast_4(%arg0: tensor<128x360xbf16>) -> tensor<128x360xf32> {
    %0 = "ttir.typecast"(%arg0) <{conservative_folding = false}> : (tensor<128x360xbf16>) -> tensor<128x360xf32>
    return %0 : tensor<128x360xf32>
  }
  func.func @typecast_5(%arg0: tensor<128x360xf32>) -> tensor<128x360xbf16> {
    %0 = "ttir.typecast"(%arg0) <{conservative_folding = false}> : (tensor<128x360xf32>) -> tensor<128x360xbf16>
    return %0 : tensor<128x360xbf16>
  }
  func.func @typecast_6(%arg0: tensor<1x128x360xbf16>) -> tensor<1x128x360xf32> {
    %0 = "ttir.typecast"(%arg0) <{conservative_folding = false}> : (tensor<1x128x360xbf16>) -> tensor<1x128x360xf32>
    return %0 : tensor<1x128x360xf32>
  }
  func.func @typecast_7(%arg0: tensor<1x128x360xf32>) -> tensor<1x128x360xbf16> {
    %0 = "ttir.typecast"(%arg0) <{conservative_folding = false}> : (tensor<1x128x360xf32>) -> tensor<1x128x360xbf16>
    return %0 : tensor<1x128x360xbf16>
  }
  func.func @typecast_8(%arg0: tensor<128x4xi32>) -> tensor<128x4xi64> {
    %0 = "ttir.typecast"(%arg0) <{conservative_folding = false}> : (tensor<128x4xi32>) -> tensor<128x4xi64>
    return %0 : tensor<128x4xi64>
  }
}
