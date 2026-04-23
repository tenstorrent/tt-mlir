module {
  func.func @typecast_0(%arg0: tensor<1x3072xbf16>) -> tensor<1x3072xf32> {
    %0 = "ttir.typecast"(%arg0) <{conservative_folding = false}> : (tensor<1x3072xbf16>) -> tensor<1x3072xf32>
    return %0 : tensor<1x3072xf32>
  }
  func.func @typecast_1(%arg0: tensor<8xf32>) -> tensor<8xi64> {
    %0 = "ttir.typecast"(%arg0) <{conservative_folding = false}> : (tensor<8xf32>) -> tensor<8xi64>
    return %0 : tensor<8xi64>
  }
  func.func @typecast_2(%arg0: tensor<1x2048xbf16>) -> tensor<1x2048xf32> {
    %0 = "ttir.typecast"(%arg0) <{conservative_folding = false}> : (tensor<1x2048xbf16>) -> tensor<1x2048xf32>
    return %0 : tensor<1x2048xf32>
  }
  func.func @typecast_3(%arg0: tensor<1x1x512xbf16>) -> tensor<1x1x512xf32> {
    %0 = "ttir.typecast"(%arg0) <{conservative_folding = false}> : (tensor<1x1x512xbf16>) -> tensor<1x1x512xf32>
    return %0 : tensor<1x1x512xf32>
  }
  func.func @typecast_4(%arg0: tensor<16384xf32>) -> tensor<16384xi64> {
    %0 = "ttir.typecast"(%arg0) <{conservative_folding = false}> : (tensor<16384xf32>) -> tensor<16384xi64>
    return %0 : tensor<16384xi64>
  }
  func.func @typecast_5(%arg0: tensor<1x32x1x32x1xbf16>) -> tensor<1x32x1x32x1xf32> {
    %0 = "ttir.typecast"(%arg0) <{conservative_folding = false}> : (tensor<1x32x1x32x1xbf16>) -> tensor<1x32x1x32x1xf32>
    return %0 : tensor<1x32x1x32x1xf32>
  }
  func.func @typecast_6(%arg0: tensor<1x32xi64>) -> tensor<1x32xui32> {
    %0 = "ttir.typecast"(%arg0) <{conservative_folding = false}> : (tensor<1x32xi64>) -> tensor<1x32xui32>
    return %0 : tensor<1x32xui32>
  }
  func.func @typecast_7(%arg0: tensor<32x2048xbf16>) -> tensor<32x2048xf32> {
    %0 = "ttir.typecast"(%arg0) <{conservative_folding = false}> : (tensor<32x2048xbf16>) -> tensor<32x2048xf32>
    return %0 : tensor<32x2048xf32>
  }
  func.func @typecast_8(%arg0: tensor<32x2048xf32>) -> tensor<32x2048xbf16> {
    %0 = "ttir.typecast"(%arg0) <{conservative_folding = false}> : (tensor<32x2048xf32>) -> tensor<32x2048xbf16>
    return %0 : tensor<32x2048xbf16>
  }
  func.func @typecast_9(%arg0: tensor<1x32x512xbf16>) -> tensor<1x32x512xf32> {
    %0 = "ttir.typecast"(%arg0) <{conservative_folding = false}> : (tensor<1x32x512xbf16>) -> tensor<1x32x512xf32>
    return %0 : tensor<1x32x512xf32>
  }
  func.func @typecast_10(%arg0: tensor<1x32x512xf32>) -> tensor<1x32x512xbf16> {
    %0 = "ttir.typecast"(%arg0) <{conservative_folding = false}> : (tensor<1x32x512xf32>) -> tensor<1x32x512xbf16>
    return %0 : tensor<1x32x512xbf16>
  }
  func.func @typecast_11(%arg0: tensor<1x32x64xbf16>) -> tensor<1x32x64xf32> {
    %0 = "ttir.typecast"(%arg0) <{conservative_folding = false}> : (tensor<1x32x64xbf16>) -> tensor<1x32x64xf32>
    return %0 : tensor<1x32x64xf32>
  }
  func.func @typecast_12(%arg0: tensor<1x32x1x32x2xf32>) -> tensor<1x32x1x32x2xbf16> {
    %0 = "ttir.typecast"(%arg0) <{conservative_folding = false}> : (tensor<1x32x1x32x2xf32>) -> tensor<1x32x1x32x2xbf16>
    return %0 : tensor<1x32x1x32x2xbf16>
  }
  func.func @typecast_13(%arg0: tensor<32x3072xbf16>) -> tensor<32x3072xf32> {
    %0 = "ttir.typecast"(%arg0) <{conservative_folding = false}> : (tensor<32x3072xbf16>) -> tensor<32x3072xf32>
    return %0 : tensor<32x3072xf32>
  }
  func.func @typecast_14(%arg0: tensor<32x3072xf32>) -> tensor<32x3072xbf16> {
    %0 = "ttir.typecast"(%arg0) <{conservative_folding = false}> : (tensor<32x3072xf32>) -> tensor<32x3072xbf16>
    return %0 : tensor<32x3072xbf16>
  }
  func.func @typecast_15(%arg0: tensor<1x32x16x64xbf16>) -> tensor<1x32x16x64xf32> {
    %0 = "ttir.typecast"(%arg0) <{conservative_folding = false}> : (tensor<1x32x16x64xbf16>) -> tensor<1x32x16x64xf32>
    return %0 : tensor<1x32x16x64xf32>
  }
  func.func @typecast_16(%arg0: tensor<1x32x16x32x2xf32>) -> tensor<1x32x16x32x2xbf16> {
    %0 = "ttir.typecast"(%arg0) <{conservative_folding = false}> : (tensor<1x32x16x32x2xf32>) -> tensor<1x32x16x32x2xbf16>
    return %0 : tensor<1x32x16x32x2xbf16>
  }
  func.func @typecast_17(%arg0: tensor<1x32x16x32xbf16>) -> tensor<1x32x16x32xf32> {
    %0 = "ttir.typecast"(%arg0) <{conservative_folding = false}> : (tensor<1x32x16x32xbf16>) -> tensor<1x32x16x32xf32>
    return %0 : tensor<1x32x16x32xf32>
  }
  func.func @typecast_18(%arg0: tensor<1x32x16x32xf32>) -> tensor<1x32x16x32xbf16> {
    %0 = "ttir.typecast"(%arg0) <{conservative_folding = false}> : (tensor<1x32x16x32xf32>) -> tensor<1x32x16x32xbf16>
    return %0 : tensor<1x32x16x32xbf16>
  }
  func.func @typecast_19(%arg0: tensor<32x10944xbf16>) -> tensor<32x10944xf32> {
    %0 = "ttir.typecast"(%arg0) <{conservative_folding = false}> : (tensor<32x10944xbf16>) -> tensor<32x10944xf32>
    return %0 : tensor<32x10944xf32>
  }
  func.func @typecast_20(%arg0: tensor<32x10944xf32>) -> tensor<32x10944xbf16> {
    %0 = "ttir.typecast"(%arg0) <{conservative_folding = false}> : (tensor<32x10944xf32>) -> tensor<32x10944xbf16>
    return %0 : tensor<32x10944xbf16>
  }
  func.func @typecast_21(%arg0: tensor<1x32x2048xf32>) -> tensor<1x32x2048xbf16> {
    %0 = "ttir.typecast"(%arg0) <{conservative_folding = false}> : (tensor<1x32x2048xf32>) -> tensor<1x32x2048xbf16>
    return %0 : tensor<1x32x2048xbf16>
  }
  func.func @typecast_22(%arg0: tensor<1x32x2048xbf16>) -> tensor<1x32x2048xf32> {
    %0 = "ttir.typecast"(%arg0) <{conservative_folding = false}> : (tensor<1x32x2048xbf16>) -> tensor<1x32x2048xf32>
    return %0 : tensor<1x32x2048xf32>
  }
}
