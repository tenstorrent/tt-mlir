module {
  func.func @typecast_0(%arg0: tensor<32xf32>) -> tensor<32xf64> {
    %0 = "ttir.typecast"(%arg0) <{conservative_folding = false}> : (tensor<32xf32>) -> tensor<32xf64>
    return %0 : tensor<32xf64>
  }
  func.func @typecast_1(%arg0: tensor<32x32xf64>) -> tensor<32x32xf32> {
    %0 = "ttir.typecast"(%arg0) <{conservative_folding = false}> : (tensor<32x32xf64>) -> tensor<32x32xf32>
    return %0 : tensor<32x32xf32>
  }
  func.func @typecast_2(%arg0: tensor<32x64xf32>) -> tensor<32x64xbf16> {
    %0 = "ttir.typecast"(%arg0) <{conservative_folding = false}> : (tensor<32x64xf32>) -> tensor<32x64xbf16>
    return %0 : tensor<32x64xbf16>
  }
  func.func @typecast_3(%arg0: tensor<64x32x7168xbf16>) -> tensor<64x32x7168xf32> {
    %0 = "ttir.typecast"(%arg0) <{conservative_folding = false}> : (tensor<64x32x7168xbf16>) -> tensor<64x32x7168xf32>
    return %0 : tensor<64x32x7168xf32>
  }
  func.func @typecast_4(%arg0: tensor<2048x7168xf32>) -> tensor<2048x7168xbf16> {
    %0 = "ttir.typecast"(%arg0) <{conservative_folding = false}> : (tensor<2048x7168xf32>) -> tensor<2048x7168xbf16>
    return %0 : tensor<2048x7168xbf16>
  }
  func.func @typecast_5(%arg0: tensor<64x64x32x32xbf16>) -> tensor<64x64x32x32xf32> {
    %0 = "ttir.typecast"(%arg0) <{conservative_folding = false}> : (tensor<64x64x32x32xbf16>) -> tensor<64x64x32x32xf32>
    return %0 : tensor<64x64x32x32xf32>
  }
  func.func @typecast_6(%arg0: tensor<64x64x32x32xf32>) -> tensor<64x64x32x32xbf16> {
    %0 = "ttir.typecast"(%arg0) <{conservative_folding = false}> : (tensor<64x64x32x32xf32>) -> tensor<64x64x32x32xbf16>
    return %0 : tensor<64x64x32x32xbf16>
  }
  func.func @typecast_7(%arg0: tensor<2048x18432xbf16>) -> tensor<2048x18432xf32> {
    %0 = "ttir.typecast"(%arg0) <{conservative_folding = false}> : (tensor<2048x18432xbf16>) -> tensor<2048x18432xf32>
    return %0 : tensor<2048x18432xf32>
  }
  func.func @typecast_8(%arg0: tensor<2048x18432xf32>) -> tensor<2048x18432xbf16> {
    %0 = "ttir.typecast"(%arg0) <{conservative_folding = false}> : (tensor<2048x18432xf32>) -> tensor<2048x18432xbf16>
    return %0 : tensor<2048x18432xbf16>
  }
  func.func @typecast_9(%arg0: tensor<64x32x7168xf32>) -> tensor<64x32x7168xbf16> {
    %0 = "ttir.typecast"(%arg0) <{conservative_folding = false}> : (tensor<64x32x7168xf32>) -> tensor<64x32x7168xbf16>
    return %0 : tensor<64x32x7168xbf16>
  }
}
