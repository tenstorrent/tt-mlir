module {
  func.func @ttir_typecast_0(%arg0: tensor<1x3072xbf16>) -> tensor<1x3072xf32> {
    %0 = "ttir.typecast"(%arg0) <{conservative_folding = false}> : (tensor<1x3072xbf16>) -> tensor<1x3072xf32>
    return %0 : tensor<1x3072xf32>
  }
  func.func @ttir_typecast_1(%arg0: tensor<8xf32>) -> tensor<8xi64> {
    %0 = "ttir.typecast"(%arg0) <{conservative_folding = false}> : (tensor<8xf32>) -> tensor<8xi64>
    return %0 : tensor<8xi64>
  }
  func.func @ttir_typecast_2(%arg0: tensor<1x2048xbf16>) -> tensor<1x2048xf32> {
    %0 = "ttir.typecast"(%arg0) <{conservative_folding = false}> : (tensor<1x2048xbf16>) -> tensor<1x2048xf32>
    return %0 : tensor<1x2048xf32>
  }
  func.func @ttir_typecast_3(%arg0: tensor<1x1x512xbf16>) -> tensor<1x1x512xf32> {
    %0 = "ttir.typecast"(%arg0) <{conservative_folding = false}> : (tensor<1x1x512xbf16>) -> tensor<1x1x512xf32>
    return %0 : tensor<1x1x512xf32>
  }
  func.func @ttir_typecast_4(%arg0: tensor<16384xf32>) -> tensor<16384xi64> {
    %0 = "ttir.typecast"(%arg0) <{conservative_folding = false}> : (tensor<16384xf32>) -> tensor<16384xi64>
    return %0 : tensor<16384xi64>
  }
  func.func @ttir_typecast_5(%arg0: tensor<1x32x1x32xbf16>) -> tensor<1x32x1x32xf32> {
    %0 = "ttir.typecast"(%arg0) <{conservative_folding = false}> : (tensor<1x32x1x32xbf16>) -> tensor<1x32x1x32xf32>
    return %0 : tensor<1x32x1x32xf32>
  }
  func.func @ttir_typecast_6(%arg0: tensor<1x32xi64>) -> tensor<1x32xui32> {
    %0 = "ttir.typecast"(%arg0) <{conservative_folding = false}> : (tensor<1x32xi64>) -> tensor<1x32xui32>
    return %0 : tensor<1x32xui32>
  }
  func.func @ttir_typecast_7(%arg0: tensor<32x2048xbf16>) -> tensor<32x2048xf32> {
    %0 = "ttir.typecast"(%arg0) <{conservative_folding = false}> : (tensor<32x2048xbf16>) -> tensor<32x2048xf32>
    return %0 : tensor<32x2048xf32>
  }
  func.func @ttir_typecast_8(%arg0: tensor<32x2048xf32>) -> tensor<32x2048xbf16> {
    %0 = "ttir.typecast"(%arg0) <{conservative_folding = false}> : (tensor<32x2048xf32>) -> tensor<32x2048xbf16>
    return %0 : tensor<32x2048xbf16>
  }
  func.func @ttir_typecast_9(%arg0: tensor<1x32x512xbf16>) -> tensor<1x32x512xf32> {
    %0 = "ttir.typecast"(%arg0) <{conservative_folding = false}> : (tensor<1x32x512xbf16>) -> tensor<1x32x512xf32>
    return %0 : tensor<1x32x512xf32>
  }
  func.func @ttir_typecast_10(%arg0: tensor<1x32x512xf32>) -> tensor<1x32x512xbf16> {
    %0 = "ttir.typecast"(%arg0) <{conservative_folding = false}> : (tensor<1x32x512xf32>) -> tensor<1x32x512xbf16>
    return %0 : tensor<1x32x512xbf16>
  }
  func.func @ttir_typecast_11(%arg0: tensor<1x32x64xbf16>) -> tensor<1x32x64xf32> {
    %0 = "ttir.typecast"(%arg0) <{conservative_folding = false}> : (tensor<1x32x64xbf16>) -> tensor<1x32x64xf32>
    return %0 : tensor<1x32x64xf32>
  }
  func.func @ttir_typecast_12(%arg0: tensor<1x32x1x32x2xf32>) -> tensor<1x32x1x32x2xbf16> {
    %0 = "ttir.typecast"(%arg0) <{conservative_folding = false}> : (tensor<1x32x1x32x2xf32>) -> tensor<1x32x1x32x2xbf16>
    return %0 : tensor<1x32x1x32x2xbf16>
  }
  func.func @ttir_typecast_13(%arg0: tensor<32x3072xbf16>) -> tensor<32x3072xf32> {
    %0 = "ttir.typecast"(%arg0) <{conservative_folding = false}> : (tensor<32x3072xbf16>) -> tensor<32x3072xf32>
    return %0 : tensor<32x3072xf32>
  }
  func.func @ttir_typecast_14(%arg0: tensor<32x3072xf32>) -> tensor<32x3072xbf16> {
    %0 = "ttir.typecast"(%arg0) <{conservative_folding = false}> : (tensor<32x3072xf32>) -> tensor<32x3072xbf16>
    return %0 : tensor<32x3072xbf16>
  }
  func.func @ttir_typecast_15(%arg0: tensor<1x32x16x64xbf16>) -> tensor<1x32x16x64xf32> {
    %0 = "ttir.typecast"(%arg0) <{conservative_folding = false}> : (tensor<1x32x16x64xbf16>) -> tensor<1x32x16x64xf32>
    return %0 : tensor<1x32x16x64xf32>
  }
  func.func @ttir_typecast_16(%arg0: tensor<1x32x16x32x2xf32>) -> tensor<1x32x16x32x2xbf16> {
    %0 = "ttir.typecast"(%arg0) <{conservative_folding = false}> : (tensor<1x32x16x32x2xf32>) -> tensor<1x32x16x32x2xbf16>
    return %0 : tensor<1x32x16x32x2xbf16>
  }
  func.func @ttir_typecast_17(%arg0: tensor<1x32x16x32xbf16>) -> tensor<1x32x16x32xf32> {
    %0 = "ttir.typecast"(%arg0) <{conservative_folding = false}> : (tensor<1x32x16x32xbf16>) -> tensor<1x32x16x32xf32>
    return %0 : tensor<1x32x16x32xf32>
  }
  func.func @ttir_typecast_18(%arg0: tensor<1x32x16x32xf32>) -> tensor<1x32x16x32xbf16> {
    %0 = "ttir.typecast"(%arg0) <{conservative_folding = false}> : (tensor<1x32x16x32xf32>) -> tensor<1x32x16x32xbf16>
    return %0 : tensor<1x32x16x32xbf16>
  }
  func.func @ttir_typecast_19(%arg0: tensor<32x10944xbf16>) -> tensor<32x10944xf32> {
    %0 = "ttir.typecast"(%arg0) <{conservative_folding = false}> : (tensor<32x10944xbf16>) -> tensor<32x10944xf32>
    return %0 : tensor<32x10944xf32>
  }
  func.func @ttir_typecast_20(%arg0: tensor<32x10944xf32>) -> tensor<32x10944xbf16> {
    %0 = "ttir.typecast"(%arg0) <{conservative_folding = false}> : (tensor<32x10944xf32>) -> tensor<32x10944xbf16>
    return %0 : tensor<32x10944xbf16>
  }
  func.func @ttir_typecast_21(%arg0: tensor<1x32x2048xf32>) -> tensor<1x32x2048xbf16> {
    %0 = "ttir.typecast"(%arg0) <{conservative_folding = false}> : (tensor<1x32x2048xf32>) -> tensor<1x32x2048xbf16>
    return %0 : tensor<1x32x2048xbf16>
  }
  func.func @ttir_typecast_22(%arg0: tensor<1x32x2048xbf16>) -> tensor<1x32x2048xf32> {
    %0 = "ttir.typecast"(%arg0) <{conservative_folding = false}> : (tensor<1x32x2048xbf16>) -> tensor<1x32x2048xf32>
    return %0 : tensor<1x32x2048xf32>
  }
  func.func @ttir_typecast_23(%arg0: tensor<1x1x32x32xbf16>) -> tensor<1x1x32x32xf32> {
    %0 = "ttir.typecast"(%arg0) <{conservative_folding = false}> : (tensor<1x1x32x32xbf16>) -> tensor<1x1x32x32xf32>
    return %0 : tensor<1x1x32x32xf32>
  }
  func.func @ttir_typecast_24(%arg0: tensor<1x32x1xbf16>) -> tensor<1x32x1xf32> {
    %0 = "ttir.typecast"(%arg0) <{conservative_folding = false}> : (tensor<1x32x1xbf16>) -> tensor<1x32x1xf32>
    return %0 : tensor<1x32x1xf32>
  }
  func.func @ttir_typecast_25(%arg0: tensor<1x1x32x64xf32>) -> tensor<1x1x32x64xbf16> {
    %0 = "ttir.typecast"(%arg0) <{conservative_folding = false}> : (tensor<1x1x32x64xf32>) -> tensor<1x1x32x64xbf16>
    return %0 : tensor<1x1x32x64xbf16>
  }
  func.func @ttir_typecast_26(%arg0: tensor<32x5120xbf16>) -> tensor<32x5120xf32> {
    %0 = "ttir.typecast"(%arg0) <{conservative_folding = false}> : (tensor<32x5120xbf16>) -> tensor<32x5120xf32>
    return %0 : tensor<32x5120xf32>
  }
  func.func @ttir_typecast_27(%arg0: tensor<32x5120xf32>) -> tensor<32x5120xbf16> {
    %0 = "ttir.typecast"(%arg0) <{conservative_folding = false}> : (tensor<32x5120xf32>) -> tensor<32x5120xbf16>
    return %0 : tensor<32x5120xbf16>
  }
  func.func @ttir_typecast_28(%arg0: tensor<32x1024xbf16>) -> tensor<32x1024xf32> {
    %0 = "ttir.typecast"(%arg0) <{conservative_folding = false}> : (tensor<32x1024xbf16>) -> tensor<32x1024xf32>
    return %0 : tensor<32x1024xf32>
  }
  func.func @ttir_typecast_29(%arg0: tensor<1x8x32x128xf32>) -> tensor<1x8x32x128xbf16> {
    %0 = "ttir.typecast"(%arg0) <{conservative_folding = false}> : (tensor<1x8x32x128xf32>) -> tensor<1x8x32x128xbf16>
    return %0 : tensor<1x8x32x128xbf16>
  }
  func.func @ttir_typecast_30(%arg0: tensor<32x12288xbf16>) -> tensor<32x12288xf32> {
    %0 = "ttir.typecast"(%arg0) <{conservative_folding = false}> : (tensor<32x12288xbf16>) -> tensor<32x12288xf32>
    return %0 : tensor<32x12288xf32>
  }
  func.func @ttir_typecast_31(%arg0: tensor<1x96x32x128xf32>) -> tensor<1x96x32x128xbf16> {
    %0 = "ttir.typecast"(%arg0) <{conservative_folding = false}> : (tensor<1x96x32x128xf32>) -> tensor<1x96x32x128xbf16>
    return %0 : tensor<1x96x32x128xbf16>
  }
  func.func @ttir_typecast_32(%arg0: tensor<1x96x32x128xbf16>) -> tensor<1x96x32x128xf32> {
    %0 = "ttir.typecast"(%arg0) <{conservative_folding = false}> : (tensor<1x96x32x128xbf16>) -> tensor<1x96x32x128xf32>
    return %0 : tensor<1x96x32x128xf32>
  }
  func.func @ttir_typecast_33(%arg0: tensor<1x8x1x32x128xbf16>) -> tensor<1x8x1x32x128xf32> {
    %0 = "ttir.typecast"(%arg0) <{conservative_folding = false}> : (tensor<1x8x1x32x128xbf16>) -> tensor<1x8x1x32x128xf32>
    return %0 : tensor<1x8x1x32x128xf32>
  }
  func.func @ttir_typecast_34(%arg0: tensor<1x96x32x32xf32>) -> tensor<1x96x32x32xf64> {
    %0 = "ttir.typecast"(%arg0) <{conservative_folding = false}> : (tensor<1x96x32x32xf32>) -> tensor<1x96x32x32xf64>
    return %0 : tensor<1x96x32x32xf64>
  }
  func.func @ttir_typecast_35(%arg0: tensor<1x1x360xbf16>) -> tensor<1x1x360xf32> {
    %0 = "ttir.typecast"(%arg0) <{conservative_folding = false}> : (tensor<1x1x360xbf16>) -> tensor<1x1x360xf32>
    return %0 : tensor<1x1x360xf32>
  }
  func.func @ttir_typecast_36(%arg0: tensor<1x360xbf16>) -> tensor<1x360xf32> {
    %0 = "ttir.typecast"(%arg0) <{conservative_folding = false}> : (tensor<1x360xbf16>) -> tensor<1x360xf32>
    return %0 : tensor<1x360xf32>
  }
  func.func @ttir_typecast_37(%arg0: tensor<1x1x128x32xf32>) -> tensor<1x1x128x32xbf16> {
    %0 = "ttir.typecast"(%arg0) <{conservative_folding = false}> : (tensor<1x1x128x32xf32>) -> tensor<1x1x128x32xbf16>
    return %0 : tensor<1x1x128x32xbf16>
  }
  func.func @ttir_typecast_38(%arg0: tensor<1x128xi64>) -> tensor<1x128xui32> {
    %0 = "ttir.typecast"(%arg0) <{conservative_folding = false}> : (tensor<1x128xi64>) -> tensor<1x128xui32>
    return %0 : tensor<1x128xui32>
  }
  func.func @ttir_typecast_39(%arg0: tensor<128x360xbf16>) -> tensor<128x360xf32> {
    %0 = "ttir.typecast"(%arg0) <{conservative_folding = false}> : (tensor<128x360xbf16>) -> tensor<128x360xf32>
    return %0 : tensor<128x360xf32>
  }
  func.func @ttir_typecast_40(%arg0: tensor<128x360xf32>) -> tensor<128x360xbf16> {
    %0 = "ttir.typecast"(%arg0) <{conservative_folding = false}> : (tensor<128x360xf32>) -> tensor<128x360xbf16>
    return %0 : tensor<128x360xbf16>
  }
  func.func @ttir_typecast_41(%arg0: tensor<1x128x360xbf16>) -> tensor<1x128x360xf32> {
    %0 = "ttir.typecast"(%arg0) <{conservative_folding = false}> : (tensor<1x128x360xbf16>) -> tensor<1x128x360xf32>
    return %0 : tensor<1x128x360xf32>
  }
  func.func @ttir_typecast_42(%arg0: tensor<1x128x360xf32>) -> tensor<1x128x360xbf16> {
    %0 = "ttir.typecast"(%arg0) <{conservative_folding = false}> : (tensor<1x128x360xf32>) -> tensor<1x128x360xbf16>
    return %0 : tensor<1x128x360xbf16>
  }
  func.func @ttir_typecast_43(%arg0: tensor<128x4xi32>) -> tensor<128x4xi64> {
    %0 = "ttir.typecast"(%arg0) <{conservative_folding = false}> : (tensor<128x4xi32>) -> tensor<128x4xi64>
    return %0 : tensor<128x4xi64>
  }
  func.func @ttir_typecast_44(%arg0: tensor<1x2880xbf16>) -> tensor<1x2880xf32> {
    %0 = "ttir.typecast"(%arg0) <{conservative_folding = false}> : (tensor<1x2880xbf16>) -> tensor<1x2880xf32>
    return %0 : tensor<1x2880xf32>
  }
  func.func @ttir_typecast_45(%arg0: tensor<1x1x2880xbf16>) -> tensor<1x1x2880xf32> {
    %0 = "ttir.typecast"(%arg0) <{conservative_folding = false}> : (tensor<1x1x2880xbf16>) -> tensor<1x1x2880xf32>
    return %0 : tensor<1x1x2880xf32>
  }
  func.func @ttir_typecast_46(%arg0: tensor<128x2880xbf16>) -> tensor<128x2880xf32> {
    %0 = "ttir.typecast"(%arg0) <{conservative_folding = false}> : (tensor<128x2880xbf16>) -> tensor<128x2880xf32>
    return %0 : tensor<128x2880xf32>
  }
  func.func @ttir_typecast_47(%arg0: tensor<128x2880xf32>) -> tensor<128x2880xbf16> {
    %0 = "ttir.typecast"(%arg0) <{conservative_folding = false}> : (tensor<128x2880xf32>) -> tensor<128x2880xbf16>
    return %0 : tensor<128x2880xbf16>
  }
  func.func @ttir_typecast_48(%arg0: tensor<1x128x2880xbf16>) -> tensor<1x128x2880xf32> {
    %0 = "ttir.typecast"(%arg0) <{conservative_folding = false}> : (tensor<1x128x2880xbf16>) -> tensor<1x128x2880xf32>
    return %0 : tensor<1x128x2880xf32>
  }
  func.func @ttir_typecast_49(%arg0: tensor<1x128x2880xf32>) -> tensor<1x128x2880xbf16> {
    %0 = "ttir.typecast"(%arg0) <{conservative_folding = false}> : (tensor<1x128x2880xf32>) -> tensor<1x128x2880xbf16>
    return %0 : tensor<1x128x2880xbf16>
  }
  func.func @ttir_typecast_50(%arg0: tensor<1x64x1xbf16>) -> tensor<1x64x1xf32> {
    %0 = "ttir.typecast"(%arg0) <{conservative_folding = false}> : (tensor<1x64x1xbf16>) -> tensor<1x64x1xf32>
    return %0 : tensor<1x64x1xf32>
  }
  func.func @ttir_typecast_51(%arg0: tensor<32x17xi64>) -> tensor<32x17xui32> {
    %0 = "ttir.typecast"(%arg0) <{conservative_folding = false}> : (tensor<32x17xi64>) -> tensor<32x17xui32>
    return %0 : tensor<32x17xui32>
  }
  func.func @ttir_typecast_52(%arg0: tensor<544x5120xbf16>) -> tensor<544x5120xf32> {
    %0 = "ttir.typecast"(%arg0) <{conservative_folding = false}> : (tensor<544x5120xbf16>) -> tensor<544x5120xf32>
    return %0 : tensor<544x5120xf32>
  }
  func.func @ttir_typecast_53(%arg0: tensor<544x5120xf32>) -> tensor<544x5120xbf16> {
    %0 = "ttir.typecast"(%arg0) <{conservative_folding = false}> : (tensor<544x5120xf32>) -> tensor<544x5120xbf16>
    return %0 : tensor<544x5120xbf16>
  }
  func.func @ttir_typecast_54(%arg0: tensor<544x128xbf16>) -> tensor<544x128xf32> {
    %0 = "ttir.typecast"(%arg0) <{conservative_folding = false}> : (tensor<544x128xbf16>) -> tensor<544x128xf32>
    return %0 : tensor<544x128xf32>
  }
  func.func @ttir_typecast_55(%arg0: tensor<32x1x17x128xf32>) -> tensor<32x1x17x128xbf16> {
    %0 = "ttir.typecast"(%arg0) <{conservative_folding = false}> : (tensor<32x1x17x128xf32>) -> tensor<32x1x17x128xbf16>
    return %0 : tensor<32x1x17x128xbf16>
  }
  func.func @ttir_typecast_56(%arg0: tensor<17xi64>) -> tensor<17xf32> {
    %0 = "ttir.typecast"(%arg0) <{conservative_folding = false}> : (tensor<17xi64>) -> tensor<17xf32>
    return %0 : tensor<17xf32>
  }
  func.func @ttir_typecast_57(%arg0: tensor<1x1x17x128xf32>) -> tensor<1x1x17x128xbf16> {
    %0 = "ttir.typecast"(%arg0) <{conservative_folding = false}> : (tensor<1x1x17x128xf32>) -> tensor<1x1x17x128xbf16>
    return %0 : tensor<1x1x17x128xbf16>
  }
  func.func @ttir_typecast_58(%arg0: tensor<544x1024xbf16>) -> tensor<544x1024xf32> {
    %0 = "ttir.typecast"(%arg0) <{conservative_folding = false}> : (tensor<544x1024xbf16>) -> tensor<544x1024xf32>
    return %0 : tensor<544x1024xf32>
  }
  func.func @ttir_typecast_59(%arg0: tensor<32x8x17x128xf32>) -> tensor<32x8x17x128xbf16> {
    %0 = "ttir.typecast"(%arg0) <{conservative_folding = false}> : (tensor<32x8x17x128xf32>) -> tensor<32x8x17x128xbf16>
    return %0 : tensor<32x8x17x128xbf16>
  }
  func.func @ttir_typecast_60(%arg0: tensor<32x8x17x128xbf16>) -> tensor<32x8x17x128xf32> {
    %0 = "ttir.typecast"(%arg0) <{conservative_folding = false}> : (tensor<32x8x17x128xbf16>) -> tensor<32x8x17x128xf32>
    return %0 : tensor<32x8x17x128xf32>
  }
  func.func @ttir_typecast_61(%arg0: tensor<32x1x128x128xbf16>) -> tensor<32x1x128x128xf32> {
    %0 = "ttir.typecast"(%arg0) <{conservative_folding = false}> : (tensor<32x1x128x128xbf16>) -> tensor<32x1x128x128xf32>
    return %0 : tensor<32x1x128x128xf32>
  }
  func.func @ttir_typecast_62(%arg0: tensor<1x1x17x128xbf16>) -> tensor<1x1x17x128xf32> {
    %0 = "ttir.typecast"(%arg0) <{conservative_folding = false}> : (tensor<1x1x17x128xbf16>) -> tensor<1x1x17x128xf32>
    return %0 : tensor<1x1x17x128xf32>
  }
  func.func @ttir_typecast_63(%arg0: tensor<32x8x17x128xf32>) -> tensor<32x8x17x128xf64> {
    %0 = "ttir.typecast"(%arg0) <{conservative_folding = false}> : (tensor<32x8x17x128xf32>) -> tensor<32x8x17x128xf64>
    return %0 : tensor<32x8x17x128xf64>
  }
}
