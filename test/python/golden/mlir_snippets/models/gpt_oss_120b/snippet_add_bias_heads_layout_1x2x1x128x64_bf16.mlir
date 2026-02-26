module {
  func.func @add_bias_heads_layout_1x2x1x128x64_bf16(%arg0: tensor<128x128xbf16>, %arg1: tensor<1x128xbf16>) -> tensor<1x2x1x128x64xbf16> {
    %0 = "ttir.add"(%arg0, %arg1) : (tensor<128x128xbf16>, tensor<1x128xbf16>) -> tensor<128x128xbf16>
    %1 = "ttir.reshape"(%0) <{shape = [1 : i32, 128 : i32, 2 : i32, 64 : i32]}> : (tensor<128x128xbf16>) -> tensor<1x128x2x64xbf16>
    %2 = "ttir.permute"(%1) <{permutation = array<i64: 0, 2, 1, 3>}> : (tensor<1x128x2x64xbf16>) -> tensor<1x2x128x64xbf16>
    %3 = "ttir.reshape"(%2) <{shape = [1 : i32, 2 : i32, 1 : i32, 128 : i32, 64 : i32]}> : (tensor<1x2x128x64xbf16>) -> tensor<1x2x1x128x64xbf16>
    return %3 : tensor<1x2x1x128x64xbf16>
  }
}
