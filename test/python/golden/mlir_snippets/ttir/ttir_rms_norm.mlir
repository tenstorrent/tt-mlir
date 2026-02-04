module {
  func.func @rms_norm(%arg0: tensor<1x136x2048xbf16>, %arg1: tensor<2048xbf16>) -> tensor<1x136x2048xbf16> {
    %1 = "ttir.rms_norm"(%arg0, %arg1) <{epsilon = 9.99999974E-6 : f32, normalized_shape = array<i64: 2048>, operandSegmentSizes = array<i32: 1, 1, 0>}> : (tensor<1x136x2048xbf16>, tensor<2048xbf16>) -> tensor<1x136x2048xbf16>
    return %1 : tensor<1x136x2048xbf16>
  }
}
