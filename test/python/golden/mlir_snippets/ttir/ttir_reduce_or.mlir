module {
  func.func public @test_reduce_or_4to3dim(%arg0: tensor<128x10x32x4xbf16>, %arg1: tensor<1xbf16>) -> tensor<128x10x32xbf16> {
    %1 = "ttir.reduce_or"(%arg0) <{dim_arg = [3 : i32], keep_dim = false}> : (tensor<128x10x32x4xbf16>) -> tensor<128x10x32xbf16>
    return %1 : tensor<128x10x32xbf16>
  }
}
