module {
  func.func @test_concat_canonicalize_multiple_non_empty_inputs(%arg0: tensor<1x2x1xf32>, %arg1: tensor<1x2x0xf32>) -> tensor<1x2x2xf32> {
    %1 = "ttir.concat"(%arg0, %arg1, %arg0) <{dim = 2 : si32}> : (tensor<1x2x1xf32>, tensor<1x2x0xf32>, tensor<1x2x1xf32>) -> tensor<1x2x2xf32>
    return %1 : tensor<1x2x2xf32>
  }
}
