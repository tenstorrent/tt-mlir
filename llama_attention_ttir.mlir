#loc = loc("test_transpose")
module {
  func.func @test_transpose(%arg0: tensor<1x12x32x100xf32> loc("test_transpose")) -> tensor<1x32x12x100xf32> {
    %0 = tensor.empty() : tensor<1x32x12x100xf32> loc(#loc1)
    %1 = "ttir.transpose"(%arg0, %0) <{dim0 = -3 : si32, dim1 = -2 : si32}> : (tensor<1x12x32x100xf32>, tensor<1x32x12x100xf32>) -> tensor<1x32x12x100xf32> loc(#loc2)
    return %1 : tensor<1x32x12x100xf32> loc(#loc)
  } loc(#loc)
} loc(#loc)
#loc1 = loc("empty_0")
#loc2 = loc("transpose_0")
