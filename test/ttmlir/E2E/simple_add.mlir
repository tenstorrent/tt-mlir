func.func @add(%arg0: tensor<256x768xf32>, %arg1: tensor<256x768xf32>) -> tensor<256x768xf32> {
  %0 = ttir.empty() : tensor<256x768xf32>
  %1 = "ttir.add"(%arg0, %arg1, %0) <{operandSegmentSizes = array<i32: 2, 1>}> : (tensor<256x768xf32>, tensor<256x768xf32>, tensor<256x768xf32>) -> tensor<256x768xf32>
  return %1 : tensor<256x768xf32>
}
