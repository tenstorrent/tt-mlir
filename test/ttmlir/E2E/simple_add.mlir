func.func @add(%arg0: tensor<128x256xf32>, %arg1: tensor<128x256xf32>) -> tensor<128x256xf32> {
  %0 = ttir.empty() : tensor<128x256xf32>
  %1 = "ttir.add"(%arg0, %arg1, %0) <{operandSegmentSizes = array<i32: 2, 1>}> : (tensor<128x256xf32>, tensor<128x256xf32>, tensor<128x256xf32>) -> tensor<128x256xf32>
  return %1 : tensor<128x256xf32>
}


