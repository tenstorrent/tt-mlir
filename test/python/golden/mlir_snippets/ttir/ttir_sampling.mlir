func.func @sampling(%arg0: tensor<32x64xbf16>, %arg1: tensor<32x64xi32>, %arg2: tensor<32xui32>, %arg3: tensor<32xbf16>, %arg4: tensor<32xbf16>) -> tensor<32xi32> {
  %0 = "ttir.sampling"(%arg0, %arg1, %arg2, %arg3, %arg4) : (tensor<32x64xbf16>, tensor<32x64xi32>, tensor<32xui32>, tensor<32xbf16>, tensor<32xbf16>) -> tensor<32xi32>
  return %0 : tensor<32xi32>
}
