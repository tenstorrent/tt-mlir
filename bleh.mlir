func.func @transpose(%arg0: tensor<512x256xf32>) -> tensor<256x512xf32> {
  %0 = ttir.empty() : tensor<256x512xf32>
  %1 = "ttir.transpose"(%arg0, %0) <{dim0 = 0 : si32, dim1 = 1 : si32}> : (tensor<512x256xf32>, tensor<256x512xf32>) -> tensor<256x512xf32>
  %2 = ttir.empty() : tensor<256x512xf32>
  %3 = "ttir.abs"(%1, %2) : (tensor<256x512xf32>, tensor<256x512xf32>) -> tensor<256x512xf32>
  // CHECK: call_opaque "transpose_wh_init"
  // CHECK: call_opaque "transpose_wh_tile"
  return %3 : tensor<256x512xf32>
}
