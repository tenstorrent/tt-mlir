func.func @matmul(%arg0: tensor<256x768xf32>, %arg1: tensor<768x1024xf32>) -> tensor<256x1024xf32> {
  %0 = ttir.empty() : tensor<256x1024xf32>
  %1 = "ttir.matmul"(%arg0, %arg1, %0) : (tensor<256x768xf32>, tensor<768x1024xf32>, tensor<256x1024xf32>) -> tensor<256x1024xf32>
  return %1 : tensor<256x1024xf32>
}
