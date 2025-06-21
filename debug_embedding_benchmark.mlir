func.func @benchmark_embedding_small(%indices: tensor<10xi32>, %weights: tensor<50x32xf32>) -> tensor<10x32xf32> {
  %0 = ttir.empty() : tensor<10x32xf32>
  %result = "ttir.embedding"(%indices, %weights, %0) : (tensor<10xi32>, tensor<50x32xf32>, tensor<10x32xf32>) -> tensor<10x32xf32>
  return %result : tensor<10x32xf32>
}

func.func @benchmark_embedding_medium(%indices: tensor<100xi32>, %weights: tensor<1000x128xf32>) -> tensor<100x128xf32> {
  %0 = ttir.empty() : tensor<100x128xf32>
  %result = "ttir.embedding"(%indices, %weights, %0) : (tensor<100xi32>, tensor<1000x128xf32>, tensor<100x128xf32>) -> tensor<100x128xf32>
  return %result : tensor<100x128xf32>
}

func.func @benchmark_embedding_large(%indices: tensor<1000xi32>, %weights: tensor<10000x512xf32>) -> tensor<1000x512xf32> {
  %0 = ttir.empty() : tensor<1000x512xf32>
  %result = "ttir.embedding"(%indices, %weights, %0) : (tensor<1000xi32>, tensor<10000x512xf32>, tensor<1000x512xf32>) -> tensor<1000x512xf32>
  return %result : tensor<1000x512xf32>
}
