module {
  func.func @sort(%arg0: tensor<64x128xf32>) -> (tensor<64x128xf32>, tensor<64x128xi32>) {
    %0, %1 = "ttir.sort"(%arg0) <{dim = -1 : si32, descending = false, stable = false}> : (tensor<64x128xf32>) -> (tensor<64x128xf32>, tensor<64x128xi32>)
    return %0, %1 : tensor<64x128xf32>, tensor<64x128xi32>
  }
}
