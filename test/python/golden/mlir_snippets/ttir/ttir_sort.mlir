module {
  func.func @sort(%arg0: tensor<4x8xf32>) -> (tensor<4x8xf32>, tensor<4x8xi64>) {
    %values, %indices = "ttir.sort"(%arg0) <{dim = -1 : si32, descending = false, stable = false}> : (tensor<4x8xf32>) -> (tensor<4x8xf32>, tensor<4x8xi64>)
    return %values, %indices : tensor<4x8xf32>, tensor<4x8xi64>
  }
}
