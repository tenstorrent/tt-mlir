module {
  func.func @logical_and(%arg0: tensor<136x136xi1>, %arg1: tensor<136x136xi1>) -> tensor<136x136xi1> {
    %0 = "ttir.logical_and"(%arg0, %arg1) : (tensor<136x136xi1>, tensor<136x136xi1>) -> tensor<136x136xi1>
    return %0 : tensor<136x136xi1>
  }
}
