module {
  func.func @comparison_ops(%arg0: tensor<136x136xi64>, %arg1: tensor<136x136xi64>) -> tensor<136x136xi1> {
    %0 = "ttir.le"(%arg0, %arg1) : (tensor<136x136xi64>, tensor<136x136xi64>) -> tensor<136x136xi1>
    return %0 : tensor<136x136xi1>
  }
}
