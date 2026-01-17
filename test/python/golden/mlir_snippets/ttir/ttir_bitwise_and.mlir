module {
  func.func @bitwise_ops(%arg0: tensor<136x136xui8>, %arg1: tensor<136x136xui8>) -> tensor<136x136xui8> {
    %1 = "ttir.bitwise_and"(%arg0, %arg1) : (tensor<136x136xui8>, tensor<136x136xui8>) -> tensor<136x136xui8>
    return %1 : tensor<136x136xui8>
  }
}
