module {
  func.func @model() -> tensor<136x136xui8> {
    %0 = "ttir.constant"() <{value = dense<1> : tensor<136x136xui8>}> : () -> tensor<136x136xui8>
    return %0 : tensor<136x136xui8>
  }
}
