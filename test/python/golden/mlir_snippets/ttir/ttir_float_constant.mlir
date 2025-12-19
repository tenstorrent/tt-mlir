module {
  func.func @model() -> tensor<32x32xf32> {
    %0 = "ttir.constant"() <{value = dense<-0.190434918> : tensor<32x32xf32>}> : () -> tensor<32x32xf32>
    return %0 : tensor<32x32xf32>
  }
}
