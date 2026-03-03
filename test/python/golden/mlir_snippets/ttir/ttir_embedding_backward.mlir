module {
  func.func @embedding_backward_module(%arg0: tensor<1x39x1xi64>, %arg1: tensor<151936x896xbf16>, %arg2: tensor<39x896xbf16>) -> tensor<151936x896xbf16> {
    %0 = "ttir.embedding_backward"(%arg0, %arg1, %arg2) : (tensor<1x39x1xi64>, tensor<151936x896xbf16>, tensor<39x896xbf16>) -> tensor<151936x896xbf16>
    return %0 : tensor<151936x896xbf16>
  }
}
