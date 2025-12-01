module attributes {} {
  func.func @logical_not(%arg0: tensor<64x128xf32>) -> tensor<64x128xf32> {
    %1 = "ttir.logical_not"(%arg0) : (tensor<64x128xf32>) -> tensor<64x128xf32>
    return %1 : tensor<64x128xf32>
  }
}
