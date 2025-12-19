module {
  func.func @modela(%arg0: tensor<32x32xf32>) -> tensor<32x32xf32> {
    %0 = "ttir.sigmoid"(%arg0) : (tensor<32x32xf32>) -> tensor<32x32xf32>
    return %0 : tensor<32x32xf32>
  }
  func.func @modelb(%arg0: tensor<32x32xf32>) -> tensor<32x32xf32> {
    %0 = "ttir.sigmoid"(%arg0) : (tensor<32x32xf32>) -> tensor<32x32xf32>
    return %0 : tensor<32x32xf32>
  }
}
