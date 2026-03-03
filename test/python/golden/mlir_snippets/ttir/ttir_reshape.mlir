module {
  func.func @model(%arg0: tensor<32x32xf32>) -> tensor<1x1024xf32> {
    %1 = "ttir.reshape"(%arg0) <{shape = [1 : i32, 1024 : i32]}> : (tensor<32x32xf32>) -> tensor<1x1024xf32>
    return %1 : tensor<1x1024xf32>
  }
}
