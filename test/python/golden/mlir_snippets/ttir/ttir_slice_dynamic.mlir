module {
  func.func @slice_dynamic_2d(%arg0: tensor<4x4xf32>, %arg1: tensor<2xi32>, %arg2: tensor<2xi32>) -> tensor<2x2xf32> {
    %0 = "ttir.slice_dynamic"(%arg0, %arg1, %arg2) <{step = [1: i32, 1: i32]}> : (tensor<4x4xf32>, tensor<2xi32>, tensor<2xi32>) -> tensor<2x2xf32>
    return %0 : tensor<2x2xf32>
  }
}
