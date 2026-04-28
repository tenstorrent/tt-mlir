
module {
  func.func @forward_sum(%arg0: tensor<2x3xf32>, %arg1: tensor<f32>) -> tensor<3xf32> {
    %0 = "stablehlo.reduce"(%arg0, %arg1) ({
    ^bb0(%arg2: tensor<f32>, %arg3: tensor<f32>):
      %1 = stablehlo.add %arg2, %arg3 : tensor<f32>
      stablehlo.return %1 : tensor<f32>
    }) {dimensions = array<i64: 0>} : (tensor<2x3xf32>, tensor<f32>) -> tensor<3xf32>
    return %0 : tensor<3xf32>
  }
  func.func @forward_max(%arg0: tensor<2x3xf32>, %arg1: tensor<f32>) -> tensor<2xf32> {
    %0 = "stablehlo.reduce"(%arg0, %arg1) ({
    ^bb0(%arg2: tensor<f32>, %arg3: tensor<f32>):
      %1 = stablehlo.maximum %arg2, %arg3 : tensor<f32>
      stablehlo.return %1 : tensor<f32>
    }) {dimensions = array<i64: 1>} : (tensor<2x3xf32>, tensor<f32>) -> tensor<2xf32>
    return %0 : tensor<2xf32>
  }
  func.func @forward_min(%arg0: tensor<4x5xf32>, %arg1: tensor<f32>) -> tensor<5xf32> {
    %0 = "stablehlo.reduce"(%arg0, %arg1) ({
    ^bb0(%arg2: tensor<f32>, %arg3: tensor<f32>):
      %1 = stablehlo.minimum %arg2, %arg3 : tensor<f32>
      stablehlo.return %1 : tensor<f32>
    }) {dimensions = array<i64: 0>} : (tensor<4x5xf32>, tensor<f32>) -> tensor<5xf32>
    return %0 : tensor<5xf32>
  }
}
