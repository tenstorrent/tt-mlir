// REQUIRES: stablehlo
// RUN: ttmlir-opt --stablehlo-to-ttir-pipeline %s | FileCheck %s

module @module_argmax attributes {} {
  func.func public @argmax_torch_2d(%arg0: tensor<1x32128xf32>) -> tensor<1xi64> {
    // CHECK-LABEL: func.func public @argmax_torch_2d(
    // CHECK: %[[EMPTY:[0-9]+]] = tensor.empty() : tensor<1xi32>
    // CHECK: %[[VAL:[0-9]+]] = "ttir.argmax"(%arg0, %[[EMPTY]])
    // CHECK-SAME: dim_arg = [1 : i32]
    // CHECK-SAME: keep_dim = false
    // CHECK-SAME: (tensor<1x32128xf32>, tensor<1xi32>) -> tensor<1xi32>
    // CHECK: return %[[VAL]] : tensor<1xi32>
    %cst = stablehlo.constant dense<0xFF800000> : tensor<f32>
    %c = stablehlo.constant dense<0> : tensor<i64>
    %0 = stablehlo.iota dim = 1 : tensor<1x32128xi64>
    %1:2 = stablehlo.reduce(%arg0 init: %cst), (%0 init: %c) across dimensions = [1] : (tensor<1x32128xf32>, tensor<1x32128xi64>, tensor<f32>, tensor<i64>) -> (tensor<1xf32>, tensor<1xi64>)
     reducer(%arg1: tensor<f32>, %arg3: tensor<f32>) (%arg2: tensor<i64>, %arg4: tensor<i64>)  {
      %2 = stablehlo.compare  GE, %arg1, %arg3,  FLOAT : (tensor<f32>, tensor<f32>) -> tensor<i1>
      %3 = stablehlo.select %2, %arg1, %arg3 : tensor<i1>, tensor<f32>
      %4 = stablehlo.compare  EQ, %arg1, %arg3,  FLOAT : (tensor<f32>, tensor<f32>) -> tensor<i1>
      %5 = stablehlo.minimum %arg2, %arg4 : tensor<i64>
      %6 = stablehlo.select %2, %arg2, %arg4 : tensor<i1>, tensor<i64>
      %7 = stablehlo.select %4, %5, %6 : tensor<i1>, tensor<i64>
      stablehlo.return %3, %7 : tensor<f32>, tensor<i64>
    }
    return %1#1 : tensor<1xi64>
  }

  func.func public @argmax_jax_2d(%arg0: tensor<64x64xf32>) -> tensor<64xi32> {
    // CHECK-LABEL: func.func public @argmax_jax_2d(
    // CHECK: %[[EMPTY:[0-9]+]] = tensor.empty() : tensor<64xi32>
    // CHECK: %[[VAL:[0-9]+]] = "ttir.argmax"(%arg0, %[[EMPTY]])
    // CHECK-SAME: dim_arg = [1 : i32]
    // CHECK-SAME: keep_dim = false
    // CHECK-SAME: (tensor<64x64xf32>, tensor<64xi32>) -> tensor<64xi32>
    // CHECK: return %[[VAL]] : tensor<64xi32>
    %c = stablehlo.constant dense<0> : tensor<i32>
    %cst = stablehlo.constant dense<0xFF800000> : tensor<f32>
    %0 = stablehlo.iota dim = 1 : tensor<64x64xi32>
    %1:2 = stablehlo.reduce(%arg0 init: %cst), (%0 init: %c) across dimensions = [1] : (tensor<64x64xf32>, tensor<64x64xi32>, tensor<f32>, tensor<i32>) -> (tensor<64xf32>, tensor<64xi32>)
     reducer(%arg1: tensor<f32>, %arg3: tensor<f32>) (%arg2: tensor<i32>, %arg4: tensor<i32>)  {
      %2 = stablehlo.compare  GT, %arg1, %arg3,  FLOAT : (tensor<f32>, tensor<f32>) -> tensor<i1>
      %3 = stablehlo.compare  NE, %arg1, %arg1,  FLOAT : (tensor<f32>, tensor<f32>) -> tensor<i1>
      %4 = stablehlo.or %2, %3 : tensor<i1>
      %5 = stablehlo.compare  EQ, %arg1, %arg3,  FLOAT : (tensor<f32>, tensor<f32>) -> tensor<i1>
      %6 = stablehlo.compare  LT, %arg2, %arg4,  SIGNED : (tensor<i32>, tensor<i32>) -> tensor<i1>
      %7 = stablehlo.and %5, %6 : tensor<i1>
      %8 = stablehlo.or %4, %7 : tensor<i1>
      %9 = stablehlo.select %4, %arg1, %arg3 : tensor<i1>, tensor<f32>
      %10 = stablehlo.select %8, %arg2, %arg4 : tensor<i1>, tensor<i32>
      stablehlo.return %9, %10 : tensor<f32>, tensor<i32>
    }
    return %1#1 : tensor<64xi32>
  }

  func.func public @argmax_jax_3d(%arg0: tensor<128x28x28xf32>) -> tensor<28x28xi32> {
    // CHECK-LABEL: func.func public @argmax_jax_3d(
    // CHECK: %[[EMPTY:[0-9]+]] = tensor.empty() : tensor<28x28xi32>
    // CHECK: %[[VAL:[0-9]+]] = "ttir.argmax"(%arg0, %[[EMPTY]])
    // CHECK-SAME: dim_arg = [0 : i32]
    // CHECK-SAME: keep_dim = false
    // CHECK-SAME: (tensor<128x28x28xf32>, tensor<28x28xi32>) -> tensor<28x28xi32>
    // CHECK: return %[[VAL]] : tensor<28x28xi32>
    %0 = stablehlo.iota dim = 0 : tensor<128x28x28xi32>
    %cst = stablehlo.constant dense<0xFF800000> : tensor<f32>
    %c = stablehlo.constant dense<0> : tensor<i32>
    %1:2 = stablehlo.reduce(%arg0 init: %cst), (%0 init: %c) across dimensions = [0] : (tensor<128x28x28xf32>, tensor<128x28x28xi32>, tensor<f32>, tensor<i32>) -> (tensor<28x28xf32>, tensor<28x28xi32>)
     reducer(%arg1: tensor<f32>, %arg3: tensor<f32>) (%arg2: tensor<i32>, %arg4: tensor<i32>)  {
      %2 = stablehlo.compare  GT, %arg1, %arg3,  FLOAT : (tensor<f32>, tensor<f32>) -> tensor<i1>
      %3 = stablehlo.compare  NE, %arg1, %arg1,  FLOAT : (tensor<f32>, tensor<f32>) -> tensor<i1>
      %4 = stablehlo.or %2, %3 : tensor<i1>
      %5 = stablehlo.compare  EQ, %arg1, %arg3,  FLOAT : (tensor<f32>, tensor<f32>) -> tensor<i1>
      %6 = stablehlo.compare  LT, %arg2, %arg4,  SIGNED : (tensor<i32>, tensor<i32>) -> tensor<i1>
      %7 = stablehlo.and %5, %6 : tensor<i1>
      %8 = stablehlo.or %4, %7 : tensor<i1>
      %9 = stablehlo.select %4, %arg1, %arg3 : tensor<i1>, tensor<f32>
      %10 = stablehlo.select %8, %arg2, %arg4 : tensor<i1>, tensor<i32>
      stablehlo.return %9, %10 : tensor<f32>, tensor<i32>
    }
    return %1#1 : tensor<28x28xi32>
  }

  func.func public @argmax_torch_4d(%arg0: tensor<4x8x128x64xf32>) -> tensor<4x8x64xi64> {
    // CHECK-LABEL: func.func public @argmax_torch_4d(
    // CHECK: %[[EMPTY:[0-9]+]] = tensor.empty() : tensor<4x8x64xi32>
    // CHECK: %[[VAL:[0-9]+]] = "ttir.argmax"(%arg0, %[[EMPTY]])
    // CHECK-SAME: dim_arg = [2 : i32]
    // CHECK-SAME: keep_dim = false
    // CHECK-SAME: (tensor<4x8x128x64xf32>, tensor<4x8x64xi32>) -> tensor<4x8x64xi32>
    // CHECK: return %[[VAL]] : tensor<4x8x64xi32>
    %cst = stablehlo.constant dense<0xFF800000> : tensor<f32>
    %c = stablehlo.constant dense<0> : tensor<i64>
    %0 = stablehlo.iota dim = 2 : tensor<4x8x128x64xi64>
    %1:2 = stablehlo.reduce(%arg0 init: %cst), (%0 init: %c) across dimensions = [2] : (tensor<4x8x128x64xf32>, tensor<4x8x128x64xi64>, tensor<f32>, tensor<i64>) -> (tensor<4x8x64xf32>, tensor<4x8x64xi64>)
     reducer(%arg1: tensor<f32>, %arg3: tensor<f32>) (%arg2: tensor<i64>, %arg4: tensor<i64>)  {
      %2 = stablehlo.compare  GE, %arg1, %arg3,  FLOAT : (tensor<f32>, tensor<f32>) -> tensor<i1>
      %3 = stablehlo.select %2, %arg1, %arg3 : tensor<i1>, tensor<f32>
      %4 = stablehlo.compare  EQ, %arg1, %arg3,  FLOAT : (tensor<f32>, tensor<f32>) -> tensor<i1>
      %5 = stablehlo.minimum %arg2, %arg4 : tensor<i64>
      %6 = stablehlo.select %2, %arg2, %arg4 : tensor<i1>, tensor<i64>
      %7 = stablehlo.select %4, %5, %6 : tensor<i1>, tensor<i64>
      stablehlo.return %3, %7 : tensor<f32>, tensor<i64>
    }
    return %1#1 : tensor<4x8x64xi64>
  }
}
