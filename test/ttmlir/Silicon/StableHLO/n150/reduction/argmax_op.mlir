// REQUIRES: stablehlo
// RUN: rm -rf %t.ttnn
// RUN: rm -rf %t.mlir
// RUN: ttmlir-opt --stablehlo-to-ttir-pipeline --ttir-to-ttnn-backend-pipeline="system-desc-path=%system_desc_path%" -o %t.mlir %s
// RUN: FileCheck --input-file=%t.mlir %s
// RUN: ttmlir-translate --ttnn-to-flatbuffer -o %t.ttnn %t.mlir

module @module_argmax attributes {} {
  func.func public @argmax_torch_2d(%arg0: tensor<1x32128xf32>) -> tensor<1xi64> {
    // CHECK-LABEL: func.func public @argmax_torch_2d(
    // CHECK: "ttnn.argmax"
    // CHECK-SAME: {dim = 3 : i32, keep_dim = false, use_multicore = false}>
    // CHECK-SAME: tensor<1x1x1x32128xf32
    // CHECK-SAME: -> tensor<1x1x1xui32
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

  func.func public @argmax_jax_3d(%arg0: tensor<1x32x32xf32>) -> tensor<1x32xi32> {
    // CHECK-LABEL: func.func public @argmax_jax_3d(
    // CHECK: "ttnn.argmax"
    // CHECK-SAME: {dim = 3 : i32, keep_dim = false, use_multicore = false}>
    // CHECK-SAME: tensor<1x1x32x32xf32
    // CHECK-SAME: -> tensor<1x1x32xui32
    %0 = stablehlo.iota dim = 2 : tensor<1x32x32xi32>
    %cst = stablehlo.constant dense<0xFF800000> : tensor<f32>
    %c = stablehlo.constant dense<0> : tensor<i32>
    %1:2 = stablehlo.reduce(%arg0 init: %cst), (%0 init: %c) across dimensions = [2] : (tensor<1x32x32xf32>, tensor<1x32x32xi32>, tensor<f32>, tensor<i32>) -> (tensor<1x32xf32>, tensor<1x32xi32>)
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
    return %1#1 : tensor<1x32xi32>
  }

  func.func public @argmax_torch_4d(%arg0: tensor<1x1x128x64xf32>) -> tensor<1x1x128xi64> {
    // CHECK-LABEL: func.func public @argmax_torch_4d(
    // CHECK: "ttnn.argmax"
    // CHECK-SAME: {dim = 3 : i32, keep_dim = false, use_multicore = false}>
    // CHECK-SAME: tensor<1x1x128x64xf32
    // CHECK-SAME: -> tensor<1x1x128xui32
    %cst = stablehlo.constant dense<0xFF800000> : tensor<f32>
    %c = stablehlo.constant dense<0> : tensor<i64>
    %0 = stablehlo.iota dim = 3 : tensor<1x1x128x64xi64>
    %1:2 = stablehlo.reduce(%arg0 init: %cst), (%0 init: %c) across dimensions = [3] : (tensor<1x1x128x64xf32>, tensor<1x1x128x64xi64>, tensor<f32>, tensor<i64>) -> (tensor<1x1x128xf32>, tensor<1x1x128xi64>)
     reducer(%arg1: tensor<f32>, %arg3: tensor<f32>) (%arg2: tensor<i64>, %arg4: tensor<i64>)  {
      %2 = stablehlo.compare  GE, %arg1, %arg3,  FLOAT : (tensor<f32>, tensor<f32>) -> tensor<i1>
      %3 = stablehlo.select %2, %arg1, %arg3 : tensor<i1>, tensor<f32>
      %4 = stablehlo.compare  EQ, %arg1, %arg3,  FLOAT : (tensor<f32>, tensor<f32>) -> tensor<i1>
      %5 = stablehlo.minimum %arg2, %arg4 : tensor<i64>
      %6 = stablehlo.select %2, %arg2, %arg4 : tensor<i1>, tensor<i64>
      %7 = stablehlo.select %4, %5, %6 : tensor<i1>, tensor<i64>
      stablehlo.return %3, %7 : tensor<f32>, tensor<i64>
    }
    return %1#1 : tensor<1x1x128xi64>
  }
}
