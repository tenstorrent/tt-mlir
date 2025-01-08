// REQUIRES: stablehlo
// RUN: ttmlir-opt --stablehlo-to-ttir-pipeline %s | FileCheck %s
module @jit_loss attributes {} {
  func.func public @main(%arg0: tensor<2x1xf32>, %arg1: tensor<f32>, %arg2: tensor<127x2xf32>, %arg3: tensor<127x1xf32>) -> (tensor<2x1xf32>, tensor<f32>) {
    %0 = stablehlo.dot_general %arg2, %arg0, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<127x2xf32>, tensor<2x1xf32>) -> tensor<127x1xf32>
    %1 = stablehlo.convert %arg1 : tensor<f32>
    %2 = stablehlo.broadcast_in_dim %1, dims = [] : (tensor<f32>) -> tensor<127x1xf32>
    %3 = stablehlo.add %0, %2 : tensor<127x1xf32>
    %4 = stablehlo.subtract %3, %arg3 : tensor<127x1xf32>
    %5 = call @integer_pow(%4) : (tensor<127x1xf32>) -> tensor<127x1xf32>
    %cst = stablehlo.constant dense<2.000000e+00> : tensor<f32>
    %6 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<127x1xf32>
    %7 = stablehlo.multiply %6, %5 : tensor<127x1xf32>
    %cst_0 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
    %cst_1 = stablehlo.constant dense<1.270000e+02> : tensor<f32>
    %8 = stablehlo.divide %cst_0, %cst_1 : tensor<f32>
    %9 = stablehlo.broadcast_in_dim %8, dims = [] : (tensor<f32>) -> tensor<127x1xf32>
    %10 = stablehlo.multiply %9, %7 : tensor<127x1xf32>
    %cst_2 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %11 = stablehlo.reduce(%10 init: %cst_2) applies stablehlo.add across dimensions = [0, 1] : (tensor<127x1xf32>, tensor<f32>) -> tensor<f32>
    %12 = stablehlo.convert %11 : tensor<f32>
    %13 = stablehlo.dot_general %10, %arg2, contracting_dims = [0] x [0], precision = [DEFAULT, DEFAULT] : (tensor<127x1xf32>, tensor<127x2xf32>) -> tensor<1x2xf32>
    // CHECK: "ttir.dot_general"
    // CHECK: {batch_dims_lhs = array<i64>, batch_dims_rhs = array<i64>, contract_dims_lhs = array<i64: 0>, contract_dims_rhs = array<i64: 0>}
    // CHECK: (tensor<127x1xf32>, tensor<127x2xf32>) -> tensor<1x2xf32>
    %14 = stablehlo.transpose %13, dims = [1, 0] : (tensor<1x2xf32>) -> tensor<2x1xf32>
    return %14, %12 : tensor<2x1xf32>, tensor<f32>
  }
  func.func private @integer_pow(%arg0: tensor<127x1xf32>) -> tensor<127x1xf32> {
    return %arg0 : tensor<127x1xf32>
  }
}
