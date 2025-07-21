// REQUIRES: stablehlo
// RUN: ttmlir-opt --stablehlo-to-ttir-pipeline -o %t %s
// RUN: FileCheck %s --input-file=%t

module @jit_composite_fn attributes {mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} {
  func.func public @main(%arg0: tensor<2x2xf32>) -> (tensor<2x2xf32> {jax.result_info = ""}) {
    // CHECK: ttir.gelu
    // CHECK-NOT: ttir.tanh
    // CHECK: ttir.cbrt
    // CHECK: ttir.exp
    %0 = stablehlo.composite "tenstorrent.gelu_tanh" %arg0 {decomposition = @tenstorrent.gelu_tanh} : (tensor<2x2xf32>) -> tensor<2x2xf32>
    %1 = stablehlo.composite "abc.test" %arg0 {decomposition = @abc.test} : (tensor<2x2xf32>) -> tensor<2x2xf32>
    %2 = stablehlo.add %0, %1 : tensor<2x2xf32>
    return %2 : tensor<2x2xf32>
  }
  func.func private @tenstorrent.gelu_tanh(%arg0: tensor<2x2xf32>) -> tensor<2x2xf32> {
    %cst = stablehlo.constant dense<5.000000e-01> : tensor<f32>
    %cst_0 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
    %cst_1 = stablehlo.constant dense<0.797884583> : tensor<f32>
    %cst_2 = stablehlo.constant dense<4.471500e-02> : tensor<f32>
    %0 = stablehlo.multiply %arg0, %arg0 : tensor<2x2xf32>
    %1 = stablehlo.multiply %0, %arg0 : tensor<2x2xf32>
    %2 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f32>) -> tensor<2x2xf32>
    %3 = stablehlo.multiply %2, %1 : tensor<2x2xf32>
    %4 = stablehlo.add %arg0, %3 : tensor<2x2xf32>
    %5 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<2x2xf32>
    %6 = stablehlo.multiply %5, %4 : tensor<2x2xf32>
    %7 = stablehlo.tanh %6 : tensor<2x2xf32>
    %8 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<2x2xf32>
    %9 = stablehlo.add %8, %7 : tensor<2x2xf32>
    %10 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<2x2xf32>
    %11 = stablehlo.multiply %10, %9 : tensor<2x2xf32>
    %12 = stablehlo.multiply %arg0, %11 : tensor<2x2xf32>
    return %12 : tensor<2x2xf32>
  }
  func.func private @abc.test(%arg0: tensor<2x2xf32>) -> tensor<2x2xf32> {
    %0 = stablehlo.cbrt %arg0 : tensor<2x2xf32>
    %1 = stablehlo.exponential %0 : tensor<2x2xf32>
    return %1 : tensor<2x2xf32>
  }
}
