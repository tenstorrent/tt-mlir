// REQUIRES: stablehlo
// RUN: ttmlir-opt --stablehlo-to-ttir-pipeline %s | FileCheck %s
// TODO(mrakita): Enable when all ops are supported.
// UNSUPPORTED: true
module @jit_predict attributes {mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} {
  func.func public @main(%arg0: tensor<512x784xf32> {mhlo.layout_mode = "default"}, %arg1: tensor<512xf32> {mhlo.layout_mode = "default"}, %arg2: tensor<10x512xf32> {mhlo.layout_mode = "default"}, %arg3: tensor<10xf32> {mhlo.layout_mode = "default"}, %arg4: tensor<128x784xui8> {mhlo.layout_mode = "default"}) -> (tensor<128x10xf32> {jax.result_info = "", mhlo.layout_mode = "default"}) {
    %0 = stablehlo.convert %arg0 : tensor<512x784xf32>
    %1 = stablehlo.convert %arg4 : (tensor<128x784xui8>) -> tensor<128x784xf32>
    %2 = stablehlo.dot_general %0, %1, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<512x784xf32>, tensor<128x784xf32>) -> tensor<512x128xf32>
    %3 = stablehlo.transpose %2, dims = [1, 0] : (tensor<512x128xf32>) -> tensor<128x512xf32>
    %4 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<512xf32>) -> tensor<1x512xf32>
    %5 = stablehlo.broadcast_in_dim %4, dims = [0, 1] : (tensor<1x512xf32>) -> tensor<128x512xf32>
    %6 = stablehlo.add %3, %5 : tensor<128x512xf32>
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %7 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<128x512xf32>
    %8 = stablehlo.maximum %7, %6 : tensor<128x512xf32>
    %9 = stablehlo.dot_general %arg2, %8, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<10x512xf32>, tensor<128x512xf32>) -> tensor<10x128xf32>
    %10 = stablehlo.transpose %9, dims = [1, 0] : (tensor<10x128xf32>) -> tensor<128x10xf32>
    %11 = stablehlo.broadcast_in_dim %arg3, dims = [1] : (tensor<10xf32>) -> tensor<1x10xf32>
    %12 = stablehlo.broadcast_in_dim %11, dims = [0, 1] : (tensor<1x10xf32>) -> tensor<128x10xf32>
    %13 = stablehlo.add %10, %12 : tensor<128x10xf32>
    %cst_0 = stablehlo.constant dense<0xFF800000> : tensor<f32>
    %14 = stablehlo.reduce(%13 init: %cst_0) applies stablehlo.maximum across dimensions = [1] : (tensor<128x10xf32>, tensor<f32>) -> tensor<128xf32>
    %cst_1 = stablehlo.constant dense<0xFF800000> : tensor<f32>
    %15 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %16 = stablehlo.maximum %15, %14 : tensor<128xf32>
    %17 = stablehlo.broadcast_in_dim %16, dims = [0] : (tensor<128xf32>) -> tensor<128x1xf32>
    %18 = stablehlo.broadcast_in_dim %17, dims = [0, 1] : (tensor<128x1xf32>) -> tensor<128x10xf32>
    %19 = stablehlo.subtract %13, %18 : tensor<128x10xf32>
    %20 = stablehlo.exponential %19 : tensor<128x10xf32>
    %cst_2 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %21 = stablehlo.reduce(%20 init: %cst_2) applies stablehlo.add across dimensions = [1] : (tensor<128x10xf32>, tensor<f32>) -> tensor<128xf32>
    %22 = stablehlo.broadcast_in_dim %21, dims = [0] : (tensor<128xf32>) -> tensor<128x1xf32>
    %23 = stablehlo.broadcast_in_dim %22, dims = [0, 1] : (tensor<128x1xf32>) -> tensor<128x10xf32>
    %24 = stablehlo.divide %20, %23 : tensor<128x10xf32>
    return %24 : tensor<128x10xf32>
  }
}
