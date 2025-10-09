// REQUIRES: stablehlo
// RUN: ttmlir-opt --stablehlo-to-ttir-pipeline -o %t %s
// RUN: FileCheck %s --input-file=%t
module @jit_batch_norm attributes {} {
  func.func public @test_batch_norm(%arg0: tensor<2x2x2x2xf32>, %arg1: tensor<2xf32>, %arg2: tensor<2xf32>, %arg3: tensor<2xf32>, %arg4: tensor<2xf32>) -> tensor<2x2x2x2xf32> {
    %result = "stablehlo.batch_norm_inference"(%arg0, %arg1, %arg2, %arg3, %arg4) {
      epsilon = 0.0 : f32,
      feature_index = 1 : i64
    } : (tensor<2x2x2x2xf32>, tensor<2xf32>, tensor<2xf32>, tensor<2xf32>, tensor<2xf32>) -> tensor<2x2x2x2xf32>
    // CHECK: [[VAL0:%[0-9]+]] = ttir.empty() : [[TENSOR_SIZE:tensor<[0-9]+x[0-9]+x[0-9]+x[0-9]+xf[0-9]+>]]
    // CHECK: [[VAL1:%[0-9]+]] = "ttir.batch_norm"(%arg0, %arg1, %arg2, %arg3, %arg4, %0) <{dimension = 1 : i32, epsilon = 0.000000e+00 : f32}> : (tensor<2x2x2x2xf32>, tensor<2xf32>, tensor<2xf32>, tensor<2xf32>, tensor<2xf32>, tensor<2x2x2x2xf32>) -> tensor<2x2x2x2xf32>
    return %result : tensor<2x2x2x2xf32>
    // CHECK: return [[VAL1]] : [[TENSOR_SIZE]]
  }
}
