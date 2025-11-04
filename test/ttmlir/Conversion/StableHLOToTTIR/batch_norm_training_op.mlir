// REQUIRES: stablehlo
// RUN: ttmlir-opt --stablehlo-to-ttir-pipeline -o %t %s
// RUN: FileCheck %s --input-file=%t
module @jit_batch_norm attributes {} {
  func.func public @test_batch_norm(%arg0: tensor<2x2x2x2xf32>, %arg1: tensor<2xf32>, %arg2: tensor<2xf32>) -> (tensor<2x2x2x2xf32>, tensor<2xf32>, tensor<2xf32>) {
    %result, %batch_mean, %batch_variance = "stablehlo.batch_norm_training"(%arg0, %arg1, %arg2) {
      epsilon = 0.0 : f32,
      feature_index = 1 : i64
    } : (tensor<2x2x2x2xf32>, tensor<2xf32>, tensor<2xf32>) -> (tensor<2x2x2x2xf32>, tensor<2xf32>, tensor<2xf32>)
    // CHECK: [[VAL0:%[0-9]+]] = "ttir.zeros"() <{dtype = f32, shape = array<i32: 2>}> : () -> tensor<2xf32>
    // CHECK: [[VAL1:%[0-9]+]] = "ttir.ones"() <{dtype = f32, shape = array<i32: 2>}> : () -> tensor<2xf32>
    // CHECK: [[VAL2:%[0-9]+]] = ttir.empty() : tensor<2x2x2x2xf32>
    // CHECK: [[VAL3:%[0-9]+]] = ttir.empty() : tensor<2xf32>
    // CHECK: [[VAL4:%[0-9]+]] = ttir.empty() : tensor<2xf32>
    // CHECK: [[RESULT:%[a-z_]+]], [[MEAN:%[a-z_]+]], [[VAR:%[a-z_]+]] = "ttir.batch_norm_training"(%arg0, %arg1, %arg2, [[VAL0]], [[VAL1]], [[VAL2]], [[VAL3]], [[VAL4]]) <{dimension = 1 : i32, epsilon = 0.000000e+00 : f32, momentum = 1.000000e+00 : f32}> : (tensor<2x2x2x2xf32>, tensor<2xf32>, tensor<2xf32>, tensor<2xf32>, tensor<2xf32>, tensor<2x2x2x2xf32>, tensor<2xf32>, tensor<2xf32>) -> (tensor<2x2x2x2xf32>, tensor<2xf32>, tensor<2xf32>)
    return %result, %batch_mean, %batch_variance : tensor<2x2x2x2xf32>, tensor<2xf32>, tensor<2xf32>
    // CHECK: return [[RESULT]], [[MEAN]], [[VAR]] : tensor<2x2x2x2xf32>, tensor<2xf32>, tensor<2xf32>
  }
}
