// RUN: ttmlir-opt --ttir-to-ttnn-backend-pipeline -o %t %s
// RUN: FileCheck %s --input-file=%t
module @jit_batch_norm_training {
  func.func public @test_batch_norm_training(%arg0: tensor<2x2x2x2xf32>, %arg1: tensor<2xf32>, %arg2: tensor<2xf32>, %arg3: tensor<2xf32>, %arg4: tensor<2xf32>) -> (tensor<2x2x2x2xf32>, tensor<2xf32>, tensor<2xf32>) {
    %0 = ttir.empty() : tensor<2x2x2x2xf32>
    %1 = ttir.empty() : tensor<2xf32>
    %2 = ttir.empty() : tensor<2xf32>
    // CHECK: [[VAL0:%[0-9]+]] = "ttnn.reshape"(%arg1)
    // CHECK: [[VAL1:%[0-9]+]] = "ttnn.reshape"(%arg2)
    // CHECK: [[VAL2:%[0-9]+]] = "ttnn.reshape"(%arg3)
    // CHECK: [[VAL3:%[0-9]+]] = "ttnn.reshape"(%arg4)
    %3:3 = "ttir.batch_norm_training"(%arg0, %arg1, %arg2, %arg3, %arg4, %0, %1, %2) <{dimension = 1 : i32, epsilon = 1.000000e-05 : f32, momentum = 1.000000e-01 : f32}> : (tensor<2x2x2x2xf32>, tensor<2xf32>, tensor<2xf32>, tensor<2xf32>, tensor<2xf32>, tensor<2x2x2x2xf32>, tensor<2xf32>, tensor<2xf32>) -> (tensor<2x2x2x2xf32>, tensor<2xf32>, tensor<2xf32>)
    // CHECK: [[RESULT:%[a-z_]+]], [[BATCH_MEAN:%[a-z_]+]], [[BATCH_VAR:%[a-z_]+]] = "ttnn.batch_norm_training"(%arg0, [[VAL2]], [[VAL3]], [[VAL0]], [[VAL1]]) <{epsilon = {{.*}}, momentum = {{.*}}, operandSegmentSizes = array<i32: 1, 1, 1, 1, 1>}>
    // CHECK: [[VAL4:%[0-9]+]] = "ttnn.reshape"([[BATCH_MEAN]])
    // CHECK: [[VAL5:%[0-9]+]] = "ttnn.reshape"([[BATCH_VAR]])
    return %3#0, %3#1, %3#2 : tensor<2x2x2x2xf32>, tensor<2xf32>, tensor<2xf32>
    // CHECK: return [[RESULT]], [[VAL4]], [[VAL5]]
  }
}
