// RUN: ttmlir-opt --ttir-to-ttnn-backend-pipeline="system-desc-path=%system_desc_path%" -o %t.mlir %s
// RUN: ttmlir-translate --ttnn-to-flatbuffer -o %t.ttnn %t.mlir
module @jit_batch_norm_training {
  func.func public @test_batch_norm_training(%arg0: tensor<2x2x2x2xf32>, %arg1: tensor<2xf32>, %arg2: tensor<2xf32>, %arg3: tensor<2xf32>, %arg4: tensor<2xf32>) -> (tensor<2x2x2x2xf32>, tensor<2xf32>, tensor<2xf32>) {
    %0 = ttir.empty() : tensor<2x2x2x2xf32>
    %1 = ttir.empty() : tensor<2xf32>
    %2 = ttir.empty() : tensor<2xf32>
    %3:3 = "ttir.batch_norm_training"(%arg0, %arg1, %arg2, %arg3, %arg4, %0, %1, %2) <{dimension = 1 : i32, epsilon = 1.000000e-05 : f32, momentum = 1.000000e-01 : f32}> : (tensor<2x2x2x2xf32>, tensor<2xf32>, tensor<2xf32>, tensor<2xf32>, tensor<2xf32>, tensor<2x2x2x2xf32>, tensor<2xf32>, tensor<2xf32>) -> (tensor<2x2x2x2xf32>, tensor<2xf32>, tensor<2xf32>)
    return %3#0, %3#1, %3#2 : tensor<2x2x2x2xf32>, tensor<2xf32>, tensor<2xf32>
  }
}
