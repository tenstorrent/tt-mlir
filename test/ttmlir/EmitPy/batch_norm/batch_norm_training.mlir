// RUN: ttmlir-opt --ttir-to-emitpy-pipeline -o %t.mlir %s
// RUN: ttmlir-translate --mlir-to-python -o %t.py %t.mlir

func.func @batch_norm_training(%arg0: tensor<1x32x64x64xf32>, %arg1: tensor<32xf32>, %arg2: tensor<32xf32>, %arg3: tensor<32xf32>, %arg4: tensor<32xf32>) -> (tensor<1x32x64x64xf32>, tensor<32xf32>, tensor<32xf32>) {
  %0 = ttir.empty() : tensor<1x32x64x64xf32>
  %1 = ttir.empty() : tensor<32xf32>
  %2 = ttir.empty() : tensor<32xf32>
  %3:3 = "ttir.batch_norm_training"(%arg0, %arg1, %arg2, %arg3, %arg4, %0, %1, %2) <{dimension = 1 : i32, epsilon = 1.000000e-05 : f32, momentum = 1.000000e-01 : f32}> : (tensor<1x32x64x64xf32>, tensor<32xf32>, tensor<32xf32>, tensor<32xf32>, tensor<32xf32>, tensor<1x32x64x64xf32>, tensor<32xf32>, tensor<32xf32>) -> (tensor<1x32x64x64xf32>, tensor<32xf32>, tensor<32xf32>)
  return %3#0, %3#1, %3#2 : tensor<1x32x64x64xf32>, tensor<32xf32>, tensor<32xf32>
}
