// RUN: ttmlir-opt --ttir-to-emitpy-pipeline -o %t.mlir %s
// RUN: ttmlir-translate --mlir-to-python -o %t.py %t.mlir

func.func @batch_norm(%arg0: tensor<1x32x64x64xf32>, %arg1: tensor<32xf32>, %arg2: tensor<32xf32>, %arg3: tensor<32xf32>, %arg4: tensor<32xf32>) -> tensor<1x32x64x64xf32> {
  %0 = ttir.empty() : tensor<1x32x64x64xf32>
  %1 = "ttir.batch_norm"(%arg0, %arg3, %arg4, %arg1, %arg2, %0) <{dimension = 1 : i32, epsilon = 1.000000e-05 : f32}> : (tensor<1x32x64x64xf32>, tensor<32xf32>, tensor<32xf32>, tensor<32xf32>, tensor<32xf32>, tensor<1x32x64x64xf32>) -> tensor<1x32x64x64xf32>
  return %1 : tensor<1x32x64x64xf32>
}
