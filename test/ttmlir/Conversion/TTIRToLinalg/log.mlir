// RUN: ttmlir-opt --convert-ttir-to-linalg %s | FileCheck %s

func.func @log_test(%arg0: tensor<64x128xf32>) -> tensor<64x128xf32> {
  // CHECK: = tensor.empty() : [[SIZE:tensor<64x128xf32>]]
  %0 = ttir.empty() : tensor<64x128xf32>
  // CHECK: [[VAL1:%[0-9]+]] = tosa.log
  %1 = "ttir.log"(%arg0, %0) : (tensor<64x128xf32>, tensor<64x128xf32>) -> tensor<64x128xf32>
  // CHECK: return [[VAL1]] : [[SIZE]]
  return %1 : tensor<64x128xf32>
}
