// RUN: ttmlir-opt --convert-ttir-to-linalg %s | FileCheck %s

func.func @log_test(%arg0: tensor<64x128xf32>) -> tensor<64x128xf32> {
  // CHECK: [[VAL0:%[0-9]+]] = tensor.empty() : [[SIZE:tensor<64x128xf32>]]
  %0 = ttir.empty() : tensor<64x128xf32>
  // CHECK: [[VAL1:%[0-9]+]] = linalg.log ins(%arg0 : [[SIZE]]) outs([[VAL0]] : [[SIZE]]) -> [[SIZE]]
  %1 = "ttir.log"(%arg0, %0) : (tensor<64x128xf32>, tensor<64x128xf32>) -> tensor<64x128xf32>
  // CHECK: return [[VAL1]] : [[SIZE]]
  return %1 : tensor<64x128xf32>
}
