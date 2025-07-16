// RUN: ttmlir-opt --convert-ttir-to-linalg %s | FileCheck %s

module {
  func.func @relu_test(%arg0: tensor<64x128xf32>) -> tensor<64x128xf32> {
    // CHECK: [[VAL0:%[0-9]+]] = tensor.empty() : [[SIZE:tensor<64x128xf32>]]
    %0 = ttir.empty() : tensor<64x128xf32>
    // CHECK: [[VAL1:%cst[0-9]*]] = arith.constant
    // CHECK: [[VAL2:%[0-9]+]] = linalg.max ins(%arg0, [[VAL1]] : [[SIZE]], [[SIZE]])
    %1 = "ttir.relu"(%arg0, %0) : (tensor<64x128xf32>, tensor<64x128xf32>) -> tensor<64x128xf32>
    // CHECK: return [[VAL2]] : [[SIZE]]
    return %1 : tensor<64x128xf32>
  }
}
