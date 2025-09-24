// RUN: ttmlir-opt --convert-ttir-to-linalg -o %t %s
// RUN: FileCheck %s --input-file=%t

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

  func.func @relu_test_bf16(%arg0: tensor<64x128xbf16>) -> tensor<64x128xbf16> {
    // CHECK: [[VAL0:%[0-9]+]] = tensor.empty() : [[SIZE:tensor<64x128xbf16>]]
    %0 = ttir.empty() : tensor<64x128xbf16>
    // CHECK: [[VAL1:%cst[0-9]*]] = arith.constant
    // CHECK: [[VAL2:%[0-9]+]] = linalg.max ins(%arg0, [[VAL1]] : [[SIZE]], [[SIZE]])
    %1 = "ttir.relu"(%arg0, %0) : (tensor<64x128xbf16>, tensor<64x128xbf16>) -> tensor<64x128xbf16>
    // CHECK: return [[VAL2]] : [[SIZE]]
    return %1 : tensor<64x128xbf16>
  }
}
