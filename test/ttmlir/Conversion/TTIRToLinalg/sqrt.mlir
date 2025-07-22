// RUN: ttmlir-opt --convert-ttir-to-linalg -o %t %s
// RUN: FileCheck %s --input-file=%t

module attributes {} {
  func.func @sqrt_test(%arg0: tensor<64x128xf32>) -> tensor<64x128xf32> {
    // CHECK: [[VAL0:%[0-9]+]] = tensor.empty() : [[SIZE:tensor<64x128xf32>]]
    %0 = ttir.empty() : tensor<64x128xf32>
    // CHECK: [[VAL1:%[0-9]+]] = linalg.sqrt ins(%arg0 : [[SIZE]]) outs([[VAL0]] : [[SIZE]]) -> [[SIZE]]
    %1 = "ttir.sqrt"(%arg0, %0) : (tensor<64x128xf32>, tensor<64x128xf32>) -> tensor<64x128xf32>
    // CHECK: return [[VAL1]] : [[SIZE]]
    return %1 : tensor<64x128xf32>
  }
}
