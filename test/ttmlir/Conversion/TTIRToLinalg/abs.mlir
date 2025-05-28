// RUN: ttmlir-opt --convert-ttir-to-linalg %s | FileCheck %s

module attributes {} {
  func.func @test_abs(%arg0: tensor<64x128xf32>) -> tensor<64x128xf32> {
    // CHECK:  = tensor.empty() : [[SIZE:tensor<64x128xf32>]]
    %0 = ttir.empty() : tensor<64x128xf32>
    // CHECK: [[VAL1:%[0-9]+]] = tosa.abs
    %1 = "ttir.abs"(%arg0, %0) : (tensor<64x128xf32>, tensor<64x128xf32>) -> tensor<64x128xf32>
    // CHECK: return [[VAL1]] : [[SIZE]]
    return %1 : tensor<64x128xf32>
  }
}
