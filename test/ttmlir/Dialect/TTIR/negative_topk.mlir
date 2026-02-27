// RUN: not ttmlir-opt %s 2>&1 | FileCheck %s
// Negative tests for topk op

// CHECK: error: 'ttir.topk' op K should be between 1 and the size of the specified dimension (3), but got: 129
func.func @test_invalid_k(%input: tensor<2x3x32x128xf32>) -> (tensor<2x3x32x129xf32>, tensor<2x3x32x129xi32>) {
  %values, %indices = "ttir.topk"(%input) { k = 129 : i32} : (tensor<2x3x32x128xf32>) -> (tensor<2x3x32x129xf32>, tensor<2x3x32x129xi32>)
  return %values, %indices : tensor<2x3x32x129xf32>, tensor<2x3x32x129xi32>
}

// CHECK: error: 'ttir.topk' op specified dimension should be between -4 and 3, but got: 4
func.func @test_invalid_dim(%input: tensor<2x8x4x256xf32>) -> (tensor<2x3x4x5xf32>, tensor<2x3x4x5xi32>) {
  %values, %indices = "ttir.topk"(%input) {k = 5: i32, dim = 4: i32} : (tensor<2x8x4x256xf32>) -> (tensor<2x3x4x5xf32>, tensor<2x3x4x5xi32>)
  return %values, %indices : tensor<2x3x4x5xf32>, tensor<2x3x4x5xi32>
}
