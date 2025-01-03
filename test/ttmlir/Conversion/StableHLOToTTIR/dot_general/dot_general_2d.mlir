// REQUIRES: stablehlo
// RUN: ttmlir-opt --stablehlo-to-ttir-pipeline %s | FileCheck %s
module @jit_dot_general attributes {} {
  func.func public @test_dot_general(%arg0 : tensor<16x32xf32>, %arg1 : tensor<8x32xf32>) -> tensor<16x8xf32> {
    %0 = stablehlo.dot_general %arg0, %arg1, contracting_dims = [1] x [1] : (tensor<16x32xf32>, tensor<8x32xf32>) -> tensor<16x8xf32>
    // CHECK: "ttir.dot_general"
    // CHECK-SAME: {batch_dims_a = array<i64>, batch_dims_b = array<i64>, contract_dims_a = array<i64: 1>, contract_dims_b = array<i64: 1>}
    // CHECK-SAME: (tensor<16x32xf32>, tensor<8x32xf32>) -> tensor<16x8xf32>
    return %0 : tensor<16x8xf32>
  }
}
