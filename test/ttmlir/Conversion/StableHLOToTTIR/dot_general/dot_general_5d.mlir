// REQUIRES: stablehlo
// RUN: ttmlir-opt --stablehlo-to-ttir-pipeline %s | FileCheck %s
module @jit_dot_general attributes {} {
  func.func public @test_dot_general(%arg0 : tensor<4x10x3x5x7xf32>, %arg1 : tensor<4x10x5x7x3xf32>) -> tensor<4x10x3x7x10x7x3xf32> {
    %0 = stablehlo.dot_general %arg0, %arg1, batching_dims = [0] x [0], contracting_dims = [3] x [2] : (tensor<4x10x3x5x7xf32>, tensor<4x10x5x7x3xf32>) -> tensor<4x10x3x7x10x7x3xf32>
    // CHECK: "ttir.dot_general"
    // CHECK-SAME: {batch_dims_lhs = array<i64: 0>, batch_dims_rhs = array<i64: 0>, contract_dims_lhs = array<i64: 3>, contract_dims_rhs = array<i64: 2>}
    // CHECK-SAME: (tensor<4x10x3x5x7xf32>, tensor<4x10x5x7x3xf32>) -> tensor<4x10x3x7x10x7x3xf32>
    return %0 : tensor<4x10x3x7x10x7x3xf32>
  }
}
