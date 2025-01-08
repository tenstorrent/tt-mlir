// REQUIRES: stablehlo
// RUN: ttmlir-opt --stablehlo-to-ttir-pipeline %s | FileCheck %s
module @jit_dot_general attributes {} {
  func.func public @test_dot_general(%arg0 : tensor<4x10x1xf32>, %arg1 : tensor<4x10x2xf32>) -> tensor<1x2xf32> {
    %0 = stablehlo.dot_general %arg0, %arg1, contracting_dims = [0, 1] x [0, 1] : (tensor<4x10x1xf32>, tensor<4x10x2xf32>) -> tensor<1x2xf32>
    // CHECK: "ttir.dot_general"
    // CHECK-SAME: {batch_dims_lhs = array<i64>, batch_dims_rhs = array<i64>, contract_dims_lhs = array<i64: 0, 1>, contract_dims_rhs = array<i64: 0, 1>}
    // CHECK-SAME: (tensor<4x10x1xf32>, tensor<4x10x2xf32>) -> tensor<1x2xf32>
    return %0 : tensor<1x2xf32>
  }
}
