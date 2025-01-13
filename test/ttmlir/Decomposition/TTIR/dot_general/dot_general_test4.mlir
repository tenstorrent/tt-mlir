// RUN: ttmlir-opt --ttir-to-ttir-decomposition %s | FileCheck %s
module @jit_dot_general {
  func.func public @test_dot_general(%arg0: tensor<4x10x3x5x7xf32>, %arg1: tensor<4x10x5x7x3xf32>) -> tensor<4x10x3x7x10x7x3xf32> {
    %0 = "ttir.dot_general"(%arg0, %arg1) <{batch_dims_lhs = array<i64: 0>, batch_dims_rhs = array<i64: 0>, contract_dims_lhs = array<i64: 3>, contract_dims_rhs = array<i64: 2>}> : (tensor<4x10x3x5x7xf32>, tensor<4x10x5x7x3xf32>) -> tensor<4x10x3x7x10x7x3xf32>
    // CHECK: "ttir.matmul"
    // CHECK: (tensor<4x210x5xf32>, tensor<4x5x210xf32>, tensor<4x210x210xf32>) -> tensor<4x210x210xf32>
    // CHECK: "ttir.reshape"
    // CHECK: (tensor<4x210x210xf32>, tensor<4x10x3x7x10x7x3xf32>) -> tensor<4x10x3x7x10x7x3xf32>
    return %0 : tensor<4x10x3x7x10x7x3xf32>
  }
}
