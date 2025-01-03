// RUN: ttmlir-opt --ttir-to-ttir-decomposition %s | FileCheck %s
module @jit_dot_general {
  func.func public @test_dot_general(%arg0: tensor<4x10x1xf32>, %arg1: tensor<4x10x2xf32>) -> tensor<1x2xf32> {
    %0 = "ttir.dot_general"(%arg0, %arg1) <{batch_dims_a = array<i64>, batch_dims_b = array<i64>, contract_dims_a = array<i64: 0, 1>, contract_dims_b = array<i64: 0, 1>}> : (tensor<4x10x1xf32>, tensor<4x10x2xf32>) -> tensor<1x2xf32>
    // CHECK: "ttir.permute"
    // CHECK: {permutation = array<i64: 2, 0, 1>}> : (tensor<4x10x1xf32>, tensor<1x4x10xf32>) -> tensor<1x4x10xf32>
    // CHECK: "ttir.reshape"
    // CHECK: {shape = [1 : i32, 40 : i32]}> : (tensor<1x4x10xf32>, tensor<1x40xf32>) -> tensor<1x40xf32>
    // CHECK: "ttir.matmul"
    // CHECK: (tensor<1x40xf32>, tensor<40x2xf32>, tensor<1x2xf32>) -> tensor<1x2xf32>
    return %0 : tensor<1x2xf32>
  }
}
