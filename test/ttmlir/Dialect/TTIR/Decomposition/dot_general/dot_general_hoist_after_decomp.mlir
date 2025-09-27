// RUN: ttmlir-opt --ttir-to-ttir-decomposition -o %t %s
// RUN: FileCheck %s --input-file=%t

// Test that when a DotGeneralOp with ttir.should_hoist is decomposed,
// all resulting operations also get the ttir.should_hoist attribute.

func.func @test_dot_general_hoist_propagation(%arg0: tensor<4x8xf32>, %arg1: tensor<8x4xf32>) -> tensor<4x4xf32> {
  // CHECK-NOT: ttir.dot_general
  // CHECK: ttir.permute{{.*}}{ttir.should_hoist}
  // CHECK: ttir.permute{{.*}}{ttir.should_hoist}
  // CHECK: ttir.reshape{{.*}}{ttir.should_hoist}
  // CHECK: ttir.reshape{{.*}}{ttir.should_hoist}
  // CHECK: ttir.matmul{{.*}}{ttir.should_hoist}
  // CHECK: ttir.reshape{{.*}}{ttir.should_hoist}
  %0 = "ttir.dot_general"(%arg0, %arg1)
       <{ batch_dims_lhs = array<i64>,
          batch_dims_rhs = array<i64>,
          contract_dims_lhs = array<i64: 1>,
          contract_dims_rhs = array<i64: 0> }>
       {ttir.should_hoist}
       : (tensor<4x8xf32>, tensor<8x4xf32>) -> tensor<4x4xf32>

  return %0 : tensor<4x4xf32>
}
