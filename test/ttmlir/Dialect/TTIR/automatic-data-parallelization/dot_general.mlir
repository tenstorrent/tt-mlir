// RUN: ttmlir-opt --ttir-automatic-data-parallelization="mesh-shape=1,2" %s > %t.mlir
// RUN: FileCheck %s --input-file=%t.mlir

func.func @test0(%arg0: tensor<16x32xf32>, %arg1: tensor<8x32xf32>) -> tensor<16x8xf32> {
  %0 = "ttir.dot_general"(%arg0, %arg1) <{batch_dims_lhs = array<i64>, batch_dims_rhs = array<i64>, contract_dims_lhs = array<i64: 1>, contract_dims_rhs = array<i64: 1>}> : (tensor<16x32xf32>, tensor<8x32xf32>) -> tensor<16x8xf32>
  return %0 : tensor<16x8xf32>
}

// CHECK: func.func @test0
// CHECK: sdy.sharding = #sdy.sharding<@mesh, [{}, {}]>
// CHECK: sdy.sharding = #sdy.sharding<@mesh, [{}, {}]>

func.func @test1(%arg0: tensor<4x10x1xf32>, %arg1: tensor<4x10x2xf32>) -> tensor<1x2xf32> {
  %0 = "ttir.dot_general"(%arg0, %arg1) <{batch_dims_lhs = array<i64>, batch_dims_rhs = array<i64>, contract_dims_lhs = array<i64: 0, 1>, contract_dims_rhs = array<i64: 0, 1>}> : (tensor<4x10x1xf32>, tensor<4x10x2xf32>) -> tensor<1x2xf32>
  return %0 : tensor<1x2xf32>
}

// CHECK: func.func @test1
// CHECK: sdy.sharding = #sdy.sharding<@mesh, [{}, {}, {}]>
// CHECK: sdy.sharding = #sdy.sharding<@mesh, [{}, {}, {}]>

func.func @test2(%arg0: tensor<32x4x10x1xf32>, %arg1: tensor<1x4x10x2xf32>) -> tensor<32x1x2xf32> {
  %0 = "ttir.dot_general"(%arg0, %arg1) <{batch_dims_lhs = array<i64: 0>, batch_dims_rhs = array<i64: 0>, contract_dims_lhs = array<i64: 1, 2>, contract_dims_rhs = array<i64: 1, 2>}> : (tensor<32x4x10x1xf32>, tensor<1x4x10x2xf32>) -> tensor<32x1x2xf32>
  return %0 : tensor<32x1x2xf32>
}

// CHECK: func.func @test2
// CHECK: sdy.sharding = #sdy.sharding<@mesh, [{"batch"}, {}, {}, {}]>
// CHECK: sdy.sharding = #sdy.sharding<@mesh, [{}, {}, {}, {}]>
// CHECK: sdy.sharding = #sdy.sharding<@mesh, [{"batch", ?}, {?}, {?}]>

// CHECK: "ttir.dot_general"
// CHECK-SAME: sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"batch", ?}, {?}, {?}]>]>

func.func @test3(%arg0: tensor<4x10x3x5x7xf32>, %arg1: tensor<4x10x5x7x3xf32>) -> tensor<4x10x3x7x10x7x3xf32> {
  %0 = "ttir.dot_general"(%arg0, %arg1) <{batch_dims_lhs = array<i64: 0>, batch_dims_rhs = array<i64: 0>, contract_dims_lhs = array<i64: 3>, contract_dims_rhs = array<i64: 2>}> : (tensor<4x10x3x5x7xf32>, tensor<4x10x5x7x3xf32>) -> tensor<4x10x3x7x10x7x3xf32>
  return %0 : tensor<4x10x3x7x10x7x3xf32>
}

// CHECK: func.func @test3
// CHECK: sdy.sharding = #sdy.sharding<@mesh, [{"batch"}, {}, {}, {}, {}]>
// CHECK: sdy.sharding = #sdy.sharding<@mesh, [{"batch"}, {}, {}, {}, {}]>
// CHECK: sdy.sharding = #sdy.sharding<@mesh, [{"batch", ?}, {?}, {?}, {?}, {?}, {?}, {?}]>

// CHECK: "ttir.dot_general"
// CHECK-SAME: sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"batch", ?}, {?}, {?}, {?}, {?}, {?}, {?}]>]>

func.func @test4(%arg0: tensor<16x32xf32>, %arg1: tensor<32x8xf32>) -> tensor<16x8xf32> {
  %0 = "ttir.dot_general"(%arg0, %arg1) <{batch_dims_lhs = array<i64>, batch_dims_rhs = array<i64>, contract_dims_lhs = array<i64: 1>, contract_dims_rhs = array<i64: 0>}> : (tensor<16x32xf32>, tensor<32x8xf32>) -> tensor<16x8xf32>
  return %0 : tensor<16x8xf32>
}

// CHECK: func.func @test4
// CHECK: sdy.sharding = #sdy.sharding<@mesh, [{}, {}]>
// CHECK: sdy.sharding = #sdy.sharding<@mesh, [{}, {}]>
