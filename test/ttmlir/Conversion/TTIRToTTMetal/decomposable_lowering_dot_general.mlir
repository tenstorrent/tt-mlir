// RUN: ttmlir-opt --ttir-to-ttmetal-backend-pipeline %s > %t.mlir
// RUN: FileCheck %s --input-file=%t.mlir

module {
  func.func @test_dot_general_2d_2d(%arg0: tensor<64x32xf32>, %arg1: tensor<32x128xf32>) -> tensor<64x128xf32> {
    // CHECK-NOT: ttir.dot_general
    // CHECK: mm_init
    // CHECK: matmul_tiles
    %0 = "ttir.dot_general"(%arg0, %arg1) <{batch_dims_lhs = array<i64>, batch_dims_rhs = array<i64>, contract_dims_lhs = array<i64: 1>, contract_dims_rhs = array<i64: 0>}> : (tensor<64x32xf32>, tensor<32x128xf32>) -> tensor<64x128xf32>
    return %0 : tensor<64x128xf32>
  }
}
