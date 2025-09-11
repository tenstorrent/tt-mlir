// RUN: ttmlir-opt --ttir-to-ttmetal-pipeline -o %t.mlir %s
// RUN: FileCheck %s --input-file=%t.mlir
// Higher-dimension tests requires:
//   Permute: https://github.com/tenstorrent/tt-mlir/issues/3025
//   Reshape: https://github.com/tenstorrent/tt-mlir/issues/3027

module {
  func.func @test_dot_general_2d_2d(%arg0: tensor<64x32xf32>, %arg1: tensor<32x128xf32>) -> tensor<64x128xf32> {
    // CHECK-NOT: ttir.dot_general
    // CHECK: mm_block_init
    // CHECK: matmul_block
    %0 = "ttir.dot_general"(%arg0, %arg1) <{batch_dims_lhs = array<i64>, batch_dims_rhs = array<i64>, contract_dims_lhs = array<i64: 1>, contract_dims_rhs = array<i64: 0>}> : (tensor<64x32xf32>, tensor<32x128xf32>) -> tensor<64x128xf32>
    return %0 : tensor<64x128xf32>
  }
}
