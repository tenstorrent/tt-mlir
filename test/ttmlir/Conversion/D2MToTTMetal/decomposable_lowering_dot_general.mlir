// RUN: ttmlir-opt --ttcore-register-device --ttir-to-ttmetal-pipeline="dst-allocation-strategy=legacy" --convert-d2m-to-ttmetal --canonicalize %s | FileCheck %s --check-prefixes=COMMON,LEGACY
// RUN: ttmlir-opt --ttcore-register-device --ttir-to-ttmetal-pipeline="dst-allocation-strategy=graph-coloring-greedy" --convert-d2m-to-ttmetal --canonicalize %s | FileCheck %s --check-prefixes=COMMON,GREEDY
// RUN: ttmlir-opt --ttcore-register-device --ttir-to-ttmetal-pipeline="dst-allocation-strategy=graph-coloring-cb" --convert-d2m-to-ttmetal --canonicalize %s | FileCheck %s --check-prefixes=COMMON,CB
// Higher-dimension tests requires:
//   Permute: https://github.com/tenstorrent/tt-mlir/issues/3025
//   Reshape: https://github.com/tenstorrent/tt-mlir/issues/3027

module {
  func.func @test_dot_general_2d_2d(%arg0: tensor<64x32xf32>, %arg1: tensor<32x128xf32>) -> tensor<64x128xf32> {
    // COMMON-NOT: ttir.dot_general
    // COMMON: mm_block_init
    // COMMON: matmul_block
    %0 = "ttir.dot_general"(%arg0, %arg1) <{batch_dims_lhs = array<i64>, batch_dims_rhs = array<i64>, contract_dims_lhs = array<i64: 1>, contract_dims_rhs = array<i64: 0>}> : (tensor<64x32xf32>, tensor<32x128xf32>) -> tensor<64x128xf32>
    return %0 : tensor<64x128xf32>
  }
}
