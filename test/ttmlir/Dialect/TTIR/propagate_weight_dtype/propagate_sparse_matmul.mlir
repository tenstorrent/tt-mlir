// RUN: ttmlir-opt --ttir-propagate-weight-dtype %s | FileCheck %s

// Test that weight_dtype propagates directly to a sparse_matmul op.

module {
  // CHECK-LABEL: func.func @propagate_sparse_matmul
  func.func @propagate_sparse_matmul(
    %arg0: tensor<2x4x32x2880xbf16>,
    %arg1: tensor<1x4x2880x5760xbf16> {ttcore.argument_type = #ttcore.argument_type<parameter>, ttcore.weight_dtype = "bfp_bf4"},
    %arg2: tensor<2x4x1x4xbf16>
  ) -> tensor<2x4x1x4x32x5760xbf16> {
    // CHECK: "ttir.sparse_matmul"
    // CHECK-SAME: ttcore.weight_dtype = "bfp_bf4"
    %0 = "ttir.sparse_matmul"(%arg0, %arg1, %arg2) <{is_input_a_sparse = false, is_input_b_sparse = true, nnz = 0 : i64}> : (tensor<2x4x32x2880xbf16>, tensor<1x4x2880x5760xbf16>, tensor<2x4x1x4xbf16>) -> tensor<2x4x1x4x32x5760xbf16>
    return %0 : tensor<2x4x1x4x32x5760xbf16>
  }
}
