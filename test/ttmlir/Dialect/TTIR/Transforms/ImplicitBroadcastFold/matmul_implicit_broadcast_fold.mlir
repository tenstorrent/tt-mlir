// RUN: ttmlir-opt --ttir-implicit-broadcast-fold -o %t %s
// RUN: FileCheck --input-file=%t %s

module {
  // The supported case: a rank-4 RHS whose batch dim 1 is broadcast to match
  // the LHS. TTNN's matmul broadcasts dim 1 implicitly, so the explicit
  // ttir.broadcast is dropped and the matmul consumes the unbroadcast operand.
  func.func @fold_rhs_batch_dim1(%arg0: tensor<1x256x128x128xf32>, %arg1: tensor<1x1x128x128xf32>) -> tensor<1x256x128x128xf32> {
    %0 = "ttir.broadcast"(%arg1) <{broadcast_dimensions = array<i64: 1, 256, 1, 1>}> : (tensor<1x1x128x128xf32>) -> tensor<1x256x128x128xf32>
    %1 = "ttir.matmul"(%arg0, %0) <{transpose_a = false, transpose_b = false}> : (tensor<1x256x128x128xf32>, tensor<1x256x128x128xf32>) -> tensor<1x256x128x128xf32>
    // CHECK-LABEL: func.func @fold_rhs_batch_dim1
    // CHECK-NOT: "ttir.broadcast"
    // CHECK: "ttir.matmul"(%arg0, %arg1)
    // CHECK-SAME: (tensor<1x256x128x128xf32>, tensor<1x1x128x128xf32>) -> tensor<1x256x128x128xf32>
    return %1 : tensor<1x256x128x128xf32>
  }

  // Negative: a broadcast on the LHS is out of scope -- the fold only targets
  // the RHS -- so it must be kept.
  func.func @no_fold_lhs_batch_dim1(%arg0: tensor<1x1x128x128xf32>, %arg1: tensor<1x256x128x128xf32>) -> tensor<1x256x128x128xf32> {
    %0 = "ttir.broadcast"(%arg0) <{broadcast_dimensions = array<i64: 1, 256, 1, 1>}> : (tensor<1x1x128x128xf32>) -> tensor<1x256x128x128xf32>
    %1 = "ttir.matmul"(%0, %arg1) <{transpose_a = false, transpose_b = false}> : (tensor<1x256x128x128xf32>, tensor<1x256x128x128xf32>) -> tensor<1x256x128x128xf32>
    // CHECK-LABEL: func.func @no_fold_lhs_batch_dim1
    // CHECK: "ttir.broadcast"
    // CHECK: "ttir.matmul"
    return %1 : tensor<1x256x128x128xf32>
  }

  // Negative: rank-3 broadcast on dim 0. TTNN's matmul cannot broadcast dim 0,
  // so the broadcast must be kept (this is the shape that previously faulted).
  func.func @no_fold_rank3_batch_dim0(%arg0: tensor<1x544x2880xf32>, %arg1: tensor<1x2880x5760xf32>) -> tensor<4x544x5760xf32> {
    %0 = "ttir.broadcast"(%arg1) <{broadcast_dimensions = array<i64: 4, 1, 1>}> : (tensor<1x2880x5760xf32>) -> tensor<4x2880x5760xf32>
    %1 = "ttir.matmul"(%arg0, %0) <{transpose_a = false, transpose_b = false}> : (tensor<1x544x2880xf32>, tensor<4x2880x5760xf32>) -> tensor<4x544x5760xf32>
    // CHECK-LABEL: func.func @no_fold_rank3_batch_dim0
    // CHECK: "ttir.broadcast"
    // CHECK: "ttir.matmul"
    return %1 : tensor<4x544x5760xf32>
  }

  // Negative: a rank-4 broadcast on dim 0. The fold only targets dim 1, so a
  // broadcast that touches any other dim must be kept.
  func.func @no_fold_rank4_batch_dim0(%arg0: tensor<4x256x128x128xf32>, %arg1: tensor<1x256x128x128xf32>) -> tensor<4x256x128x128xf32> {
    %0 = "ttir.broadcast"(%arg1) <{broadcast_dimensions = array<i64: 4, 1, 1, 1>}> : (tensor<1x256x128x128xf32>) -> tensor<4x256x128x128xf32>
    %1 = "ttir.matmul"(%arg0, %0) <{transpose_a = false, transpose_b = false}> : (tensor<4x256x128x128xf32>, tensor<4x256x128x128xf32>) -> tensor<4x256x128x128xf32>
    // CHECK-LABEL: func.func @no_fold_rank4_batch_dim0
    // CHECK: "ttir.broadcast"
    // CHECK: "ttir.matmul"
    return %1 : tensor<4x256x128x128xf32>
  }

  // Negative: the broadcast is on dim 1 of the RHS, but the LHS dim 1 is itself
  // size 1 -- so the matmul's dim-1 output comes entirely from the RHS
  // broadcast. Dropping it would shrink the result from 256 to 1, so it must be
  // kept. This is why "broadcast on dim 1 of the RHS" alone is not sufficient.
  func.func @no_fold_lhs_dim1_is_one(%arg0: tensor<1x1x128x128xf32>, %arg1: tensor<1x1x128x128xf32>) -> tensor<1x256x128x128xf32> {
    %0 = "ttir.broadcast"(%arg1) <{broadcast_dimensions = array<i64: 1, 256, 1, 1>}> : (tensor<1x1x128x128xf32>) -> tensor<1x256x128x128xf32>
    %1 = "ttir.matmul"(%arg0, %0) <{transpose_a = false, transpose_b = false}> : (tensor<1x1x128x128xf32>, tensor<1x256x128x128xf32>) -> tensor<1x256x128x128xf32>
    // CHECK-LABEL: func.func @no_fold_lhs_dim1_is_one
    // CHECK: "ttir.broadcast"
    // CHECK: "ttir.matmul"
    return %1 : tensor<1x256x128x128xf32>
  }

  // Negative: dim 1 is broadcast and the LHS supplies its size, but the outer
  // batch dim 0 is > 1. After folding the RHS would be 4x1x128x128, whose batch
  // size (4) is not 1, so metal would NOT implicitly broadcast the RHS batch and
  // would instead fault with "batch dimension 1 mismatch: a=8 vs b=1". The
  // broadcast must be kept. This is the real-world shape the tightening protects.
  func.func @no_fold_outer_batch_dim0_gt_one(%arg0: tensor<4x8x128x128xf32>, %arg1: tensor<4x1x128x128xf32>) -> tensor<4x8x128x128xf32> {
    %0 = "ttir.broadcast"(%arg1) <{broadcast_dimensions = array<i64: 1, 8, 1, 1>}> : (tensor<4x1x128x128xf32>) -> tensor<4x8x128x128xf32>
    %1 = "ttir.matmul"(%arg0, %0) <{transpose_a = false, transpose_b = false}> : (tensor<4x8x128x128xf32>, tensor<4x8x128x128xf32>) -> tensor<4x8x128x128xf32>
    // CHECK-LABEL: func.func @no_fold_outer_batch_dim0_gt_one
    // CHECK: "ttir.broadcast"
    // CHECK: "ttir.matmul"
    return %1 : tensor<4x8x128x128xf32>
  }
}
