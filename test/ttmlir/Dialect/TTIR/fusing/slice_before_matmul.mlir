// RUN: ttmlir-opt --ttir-fusing %s -o %t.mlir
// RUN: FileCheck %s --input-file=%t.mlir

// An output-row slice of a matmul is pushed up into the LHS operand, so the
// matmul only computes the rows that are actually used.

// Direct slice of the last row of a 2D matmul output (greedy-decode lm_head
// after the rank-3 logits tensor has been collapsed to 2D).
// CHECK-LABEL: func.func @left_matrix
func.func @left_matrix(%arg0: tensor<1024x4096xbf16>, %arg1: tensor<4096x128256xbf16>) -> tensor<1x128256xbf16> {
  // The LHS is sliced to its last row first, then the matmul is narrowed.
  // CHECK: %[[A:.*]] = "ttir.slice_static"(%arg0) <{begins = [1023 : i32, 0 : i32], ends = [1024 : i32, 4096 : i32], step = [1 : i32, 1 : i32]}> : (tensor<1024x4096xbf16>) -> tensor<1x4096xbf16>
  // CHECK: "ttir.matmul"(%[[A]], %arg1) <{transpose_a = false, transpose_b = false}> : (tensor<1x4096xbf16>, tensor<4096x128256xbf16>) -> tensor<1x128256xbf16>
  // CHECK-NOT: tensor<1024x128256xbf16>
  %0 = "ttir.matmul"(%arg0, %arg1) : (tensor<1024x4096xbf16>, tensor<4096x128256xbf16>) -> tensor<1024x128256xbf16>
  %1 = "ttir.slice_static"(%0) <{begins = [1023 : i32, 0 : i32], ends = [1024 : i32, 128256 : i32], step = [1 : i32, 1 : i32]}> : (tensor<1024x128256xbf16>) -> tensor<1x128256xbf16>
  return %1 : tensor<1x128256xbf16>
}

// Same, but with a leading-unit-dim reshape between matmul and slice (the
// rank-3 logits tensor [1, seq, vocab] produced by HF causal-LM heads).
// CHECK-LABEL: func.func @left_matrix_reshape
func.func @left_matrix_reshape(%arg0: tensor<1024x4096xbf16>, %arg1: tensor<4096x128256xbf16>) -> tensor<1x1x128256xbf16> {
  // CHECK: %[[A:.*]] = "ttir.slice_static"(%arg0) <{begins = [1023 : i32, 0 : i32], ends = [1024 : i32, 4096 : i32], step = [1 : i32, 1 : i32]}> : (tensor<1024x4096xbf16>) -> tensor<1x4096xbf16>
  // CHECK: %[[M:.*]] = "ttir.matmul"(%[[A]], %arg1) <{transpose_a = false, transpose_b = false}> : (tensor<1x4096xbf16>, tensor<4096x128256xbf16>) -> tensor<1x128256xbf16>
  // CHECK: "ttir.reshape"(%[[M]]) <{shape = [1 : i32, 1 : i32, 128256 : i32]}>
  // CHECK-NOT: tensor<1024x128256xbf16>
  %0 = "ttir.matmul"(%arg0, %arg1) : (tensor<1024x4096xbf16>, tensor<4096x128256xbf16>) -> tensor<1024x128256xbf16>
  %1 = "ttir.reshape"(%0) <{shape = [1 : i32, 1024 : i32, 128256 : i32]}> : (tensor<1024x128256xbf16>) -> tensor<1x1024x128256xbf16>
  %2 = "ttir.slice_static"(%1) <{begins = [0 : i32, 1023 : i32, 0 : i32], ends = [1 : i32, 1024 : i32, 128256 : i32], step = [1 : i32, 1 : i32, 1 : i32]}> : (tensor<1x1024x128256xbf16>) -> tensor<1x1x128256xbf16>
  return %2 : tensor<1x1x128256xbf16>
}

// Negative: the matmul result has another user, so narrowing would force the
// full matmul to be recomputed. Leave it alone.
// CHECK-LABEL: func.func @left_matrix_multiple_uses_negative
func.func @left_matrix_multiple_uses_negative(%arg0: tensor<1024x4096xbf16>, %arg1: tensor<4096x128256xbf16>) -> (tensor<1x128256xbf16>, tensor<1024x128256xbf16>) {
  // CHECK: "ttir.matmul"(%arg0, %arg1) <{transpose_a = false, transpose_b = false}> : (tensor<1024x4096xbf16>, tensor<4096x128256xbf16>) -> tensor<1024x128256xbf16>
  %0 = "ttir.matmul"(%arg0, %arg1) : (tensor<1024x4096xbf16>, tensor<4096x128256xbf16>) -> tensor<1024x128256xbf16>
  %1 = "ttir.slice_static"(%0) <{begins = [1023 : i32, 0 : i32], ends = [1024 : i32, 128256 : i32], step = [1 : i32, 1 : i32]}> : (tensor<1024x128256xbf16>) -> tensor<1x128256xbf16>
  return %1, %0 : tensor<1x128256xbf16>, tensor<1024x128256xbf16>
}

// A column (N) slice of the output is pushed up into the RHS (B): only the
// selected output columns are computed, narrowing B's trailing dim.
// CHECK-LABEL: func.func @right_matrix
func.func @right_matrix(%arg0: tensor<1024x4096xbf16>, %arg1: tensor<4096x128256xbf16>) -> tensor<1024x64xbf16> {
  // CHECK: %[[B:.*]] = "ttir.slice_static"(%arg1) <{begins = [0 : i32, 0 : i32], ends = [4096 : i32, 64 : i32], step = [1 : i32, 1 : i32]}> : (tensor<4096x128256xbf16>) -> tensor<4096x64xbf16>
  // CHECK: "ttir.matmul"(%arg0, %[[B]]) <{transpose_a = false, transpose_b = false}> : (tensor<1024x4096xbf16>, tensor<4096x64xbf16>) -> tensor<1024x64xbf16>
  // CHECK-NOT: tensor<1024x128256xbf16>
  %0 = "ttir.matmul"(%arg0, %arg1) : (tensor<1024x4096xbf16>, tensor<4096x128256xbf16>) -> tensor<1024x128256xbf16>
  %1 = "ttir.slice_static"(%0) <{begins = [0 : i32, 0 : i32], ends = [1024 : i32, 64 : i32], step = [1 : i32, 1 : i32]}> : (tensor<1024x128256xbf16>) -> tensor<1024x64xbf16>
  return %1 : tensor<1024x64xbf16>
}

// A middle column block ([5:17]) is pushed up into the RHS just like a single
// trailing column slice.
// CHECK-LABEL: func.func @right_matrix_middle_cols
func.func @right_matrix_middle_cols(%arg0: tensor<1024x4096xbf16>, %arg1: tensor<4096x128256xbf16>) -> tensor<1024x12xbf16> {
  // CHECK: %[[B:.*]] = "ttir.slice_static"(%arg1) <{begins = [0 : i32, 5 : i32], ends = [4096 : i32, 17 : i32], step = [1 : i32, 1 : i32]}> : (tensor<4096x128256xbf16>) -> tensor<4096x12xbf16>
  // CHECK: "ttir.matmul"(%arg0, %[[B]]) <{transpose_a = false, transpose_b = false}> : (tensor<1024x4096xbf16>, tensor<4096x12xbf16>) -> tensor<1024x12xbf16>
  // CHECK-NOT: tensor<1024x128256xbf16>
  %0 = "ttir.matmul"(%arg0, %arg1) : (tensor<1024x4096xbf16>, tensor<4096x128256xbf16>) -> tensor<1024x128256xbf16>
  %1 = "ttir.slice_static"(%0) <{begins = [0 : i32, 5 : i32], ends = [1024 : i32, 17 : i32], step = [1 : i32, 1 : i32]}> : (tensor<1024x128256xbf16>) -> tensor<1024x12xbf16>
  return %1 : tensor<1024x12xbf16>
}

// transpose_b=true: B is [N, K], so the output columns come from B's *rank-2*
// dim. The column slice is pushed into that dim, and transpose_b is preserved.
// CHECK-LABEL: func.func @right_matrix_transpose
func.func @right_matrix_transpose(%arg0: tensor<1024x4096xbf16>, %arg1: tensor<128256x4096xbf16>) -> tensor<1024x64xbf16> {
  // CHECK: %[[B:.*]] = "ttir.slice_static"(%arg1) <{begins = [0 : i32, 0 : i32], ends = [64 : i32, 4096 : i32], step = [1 : i32, 1 : i32]}> : (tensor<128256x4096xbf16>) -> tensor<64x4096xbf16>
  // CHECK: "ttir.matmul"(%arg0, %[[B]]) <{transpose_a = false, transpose_b = true}> : (tensor<1024x4096xbf16>, tensor<64x4096xbf16>) -> tensor<1024x64xbf16>
  // CHECK-NOT: tensor<1024x128256xbf16>
  %0 = "ttir.matmul"(%arg0, %arg1) <{transpose_a = false, transpose_b = true}> : (tensor<1024x4096xbf16>, tensor<128256x4096xbf16>) -> tensor<1024x128256xbf16>
  %1 = "ttir.slice_static"(%0) <{begins = [0 : i32, 0 : i32], ends = [1024 : i32, 64 : i32], step = [1 : i32, 1 : i32]}> : (tensor<1024x128256xbf16>) -> tensor<1024x64xbf16>
  return %1 : tensor<1024x64xbf16>
}

// Negative: the slice narrows both the row and column dims at once. The pattern
// only pushes a single narrowed dim into one operand, so this is left alone.
// CHECK-LABEL: func.func @both_matrix_negative
func.func @both_matrix_negative(%arg0: tensor<1024x4096xbf16>, %arg1: tensor<4096x128256xbf16>) -> tensor<1x64xbf16> {
  // CHECK: "ttir.matmul"(%arg0, %arg1) <{transpose_a = false, transpose_b = false}> : (tensor<1024x4096xbf16>, tensor<4096x128256xbf16>) -> tensor<1024x128256xbf16>
  %0 = "ttir.matmul"(%arg0, %arg1) : (tensor<1024x4096xbf16>, tensor<4096x128256xbf16>) -> tensor<1024x128256xbf16>
  %1 = "ttir.slice_static"(%0) <{begins = [1023 : i32, 0 : i32], ends = [1024 : i32, 64 : i32], step = [1 : i32, 1 : i32]}> : (tensor<1024x128256xbf16>) -> tensor<1x64xbf16>
  return %1 : tensor<1x64xbf16>
}

// A contiguous block of rows in the middle of the output ([5:17]) is pushed up
// just like a single trailing row: only the selected rows feed the matmul.
// CHECK-LABEL: func.func @left_matrix_middle_rows
func.func @left_matrix_middle_rows(%arg0: tensor<1024x4096xbf16>, %arg1: tensor<4096x128256xbf16>) -> tensor<12x128256xbf16> {
  // CHECK: %[[A:.*]] = "ttir.slice_static"(%arg0) <{begins = [5 : i32, 0 : i32], ends = [17 : i32, 4096 : i32], step = [1 : i32, 1 : i32]}> : (tensor<1024x4096xbf16>) -> tensor<12x4096xbf16>
  // CHECK: "ttir.matmul"(%[[A]], %arg1) <{transpose_a = false, transpose_b = false}> : (tensor<12x4096xbf16>, tensor<4096x128256xbf16>) -> tensor<12x128256xbf16>
  // CHECK-NOT: tensor<1024x128256xbf16>
  %0 = "ttir.matmul"(%arg0, %arg1) : (tensor<1024x4096xbf16>, tensor<4096x128256xbf16>) -> tensor<1024x128256xbf16>
  %1 = "ttir.slice_static"(%0) <{begins = [5 : i32, 0 : i32], ends = [17 : i32, 128256 : i32], step = [1 : i32, 1 : i32]}> : (tensor<1024x128256xbf16>) -> tensor<12x128256xbf16>
  return %1 : tensor<12x128256xbf16>
}

// Negative begin/end indices on the row dim are normalized against the dim
// size before being pushed up: rows [-4:-1] of a 1024-row output become the
// concrete range [1020:1023] on A.
// CHECK-LABEL: func.func @left_matrix_negative_indices
func.func @left_matrix_negative_indices(%arg0: tensor<1024x4096xbf16>, %arg1: tensor<4096x128256xbf16>) -> tensor<3x128256xbf16> {
  // CHECK: %[[A:.*]] = "ttir.slice_static"(%arg0) <{begins = [1020 : i32, 0 : i32], ends = [1023 : i32, 4096 : i32], step = [1 : i32, 1 : i32]}> : (tensor<1024x4096xbf16>) -> tensor<3x4096xbf16>
  // CHECK: "ttir.matmul"(%[[A]], %arg1) <{transpose_a = false, transpose_b = false}> : (tensor<3x4096xbf16>, tensor<4096x128256xbf16>) -> tensor<3x128256xbf16>
  // CHECK-NOT: tensor<1024x128256xbf16>
  %0 = "ttir.matmul"(%arg0, %arg1) : (tensor<1024x4096xbf16>, tensor<4096x128256xbf16>) -> tensor<1024x128256xbf16>
  %1 = "ttir.slice_static"(%0) <{begins = [-4 : i32, 0 : i32], ends = [-1 : i32, 128256 : i32], step = [1 : i32, 1 : i32]}> : (tensor<1024x128256xbf16>) -> tensor<3x128256xbf16>
  return %1 : tensor<3x128256xbf16>
}

// Batched 3D x 3D matmul: the leading batch dim is kept full while the output
// row dim (rank-2) is sliced and pushed into A's matching dim.
// CHECK-LABEL: func.func @left_matrix_batched
func.func @left_matrix_batched(%arg0: tensor<2x1024x4096xbf16>, %arg1: tensor<2x4096x128256xbf16>) -> tensor<2x1x128256xbf16> {
  // CHECK: %[[A:.*]] = "ttir.slice_static"(%arg0) <{begins = [0 : i32, 1023 : i32, 0 : i32], ends = [2 : i32, 1024 : i32, 4096 : i32], step = [1 : i32, 1 : i32, 1 : i32]}> : (tensor<2x1024x4096xbf16>) -> tensor<2x1x4096xbf16>
  // CHECK: "ttir.matmul"(%[[A]], %arg1) <{transpose_a = false, transpose_b = false}> : (tensor<2x1x4096xbf16>, tensor<2x4096x128256xbf16>) -> tensor<2x1x128256xbf16>
  // CHECK-NOT: tensor<2x1024x128256xbf16>
  %0 = "ttir.matmul"(%arg0, %arg1) : (tensor<2x1024x4096xbf16>, tensor<2x4096x128256xbf16>) -> tensor<2x1024x128256xbf16>
  %1 = "ttir.slice_static"(%0) <{begins = [0 : i32, 1023 : i32, 0 : i32], ends = [2 : i32, 1024 : i32, 128256 : i32], step = [1 : i32, 1 : i32, 1 : i32]}> : (tensor<2x1024x128256xbf16>) -> tensor<2x1x128256xbf16>
  return %1 : tensor<2x1x128256xbf16>
}

// transpose_a=true: A is [K, M], so the output rows come from A's *last* dim.
// The row slice is pushed into that trailing dim, and transpose_a is preserved.
// CHECK-LABEL: func.func @left_matrix_transpose
func.func @left_matrix_transpose(%arg0: tensor<4096x1024xbf16>, %arg1: tensor<4096x128256xbf16>) -> tensor<1x128256xbf16> {
  // CHECK: %[[A:.*]] = "ttir.slice_static"(%arg0) <{begins = [0 : i32, 1023 : i32], ends = [4096 : i32, 1024 : i32], step = [1 : i32, 1 : i32]}> : (tensor<4096x1024xbf16>) -> tensor<4096x1xbf16>
  // CHECK: "ttir.matmul"(%[[A]], %arg1) <{transpose_a = true, transpose_b = false}> : (tensor<4096x1xbf16>, tensor<4096x128256xbf16>) -> tensor<1x128256xbf16>
  // CHECK-NOT: tensor<1024x128256xbf16>
  %0 = "ttir.matmul"(%arg0, %arg1) <{transpose_a = true, transpose_b = false}> : (tensor<4096x1024xbf16>, tensor<4096x128256xbf16>) -> tensor<1024x128256xbf16>
  %1 = "ttir.slice_static"(%0) <{begins = [1023 : i32, 0 : i32], ends = [1024 : i32, 128256 : i32], step = [1 : i32, 1 : i32]}> : (tensor<1024x128256xbf16>) -> tensor<1x128256xbf16>
  return %1 : tensor<1x128256xbf16>
}

// Negative: the reshape is a "real" reshape that rearranges the matmul output
// dims (here [1024, 128256] -> [1, 2048, 64128]). Even though it prepends a
// leading 1, the trailing dims no longer match the matmul shape, so the slice
// dims do not map 1:1 onto the matmul output dims and the fusion is skipped.
// CHECK-LABEL: func.func @left_matrix_real_reshape_negative
func.func @left_matrix_real_reshape_negative(%arg0: tensor<1024x4096xbf16>, %arg1: tensor<4096x128256xbf16>) -> tensor<1x1x64128xbf16> {
  // CHECK: "ttir.matmul"(%arg0, %arg1) <{transpose_a = false, transpose_b = false}> : (tensor<1024x4096xbf16>, tensor<4096x128256xbf16>) -> tensor<1024x128256xbf16>
  %0 = "ttir.matmul"(%arg0, %arg1) : (tensor<1024x4096xbf16>, tensor<4096x128256xbf16>) -> tensor<1024x128256xbf16>
  %1 = "ttir.reshape"(%0) <{shape = [1 : i32, 2048 : i32, 64128 : i32]}> : (tensor<1024x128256xbf16>) -> tensor<1x2048x64128xbf16>
  %2 = "ttir.slice_static"(%1) <{begins = [0 : i32, 2047 : i32, 0 : i32], ends = [1 : i32, 2048 : i32, 64128 : i32], step = [1 : i32, 1 : i32, 1 : i32]}> : (tensor<1x2048x64128xbf16>) -> tensor<1x1x64128xbf16>
  return %2 : tensor<1x1x64128xbf16>
}

// Negative: the reshape inflates the row dim into a leading non-unit dim
// ([1024, 128256] -> [2, 512, 128256]). The leading pad dim is 2, not 1, so the
// slice dims cannot be mapped back to the matmul output and fusion is skipped.
// CHECK-LABEL: func.func @left_matrix_reshape_inflate_leading_negative
func.func @left_matrix_reshape_inflate_leading_negative(%arg0: tensor<1024x4096xbf16>, %arg1: tensor<4096x128256xbf16>) -> tensor<1x1x128256xbf16> {
  // CHECK: "ttir.matmul"(%arg0, %arg1) <{transpose_a = false, transpose_b = false}> : (tensor<1024x4096xbf16>, tensor<4096x128256xbf16>) -> tensor<1024x128256xbf16>
  %0 = "ttir.matmul"(%arg0, %arg1) : (tensor<1024x4096xbf16>, tensor<4096x128256xbf16>) -> tensor<1024x128256xbf16>
  %1 = "ttir.reshape"(%0) <{shape = [2 : i32, 512 : i32, 128256 : i32]}> : (tensor<1024x128256xbf16>) -> tensor<2x512x128256xbf16>
  %2 = "ttir.slice_static"(%1) <{begins = [0 : i32, 511 : i32, 0 : i32], ends = [1 : i32, 512 : i32, 128256 : i32], step = [1 : i32, 1 : i32, 1 : i32]}> : (tensor<2x512x128256xbf16>) -> tensor<1x1x128256xbf16>
  return %2 : tensor<1x1x128256xbf16>
}

// Negative: the reshape inflates a non-unit dim in the middle of the leading
// pad region ([1024, 128256] -> [1, 2, 512, 128256]). A pad dim (the 2) is not
// 1, so the slice dims do not line up with the matmul output and fusion is
// skipped.
// CHECK-LABEL: func.func @left_matrix_reshape_inflate_middle_negative
func.func @left_matrix_reshape_inflate_middle_negative(%arg0: tensor<1024x4096xbf16>, %arg1: tensor<4096x128256xbf16>) -> tensor<1x1x1x128256xbf16> {
  // CHECK: "ttir.matmul"(%arg0, %arg1) <{transpose_a = false, transpose_b = false}> : (tensor<1024x4096xbf16>, tensor<4096x128256xbf16>) -> tensor<1024x128256xbf16>
  %0 = "ttir.matmul"(%arg0, %arg1) : (tensor<1024x4096xbf16>, tensor<4096x128256xbf16>) -> tensor<1024x128256xbf16>
  %1 = "ttir.reshape"(%0) <{shape = [1 : i32, 2 : i32, 512 : i32, 128256 : i32]}> : (tensor<1024x128256xbf16>) -> tensor<1x2x512x128256xbf16>
  %2 = "ttir.slice_static"(%1) <{begins = [0 : i32, 0 : i32, 511 : i32, 0 : i32], ends = [1 : i32, 1 : i32, 512 : i32, 128256 : i32], step = [1 : i32, 1 : i32, 1 : i32, 1 : i32]}> : (tensor<1x2x512x128256xbf16>) -> tensor<1x1x1x128256xbf16>
  return %2 : tensor<1x1x1x128256xbf16>
}

// Negative: a leading-unit dim is *deflated* away by the reshape
// ([1, 1024, 128256] -> [1024, 128256]) so the slice has lower rank than the
// matmul output. The pattern only looks through a reshape that prepends size-1
// dims, not one that removes them, so this is left untouched.
// CHECK-LABEL: func.func @left_matrix_reshape_deflate_negative
func.func @left_matrix_reshape_deflate_negative(%arg0: tensor<1x1024x4096xbf16>, %arg1: tensor<1x4096x128256xbf16>) -> tensor<1x128256xbf16> {
  // CHECK: "ttir.matmul"(%arg0, %arg1) <{transpose_a = false, transpose_b = false}> : (tensor<1x1024x4096xbf16>, tensor<1x4096x128256xbf16>) -> tensor<1x1024x128256xbf16>
  %0 = "ttir.matmul"(%arg0, %arg1) : (tensor<1x1024x4096xbf16>, tensor<1x4096x128256xbf16>) -> tensor<1x1024x128256xbf16>
  %1 = "ttir.reshape"(%0) <{shape = [1024 : i32, 128256 : i32]}> : (tensor<1x1024x128256xbf16>) -> tensor<1024x128256xbf16>
  %2 = "ttir.slice_static"(%1) <{begins = [1023 : i32, 0 : i32], ends = [1024 : i32, 128256 : i32], step = [1 : i32, 1 : i32]}> : (tensor<1024x128256xbf16>) -> tensor<1x128256xbf16>
  return %2 : tensor<1x128256xbf16>
}
