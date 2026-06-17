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

// Negative: the row slice uses a non-unit step, so the selected rows are not a
// contiguous range and cannot be pushed into A. Leave it alone.
// CHECK-LABEL: func.func @left_matrix_non_unit_step_negative
func.func @left_matrix_non_unit_step_negative(%arg0: tensor<1024x4096xbf16>, %arg1: tensor<4096x128256xbf16>) -> tensor<512x128256xbf16> {
  // CHECK: "ttir.matmul"(%arg0, %arg1) <{transpose_a = false, transpose_b = false}> : (tensor<1024x4096xbf16>, tensor<4096x128256xbf16>) -> tensor<1024x128256xbf16>
  %0 = "ttir.matmul"(%arg0, %arg1) : (tensor<1024x4096xbf16>, tensor<4096x128256xbf16>) -> tensor<1024x128256xbf16>
  %1 = "ttir.slice_static"(%0) <{begins = [0 : i32, 0 : i32], ends = [1024 : i32, 128256 : i32], step = [2 : i32, 1 : i32]}> : (tensor<1024x128256xbf16>) -> tensor<512x128256xbf16>
  return %1 : tensor<512x128256xbf16>
}

// Negative: a 1D operand makes the matmul output rank 1, so there is no row/col
// dim to push the slice into; the output-rank guard leaves it alone.
// CHECK-LABEL: func.func @vector_matmul_negative
func.func @vector_matmul_negative(%arg0: tensor<4096xbf16>, %arg1: tensor<4096x128256xbf16>) -> tensor<64xbf16> {
  // CHECK: "ttir.matmul"(%arg0, %arg1) <{transpose_a = false, transpose_b = false}> : (tensor<4096xbf16>, tensor<4096x128256xbf16>) -> tensor<128256xbf16>
  %0 = "ttir.matmul"(%arg0, %arg1) : (tensor<4096xbf16>, tensor<4096x128256xbf16>) -> tensor<128256xbf16>
  %1 = "ttir.slice_static"(%0) <{begins = [0 : i32], ends = [64 : i32], step = [1 : i32]}> : (tensor<128256xbf16>) -> tensor<64xbf16>
  return %1 : tensor<64xbf16>
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

// transpose_a=true and transpose_b=true together: A is [K, M] and B is [N, K].
// A row slice is pushed into A's trailing (M) dim and both transpose flags are
// preserved.
// CHECK-LABEL: func.func @left_matrix_transpose_both
func.func @left_matrix_transpose_both(%arg0: tensor<4096x1024xbf16>, %arg1: tensor<128256x4096xbf16>) -> tensor<1x128256xbf16> {
  // CHECK: %[[A:.*]] = "ttir.slice_static"(%arg0) <{begins = [0 : i32, 1023 : i32], ends = [4096 : i32, 1024 : i32], step = [1 : i32, 1 : i32]}> : (tensor<4096x1024xbf16>) -> tensor<4096x1xbf16>
  // CHECK: "ttir.matmul"(%[[A]], %arg1) <{transpose_a = true, transpose_b = true}> : (tensor<4096x1xbf16>, tensor<128256x4096xbf16>) -> tensor<1x128256xbf16>
  // CHECK-NOT: tensor<1024x128256xbf16>
  %0 = "ttir.matmul"(%arg0, %arg1) <{transpose_a = true, transpose_b = true}> : (tensor<4096x1024xbf16>, tensor<128256x4096xbf16>) -> tensor<1024x128256xbf16>
  %1 = "ttir.slice_static"(%0) <{begins = [1023 : i32, 0 : i32], ends = [1024 : i32, 128256 : i32], step = [1 : i32, 1 : i32]}> : (tensor<1024x128256xbf16>) -> tensor<1x128256xbf16>
  return %1 : tensor<1x128256xbf16>
}

// The same fusion applies to ttir.linear (matmul + bias). For a row (M) slice
// the slice is pushed into A; the bias is indexed by the N (column) dim, so a
// [vocab] bias is identical across rows and passes through untouched.
// CHECK-LABEL: func.func @linear_left
func.func @linear_left(%arg0: tensor<1024x4096xbf16>, %arg1: tensor<4096x128256xbf16>, %arg2: tensor<128256xbf16>) -> tensor<1x128256xbf16> {
  // CHECK: %[[A:.*]] = "ttir.slice_static"(%arg0) <{begins = [1023 : i32, 0 : i32], ends = [1024 : i32, 4096 : i32], step = [1 : i32, 1 : i32]}> : (tensor<1024x4096xbf16>) -> tensor<1x4096xbf16>
  // CHECK: "ttir.linear"(%[[A]], %arg1, %arg2) <{transpose_a = false, transpose_b = false}> : (tensor<1x4096xbf16>, tensor<4096x128256xbf16>, tensor<128256xbf16>) -> tensor<1x128256xbf16>
  // CHECK-NOT: tensor<1024x128256xbf16>
  %0 = "ttir.linear"(%arg0, %arg1, %arg2) : (tensor<1024x4096xbf16>, tensor<4096x128256xbf16>, tensor<128256xbf16>) -> tensor<1024x128256xbf16>
  %1 = "ttir.slice_static"(%0) <{begins = [1023 : i32, 0 : i32], ends = [1024 : i32, 128256 : i32], step = [1 : i32, 1 : i32]}> : (tensor<1024x128256xbf16>) -> tensor<1x128256xbf16>
  return %1 : tensor<1x128256xbf16>
}

// Row slice with a full 2D bias [M, N]: the bias dim aligned with the sliced M
// dim is real, so the bias is sliced along the same row range as A.
// CHECK-LABEL: func.func @linear_left_bias_2d
func.func @linear_left_bias_2d(%arg0: tensor<1024x4096xbf16>, %arg1: tensor<4096x128256xbf16>, %arg2: tensor<1024x128256xbf16>) -> tensor<1x128256xbf16> {
  // CHECK: %[[A:.*]] = "ttir.slice_static"(%arg0) <{begins = [1023 : i32, 0 : i32], ends = [1024 : i32, 4096 : i32], step = [1 : i32, 1 : i32]}> : (tensor<1024x4096xbf16>) -> tensor<1x4096xbf16>
  // CHECK: %[[BIAS:.*]] = "ttir.slice_static"(%arg2) <{begins = [1023 : i32, 0 : i32], ends = [1024 : i32, 128256 : i32], step = [1 : i32, 1 : i32]}> : (tensor<1024x128256xbf16>) -> tensor<1x128256xbf16>
  // CHECK: "ttir.linear"(%[[A]], %arg1, %[[BIAS]]) <{transpose_a = false, transpose_b = false}> : (tensor<1x4096xbf16>, tensor<4096x128256xbf16>, tensor<1x128256xbf16>) -> tensor<1x128256xbf16>
  %0 = "ttir.linear"(%arg0, %arg1, %arg2) : (tensor<1024x4096xbf16>, tensor<4096x128256xbf16>, tensor<1024x128256xbf16>) -> tensor<1024x128256xbf16>
  %1 = "ttir.slice_static"(%0) <{begins = [1023 : i32, 0 : i32], ends = [1024 : i32, 128256 : i32], step = [1 : i32, 1 : i32]}> : (tensor<1024x128256xbf16>) -> tensor<1x128256xbf16>
  return %1 : tensor<1x128256xbf16>
}

// Row slice with a bias [1, N] (broadcast over M): the bias dim aligned with M
// is size-1, so it already covers every row and is left untouched.
// CHECK-LABEL: func.func @linear_left_bias_broadcast_m
func.func @linear_left_bias_broadcast_m(%arg0: tensor<1024x4096xbf16>, %arg1: tensor<4096x128256xbf16>, %arg2: tensor<1x128256xbf16>) -> tensor<1x128256xbf16> {
  // CHECK: %[[A:.*]] = "ttir.slice_static"(%arg0) <{begins = [1023 : i32, 0 : i32], ends = [1024 : i32, 4096 : i32], step = [1 : i32, 1 : i32]}> : (tensor<1024x4096xbf16>) -> tensor<1x4096xbf16>
  // CHECK: "ttir.linear"(%[[A]], %arg1, %arg2) <{transpose_a = false, transpose_b = false}> : (tensor<1x4096xbf16>, tensor<4096x128256xbf16>, tensor<1x128256xbf16>) -> tensor<1x128256xbf16>
  // CHECK-NOT: tensor<1024x128256xbf16>
  %0 = "ttir.linear"(%arg0, %arg1, %arg2) : (tensor<1024x4096xbf16>, tensor<4096x128256xbf16>, tensor<1x128256xbf16>) -> tensor<1024x128256xbf16>
  %1 = "ttir.slice_static"(%0) <{begins = [1023 : i32, 0 : i32], ends = [1024 : i32, 128256 : i32], step = [1 : i32, 1 : i32]}> : (tensor<1024x128256xbf16>) -> tensor<1x128256xbf16>
  return %1 : tensor<1x128256xbf16>
}

// transpose_a=true row slice on a linear: A is [K, M], so the slice is pushed
// into A's trailing dim and the [N] bias still passes through untouched.
// CHECK-LABEL: func.func @linear_left_transpose
func.func @linear_left_transpose(%arg0: tensor<4096x1024xbf16>, %arg1: tensor<4096x128256xbf16>, %arg2: tensor<128256xbf16>) -> tensor<1x128256xbf16> {
  // CHECK: %[[A:.*]] = "ttir.slice_static"(%arg0) <{begins = [0 : i32, 1023 : i32], ends = [4096 : i32, 1024 : i32], step = [1 : i32, 1 : i32]}> : (tensor<4096x1024xbf16>) -> tensor<4096x1xbf16>
  // CHECK: "ttir.linear"(%[[A]], %arg1, %arg2) <{transpose_a = true, transpose_b = false}> : (tensor<4096x1xbf16>, tensor<4096x128256xbf16>, tensor<128256xbf16>) -> tensor<1x128256xbf16>
  // CHECK-NOT: tensor<1024x128256xbf16>
  %0 = "ttir.linear"(%arg0, %arg1, %arg2) <{transpose_a = true, transpose_b = false}> : (tensor<4096x1024xbf16>, tensor<4096x128256xbf16>, tensor<128256xbf16>) -> tensor<1024x128256xbf16>
  %1 = "ttir.slice_static"(%0) <{begins = [1023 : i32, 0 : i32], ends = [1024 : i32, 128256 : i32], step = [1 : i32, 1 : i32]}> : (tensor<1024x128256xbf16>) -> tensor<1x128256xbf16>
  return %1 : tensor<1x128256xbf16>
}

// A column (N) slice of a linear is pushed into B, and the [vocab] bias (which
// is indexed by N) is sliced along the same column range.
// CHECK-LABEL: func.func @linear_right
func.func @linear_right(%arg0: tensor<1024x4096xbf16>, %arg1: tensor<4096x128256xbf16>, %arg2: tensor<128256xbf16>) -> tensor<1024x64xbf16> {
  // CHECK: %[[B:.*]] = "ttir.slice_static"(%arg1) <{begins = [0 : i32, 0 : i32], ends = [4096 : i32, 64 : i32], step = [1 : i32, 1 : i32]}> : (tensor<4096x128256xbf16>) -> tensor<4096x64xbf16>
  // CHECK: %[[BIAS:.*]] = "ttir.slice_static"(%arg2) <{begins = [0 : i32], ends = [64 : i32], step = [1 : i32]}> : (tensor<128256xbf16>) -> tensor<64xbf16>
  // CHECK: "ttir.linear"(%arg0, %[[B]], %[[BIAS]]) <{transpose_a = false, transpose_b = false}> : (tensor<1024x4096xbf16>, tensor<4096x64xbf16>, tensor<64xbf16>) -> tensor<1024x64xbf16>
  // CHECK-NOT: tensor<1024x128256xbf16>
  %0 = "ttir.linear"(%arg0, %arg1, %arg2) : (tensor<1024x4096xbf16>, tensor<4096x128256xbf16>, tensor<128256xbf16>) -> tensor<1024x128256xbf16>
  %1 = "ttir.slice_static"(%0) <{begins = [0 : i32, 0 : i32], ends = [1024 : i32, 64 : i32], step = [1 : i32, 1 : i32]}> : (tensor<1024x128256xbf16>) -> tensor<1024x64xbf16>
  return %1 : tensor<1024x64xbf16>
}

// Column slice with a full 2D bias [M, N]: the bias dim aligned with N is real,
// so the bias is sliced along the same column range as B (its M dim is kept).
// CHECK-LABEL: func.func @linear_right_bias_2d
func.func @linear_right_bias_2d(%arg0: tensor<1024x4096xbf16>, %arg1: tensor<4096x128256xbf16>, %arg2: tensor<1024x128256xbf16>) -> tensor<1024x64xbf16> {
  // CHECK: %[[B:.*]] = "ttir.slice_static"(%arg1) <{begins = [0 : i32, 0 : i32], ends = [4096 : i32, 64 : i32], step = [1 : i32, 1 : i32]}> : (tensor<4096x128256xbf16>) -> tensor<4096x64xbf16>
  // CHECK: %[[BIAS:.*]] = "ttir.slice_static"(%arg2) <{begins = [0 : i32, 0 : i32], ends = [1024 : i32, 64 : i32], step = [1 : i32, 1 : i32]}> : (tensor<1024x128256xbf16>) -> tensor<1024x64xbf16>
  // CHECK: "ttir.linear"(%arg0, %[[B]], %[[BIAS]]) <{transpose_a = false, transpose_b = false}> : (tensor<1024x4096xbf16>, tensor<4096x64xbf16>, tensor<1024x64xbf16>) -> tensor<1024x64xbf16>
  %0 = "ttir.linear"(%arg0, %arg1, %arg2) : (tensor<1024x4096xbf16>, tensor<4096x128256xbf16>, tensor<1024x128256xbf16>) -> tensor<1024x128256xbf16>
  %1 = "ttir.slice_static"(%0) <{begins = [0 : i32, 0 : i32], ends = [1024 : i32, 64 : i32], step = [1 : i32, 1 : i32]}> : (tensor<1024x128256xbf16>) -> tensor<1024x64xbf16>
  return %1 : tensor<1024x64xbf16>
}

// Column slice with a bias [M, 1] (broadcast over N): the bias dim aligned with
// N is size-1, so it already covers every column and is left untouched.
// CHECK-LABEL: func.func @linear_right_bias_broadcast_n
func.func @linear_right_bias_broadcast_n(%arg0: tensor<1024x4096xbf16>, %arg1: tensor<4096x128256xbf16>, %arg2: tensor<1024x1xbf16>) -> tensor<1024x64xbf16> {
  // CHECK: %[[B:.*]] = "ttir.slice_static"(%arg1) <{begins = [0 : i32, 0 : i32], ends = [4096 : i32, 64 : i32], step = [1 : i32, 1 : i32]}> : (tensor<4096x128256xbf16>) -> tensor<4096x64xbf16>
  // CHECK: "ttir.linear"(%arg0, %[[B]], %arg2) <{transpose_a = false, transpose_b = false}> : (tensor<1024x4096xbf16>, tensor<4096x64xbf16>, tensor<1024x1xbf16>) -> tensor<1024x64xbf16>
  // CHECK-NOT: tensor<1024x128256xbf16>
  %0 = "ttir.linear"(%arg0, %arg1, %arg2) : (tensor<1024x4096xbf16>, tensor<4096x128256xbf16>, tensor<1024x1xbf16>) -> tensor<1024x128256xbf16>
  %1 = "ttir.slice_static"(%0) <{begins = [0 : i32, 0 : i32], ends = [1024 : i32, 64 : i32], step = [1 : i32, 1 : i32]}> : (tensor<1024x128256xbf16>) -> tensor<1024x64xbf16>
  return %1 : tensor<1024x64xbf16>
}

// Column slice with a scalar bias [1] (broadcast over everything): the aligned
// dim is size-1, so the bias is left untouched while B is narrowed.
// CHECK-LABEL: func.func @linear_right_bias_scalar
func.func @linear_right_bias_scalar(%arg0: tensor<1024x4096xbf16>, %arg1: tensor<4096x128256xbf16>, %arg2: tensor<1xbf16>) -> tensor<1024x64xbf16> {
  // CHECK: %[[B:.*]] = "ttir.slice_static"(%arg1) <{begins = [0 : i32, 0 : i32], ends = [4096 : i32, 64 : i32], step = [1 : i32, 1 : i32]}> : (tensor<4096x128256xbf16>) -> tensor<4096x64xbf16>
  // CHECK: "ttir.linear"(%arg0, %[[B]], %arg2) <{transpose_a = false, transpose_b = false}> : (tensor<1024x4096xbf16>, tensor<4096x64xbf16>, tensor<1xbf16>) -> tensor<1024x64xbf16>
  // CHECK-NOT: tensor<1024x128256xbf16>
  %0 = "ttir.linear"(%arg0, %arg1, %arg2) : (tensor<1024x4096xbf16>, tensor<4096x128256xbf16>, tensor<1xbf16>) -> tensor<1024x128256xbf16>
  %1 = "ttir.slice_static"(%0) <{begins = [0 : i32, 0 : i32], ends = [1024 : i32, 64 : i32], step = [1 : i32, 1 : i32]}> : (tensor<1024x128256xbf16>) -> tensor<1024x64xbf16>
  return %1 : tensor<1024x64xbf16>
}

// transpose_b=true column slice on a linear: B is [N, K], so the slice is
// pushed into B's rank-2 dim and the [N] bias is sliced along the same range.
// CHECK-LABEL: func.func @linear_right_transpose
func.func @linear_right_transpose(%arg0: tensor<1024x4096xbf16>, %arg1: tensor<128256x4096xbf16>, %arg2: tensor<128256xbf16>) -> tensor<1024x64xbf16> {
  // CHECK: %[[B:.*]] = "ttir.slice_static"(%arg1) <{begins = [0 : i32, 0 : i32], ends = [64 : i32, 4096 : i32], step = [1 : i32, 1 : i32]}> : (tensor<128256x4096xbf16>) -> tensor<64x4096xbf16>
  // CHECK: %[[BIAS:.*]] = "ttir.slice_static"(%arg2) <{begins = [0 : i32], ends = [64 : i32], step = [1 : i32]}> : (tensor<128256xbf16>) -> tensor<64xbf16>
  // CHECK: "ttir.linear"(%arg0, %[[B]], %[[BIAS]]) <{transpose_a = false, transpose_b = true}> : (tensor<1024x4096xbf16>, tensor<64x4096xbf16>, tensor<64xbf16>) -> tensor<1024x64xbf16>
  // CHECK-NOT: tensor<1024x128256xbf16>
  %0 = "ttir.linear"(%arg0, %arg1, %arg2) <{transpose_a = false, transpose_b = true}> : (tensor<1024x4096xbf16>, tensor<128256x4096xbf16>, tensor<128256xbf16>) -> tensor<1024x128256xbf16>
  %1 = "ttir.slice_static"(%0) <{begins = [0 : i32, 0 : i32], ends = [1024 : i32, 64 : i32], step = [1 : i32, 1 : i32]}> : (tensor<1024x128256xbf16>) -> tensor<1024x64xbf16>
  return %1 : tensor<1024x64xbf16>
}

// Negative: the slice narrows both the row and column dims of a linear output
// at once. As with matmul, only a single narrowed dim is supported, so the
// linear is left alone.
// CHECK-LABEL: func.func @linear_both_negative
func.func @linear_both_negative(%arg0: tensor<1024x4096xbf16>, %arg1: tensor<4096x128256xbf16>, %arg2: tensor<128256xbf16>) -> tensor<1x64xbf16> {
  // CHECK: "ttir.linear"(%arg0, %arg1, %arg2) <{transpose_a = false, transpose_b = false}> : (tensor<1024x4096xbf16>, tensor<4096x128256xbf16>, tensor<128256xbf16>) -> tensor<1024x128256xbf16>
  %0 = "ttir.linear"(%arg0, %arg1, %arg2) : (tensor<1024x4096xbf16>, tensor<4096x128256xbf16>, tensor<128256xbf16>) -> tensor<1024x128256xbf16>
  %1 = "ttir.slice_static"(%0) <{begins = [1023 : i32, 0 : i32], ends = [1024 : i32, 64 : i32], step = [1 : i32, 1 : i32]}> : (tensor<1024x128256xbf16>) -> tensor<1x64xbf16>
  return %1 : tensor<1x64xbf16>
}

// Batched 3D linear with a full 3D bias [B, M, N]. A column (N) slice is pushed
// into B and the bias is sliced along its trailing N dim (batch + M kept full).
// CHECK-LABEL: func.func @linear_batched_bias_3d
func.func @linear_batched_bias_3d(%arg0: tensor<2x1024x4096xbf16>, %arg1: tensor<2x4096x128256xbf16>, %arg2: tensor<2x1024x128256xbf16>) -> tensor<2x1024x64xbf16> {
  // CHECK: %[[B:.*]] = "ttir.slice_static"(%arg1) <{begins = [0 : i32, 0 : i32, 0 : i32], ends = [2 : i32, 4096 : i32, 64 : i32], step = [1 : i32, 1 : i32, 1 : i32]}> : (tensor<2x4096x128256xbf16>) -> tensor<2x4096x64xbf16>
  // CHECK: %[[BIAS:.*]] = "ttir.slice_static"(%arg2) <{begins = [0 : i32, 0 : i32, 0 : i32], ends = [2 : i32, 1024 : i32, 64 : i32], step = [1 : i32, 1 : i32, 1 : i32]}> : (tensor<2x1024x128256xbf16>) -> tensor<2x1024x64xbf16>
  // CHECK: "ttir.linear"(%arg0, %[[B]], %[[BIAS]]) <{transpose_a = false, transpose_b = false}> : (tensor<2x1024x4096xbf16>, tensor<2x4096x64xbf16>, tensor<2x1024x64xbf16>) -> tensor<2x1024x64xbf16>
  %0 = "ttir.linear"(%arg0, %arg1, %arg2) : (tensor<2x1024x4096xbf16>, tensor<2x4096x128256xbf16>, tensor<2x1024x128256xbf16>) -> tensor<2x1024x128256xbf16>
  %1 = "ttir.slice_static"(%0) <{begins = [0 : i32, 0 : i32, 0 : i32], ends = [2 : i32, 1024 : i32, 64 : i32], step = [1 : i32, 1 : i32, 1 : i32]}> : (tensor<2x1024x128256xbf16>) -> tensor<2x1024x64xbf16>
  return %1 : tensor<2x1024x64xbf16>
}

// Batched 3D linear with a full 3D bias, row (M) slice: pushed into A and the
// bias is sliced along its M dim (batch + N kept full).
// CHECK-LABEL: func.func @linear_batched_bias_3d_row
func.func @linear_batched_bias_3d_row(%arg0: tensor<2x1024x4096xbf16>, %arg1: tensor<2x4096x128256xbf16>, %arg2: tensor<2x1024x128256xbf16>) -> tensor<2x1x128256xbf16> {
  // CHECK: %[[A:.*]] = "ttir.slice_static"(%arg0) <{begins = [0 : i32, 1023 : i32, 0 : i32], ends = [2 : i32, 1024 : i32, 4096 : i32], step = [1 : i32, 1 : i32, 1 : i32]}> : (tensor<2x1024x4096xbf16>) -> tensor<2x1x4096xbf16>
  // CHECK: %[[BIAS:.*]] = "ttir.slice_static"(%arg2) <{begins = [0 : i32, 1023 : i32, 0 : i32], ends = [2 : i32, 1024 : i32, 128256 : i32], step = [1 : i32, 1 : i32, 1 : i32]}> : (tensor<2x1024x128256xbf16>) -> tensor<2x1x128256xbf16>
  // CHECK: "ttir.linear"(%[[A]], %arg1, %[[BIAS]]) <{transpose_a = false, transpose_b = false}> : (tensor<2x1x4096xbf16>, tensor<2x4096x128256xbf16>, tensor<2x1x128256xbf16>) -> tensor<2x1x128256xbf16>
  %0 = "ttir.linear"(%arg0, %arg1, %arg2) : (tensor<2x1024x4096xbf16>, tensor<2x4096x128256xbf16>, tensor<2x1024x128256xbf16>) -> tensor<2x1024x128256xbf16>
  %1 = "ttir.slice_static"(%0) <{begins = [0 : i32, 1023 : i32, 0 : i32], ends = [2 : i32, 1024 : i32, 128256 : i32], step = [1 : i32, 1 : i32, 1 : i32]}> : (tensor<2x1024x128256xbf16>) -> tensor<2x1x128256xbf16>
  return %1 : tensor<2x1x128256xbf16>
}

// Batched 3D linear with a 3D bias [1, M, N] (broadcast over the batch dim). A
// column slice narrows the bias's N dim; the size-1 batch dim stays full ([0:1])
// and keeps broadcasting onto the 2-batch output.
// CHECK-LABEL: func.func @linear_batched_bias_broadcast_batch
func.func @linear_batched_bias_broadcast_batch(%arg0: tensor<2x1024x4096xbf16>, %arg1: tensor<2x4096x128256xbf16>, %arg2: tensor<1x1024x128256xbf16>) -> tensor<2x1024x64xbf16> {
  // CHECK: %[[B:.*]] = "ttir.slice_static"(%arg1) <{begins = [0 : i32, 0 : i32, 0 : i32], ends = [2 : i32, 4096 : i32, 64 : i32], step = [1 : i32, 1 : i32, 1 : i32]}> : (tensor<2x4096x128256xbf16>) -> tensor<2x4096x64xbf16>
  // CHECK: %[[BIAS:.*]] = "ttir.slice_static"(%arg2) <{begins = [0 : i32, 0 : i32, 0 : i32], ends = [1 : i32, 1024 : i32, 64 : i32], step = [1 : i32, 1 : i32, 1 : i32]}> : (tensor<1x1024x128256xbf16>) -> tensor<1x1024x64xbf16>
  // CHECK: "ttir.linear"(%arg0, %[[B]], %[[BIAS]]) <{transpose_a = false, transpose_b = false}> : (tensor<2x1024x4096xbf16>, tensor<2x4096x64xbf16>, tensor<1x1024x64xbf16>) -> tensor<2x1024x64xbf16>
  %0 = "ttir.linear"(%arg0, %arg1, %arg2) : (tensor<2x1024x4096xbf16>, tensor<2x4096x128256xbf16>, tensor<1x1024x128256xbf16>) -> tensor<2x1024x128256xbf16>
  %1 = "ttir.slice_static"(%0) <{begins = [0 : i32, 0 : i32, 0 : i32], ends = [2 : i32, 1024 : i32, 64 : i32], step = [1 : i32, 1 : i32, 1 : i32]}> : (tensor<2x1024x128256xbf16>) -> tensor<2x1024x64xbf16>
  return %1 : tensor<2x1024x64xbf16>
}

// Negative: the linear result has another user, so narrowing would force the
// full linear to be recomputed. Leave it alone.
// CHECK-LABEL: func.func @linear_multiple_uses_negative
func.func @linear_multiple_uses_negative(%arg0: tensor<1024x4096xbf16>, %arg1: tensor<4096x128256xbf16>, %arg2: tensor<128256xbf16>) -> (tensor<1x128256xbf16>, tensor<1024x128256xbf16>) {
  // CHECK: "ttir.linear"(%arg0, %arg1, %arg2) <{transpose_a = false, transpose_b = false}> : (tensor<1024x4096xbf16>, tensor<4096x128256xbf16>, tensor<128256xbf16>) -> tensor<1024x128256xbf16>
  %0 = "ttir.linear"(%arg0, %arg1, %arg2) : (tensor<1024x4096xbf16>, tensor<4096x128256xbf16>, tensor<128256xbf16>) -> tensor<1024x128256xbf16>
  %1 = "ttir.slice_static"(%0) <{begins = [1023 : i32, 0 : i32], ends = [1024 : i32, 128256 : i32], step = [1 : i32, 1 : i32]}> : (tensor<1024x128256xbf16>) -> tensor<1x128256xbf16>
  return %1, %0 : tensor<1x128256xbf16>, tensor<1024x128256xbf16>
}

// Negative: the slice narrows the leading *batch* dim of a batched linear, not
// the output row/col dims, so the pattern leaves it alone.
// CHECK-LABEL: func.func @linear_batch_slice_negative
func.func @linear_batch_slice_negative(%arg0: tensor<2x1024x4096xbf16>, %arg1: tensor<2x4096x128256xbf16>, %arg2: tensor<128256xbf16>) -> tensor<1x1024x128256xbf16> {
  // CHECK: "ttir.linear"(%arg0, %arg1, %arg2) <{transpose_a = false, transpose_b = false}> : (tensor<2x1024x4096xbf16>, tensor<2x4096x128256xbf16>, tensor<128256xbf16>) -> tensor<2x1024x128256xbf16>
  %0 = "ttir.linear"(%arg0, %arg1, %arg2) : (tensor<2x1024x4096xbf16>, tensor<2x4096x128256xbf16>, tensor<128256xbf16>) -> tensor<2x1024x128256xbf16>
  %1 = "ttir.slice_static"(%0) <{begins = [0 : i32, 0 : i32, 0 : i32], ends = [1 : i32, 1024 : i32, 128256 : i32], step = [1 : i32, 1 : i32, 1 : i32]}> : (tensor<2x1024x128256xbf16>) -> tensor<1x1024x128256xbf16>
  return %1 : tensor<1x1024x128256xbf16>
}

// Interaction with SharedLHSMatmulFusion: three matmuls sharing the same LHS
// are fused into one matmul over the concatenated weights, then split back out
// with per-output *column* slices. Those slices feed off a matmul result with
// three uses, so this pattern's single-use guard must leave them alone --
// otherwise pushing the column slices into the (concatenated) RHS would undo
// the concatenation. The fused matmul must keep its full concatenated N dim
// (1152), with the three slices remaining on its output.
// CHECK-LABEL: func.func @shared_lhs_fusion_not_undone
func.func @shared_lhs_fusion_not_undone(%arg0: tensor<32x512xbf16>, %arg1: tensor<512x384xbf16>, %arg2: tensor<512x384xbf16>, %arg3: tensor<512x384xbf16>) -> (tensor<32x384xbf16>, tensor<32x384xbf16>, tensor<32x384xbf16>) {
  // CHECK: %[[W:.*]] = "ttir.concat"({{.*}}) <{dim = 1 : si32}> : (tensor<512x384xbf16>, tensor<512x384xbf16>, tensor<512x384xbf16>) -> tensor<512x1152xbf16>
  // CHECK: %[[M:.*]] = "ttir.matmul"(%arg0, %[[W]]) <{transpose_a = false, transpose_b = false}> : (tensor<32x512xbf16>, tensor<512x1152xbf16>) -> tensor<32x1152xbf16>
  // CHECK: "ttir.slice_static"(%[[M]])
  // CHECK: "ttir.slice_static"(%[[M]])
  // CHECK: "ttir.slice_static"(%[[M]])
  %0 = "ttir.matmul"(%arg0, %arg1) : (tensor<32x512xbf16>, tensor<512x384xbf16>) -> tensor<32x384xbf16>
  %1 = "ttir.matmul"(%arg0, %arg2) : (tensor<32x512xbf16>, tensor<512x384xbf16>) -> tensor<32x384xbf16>
  %2 = "ttir.matmul"(%arg0, %arg3) : (tensor<32x512xbf16>, tensor<512x384xbf16>) -> tensor<32x384xbf16>
  return %0, %1, %2 : tensor<32x384xbf16>, tensor<32x384xbf16>, tensor<32x384xbf16>
}

// Same shared-LHS group, but one of its outputs feeds a downstream "final"
// matmul whose result is row-sliced. Both fusions apply and compose: the
// shared-LHS group still fuses to one matmul over the concatenated weights
// (full N=1152, three output slices off it -- left untouched by this pattern's
// single-use guard), while the trailing row slice IS pushed up into the final
// matmul's LHS. That LHS is itself a (column) slice of the fused result, so the
// pushed row slice folds together with it into a single combined slice of the
// fused output ([31:32, 0:384]), and the final matmul is narrowed to M=1.
// CHECK-LABEL: func.func @shared_lhs_fusion_with_final_slice
func.func @shared_lhs_fusion_with_final_slice(%arg0: tensor<32x512xbf16>, %arg1: tensor<512x384xbf16>, %arg2: tensor<512x384xbf16>, %arg3: tensor<512x384xbf16>, %arg4: tensor<384x256xbf16>) -> (tensor<32x384xbf16>, tensor<32x384xbf16>, tensor<1x256xbf16>) {
  // The shared-LHS group is fused and preserved (full concatenated N=1152).
  // CHECK: %[[W:.*]] = "ttir.concat"({{.*}}) <{dim = 1 : si32}> : (tensor<512x384xbf16>, tensor<512x384xbf16>, tensor<512x384xbf16>) -> tensor<512x1152xbf16>
  // CHECK: %[[FUSED:.*]] = "ttir.matmul"(%arg0, %[[W]]) <{transpose_a = false, transpose_b = false}> : (tensor<32x512xbf16>, tensor<512x1152xbf16>) -> tensor<32x1152xbf16>
  // The two directly-returned outputs stay as full-row column slices of the fused result.
  // CHECK: "ttir.slice_static"(%[[FUSED]]) <{begins = [0 : i32, 384 : i32], ends = [32 : i32, 768 : i32], step = [1 : i32, 1 : i32]}>
  // CHECK: "ttir.slice_static"(%[[FUSED]]) <{begins = [0 : i32, 768 : i32], ends = [32 : i32, 1152 : i32], step = [1 : i32, 1 : i32]}>
  // The final matmul's LHS slice and the trailing row slice fold into one slice
  // of the fused output, and the final matmul is narrowed to a single row.
  // CHECK: %[[L:.*]] = "ttir.slice_static"(%[[FUSED]]) <{begins = [31 : i32, 0 : i32], ends = [32 : i32, 384 : i32], step = [1 : i32, 1 : i32]}> : (tensor<32x1152xbf16>) -> tensor<1x384xbf16>
  // CHECK: "ttir.matmul"(%[[L]], %arg4) <{transpose_a = false, transpose_b = false}> : (tensor<1x384xbf16>, tensor<384x256xbf16>) -> tensor<1x256xbf16>
  %0 = "ttir.matmul"(%arg0, %arg1) : (tensor<32x512xbf16>, tensor<512x384xbf16>) -> tensor<32x384xbf16>
  %1 = "ttir.matmul"(%arg0, %arg2) : (tensor<32x512xbf16>, tensor<512x384xbf16>) -> tensor<32x384xbf16>
  %2 = "ttir.matmul"(%arg0, %arg3) : (tensor<32x512xbf16>, tensor<512x384xbf16>) -> tensor<32x384xbf16>
  %3 = "ttir.matmul"(%2, %arg4) : (tensor<32x384xbf16>, tensor<384x256xbf16>) -> tensor<32x256xbf16>
  %s = "ttir.slice_static"(%3) <{begins = [31 : i32, 0 : i32], ends = [32 : i32, 256 : i32], step = [1 : i32, 1 : i32]}> : (tensor<32x256xbf16>) -> tensor<1x256xbf16>
  return %0, %1, %s : tensor<32x384xbf16>, tensor<32x384xbf16>, tensor<1x256xbf16>
}

// Collision: all three matmuls share the same LHS (%arg0) *and* the last one's
// output is sliced -- so both SharedLHSMatmulFusion (wants to fuse the three)
// and PermuteSliceAfterMatmul (wants to push the slice into %arg0) match. With
// the pass's top-down traversal the matmuls are visited before the slice, so
// SharedLHSMatmulFusion fires first and fuses all three into one matmul over
// the concatenated weights. The fused result then has three uses, so
// PermuteSliceAfterMatmul's single-use guard blocks it: the row slice composes
// with the fused output's column slice ([31:32, 0:384]) rather than being
// pushed into %arg0. SharedLHS wins, and only one matmul remains.
// CHECK-LABEL: func.func @shared_lhs_wins_collision
func.func @shared_lhs_wins_collision(%arg0: tensor<32x512xbf16>, %arg1: tensor<512x384xbf16>, %arg2: tensor<512x384xbf16>, %arg3: tensor<512x384xbf16>) -> (tensor<32x384xbf16>, tensor<32x384xbf16>, tensor<1x384xbf16>) {
  // CHECK: %[[W:.*]] = "ttir.concat"({{.*}}) <{dim = 1 : si32}> : (tensor<512x384xbf16>, tensor<512x384xbf16>, tensor<512x384xbf16>) -> tensor<512x1152xbf16>
  // CHECK: %[[FUSED:.*]] = "ttir.matmul"(%arg0, %[[W]]) <{transpose_a = false, transpose_b = false}> : (tensor<32x512xbf16>, tensor<512x1152xbf16>) -> tensor<32x1152xbf16>
  // CHECK: "ttir.slice_static"(%[[FUSED]]) <{begins = [0 : i32, 384 : i32], ends = [32 : i32, 768 : i32], step = [1 : i32, 1 : i32]}>
  // CHECK: "ttir.slice_static"(%[[FUSED]]) <{begins = [0 : i32, 768 : i32], ends = [32 : i32, 1152 : i32], step = [1 : i32, 1 : i32]}>
  // The sliced output composes the row slice with the fused column slice.
  // CHECK: "ttir.slice_static"(%[[FUSED]]) <{begins = [31 : i32, 0 : i32], ends = [32 : i32, 384 : i32], step = [1 : i32, 1 : i32]}> : (tensor<32x1152xbf16>) -> tensor<1x384xbf16>
  // PermuteSliceAfterMatmul did not fire: no second, narrowed matmul split off.
  // CHECK-NOT: "ttir.matmul"
  %0 = "ttir.matmul"(%arg0, %arg1) : (tensor<32x512xbf16>, tensor<512x384xbf16>) -> tensor<32x384xbf16>
  %1 = "ttir.matmul"(%arg0, %arg2) : (tensor<32x512xbf16>, tensor<512x384xbf16>) -> tensor<32x384xbf16>
  %2 = "ttir.matmul"(%arg0, %arg3) : (tensor<32x512xbf16>, tensor<512x384xbf16>) -> tensor<32x384xbf16>
  %s = "ttir.slice_static"(%2) <{begins = [31 : i32, 0 : i32], ends = [32 : i32, 384 : i32], step = [1 : i32, 1 : i32]}> : (tensor<32x384xbf16>) -> tensor<1x384xbf16>
  return %0, %1, %s : tensor<32x384xbf16>, tensor<32x384xbf16>, tensor<1x384xbf16>
}

// Cascade through a chain of matmuls: each matmul result is the (single-use)
// LHS of the next, so a row slice at the bottom is pushed up one level into the
// LHS, which is itself a matmul -- re-matching this pattern. The greedy driver
// iterates it to a fixpoint, so the slice walks all the way up to the original
// %arg0 and every matmul in the chain is narrowed to a single row (M=1). No
// explicit loop in the pattern is needed.
// CHECK-LABEL: func.func @cascade_three_matmuls
func.func @cascade_three_matmuls(%arg0: tensor<1024x4096xbf16>, %arg1: tensor<4096x2048xbf16>, %arg2: tensor<2048x1024xbf16>, %arg3: tensor<1024x512xbf16>) -> tensor<1x512xbf16> {
  // The slice ends up on %arg0 (the top of the chain), narrowing it to one row.
  // CHECK: %[[A:.*]] = "ttir.slice_static"(%arg0) <{begins = [1023 : i32, 0 : i32], ends = [1024 : i32, 4096 : i32], step = [1 : i32, 1 : i32]}> : (tensor<1024x4096xbf16>) -> tensor<1x4096xbf16>
  // All three matmuls are narrowed to M=1 and no slice remains on any output.
  // CHECK: %[[M1:.*]] = "ttir.matmul"(%[[A]], %arg1) <{transpose_a = false, transpose_b = false}> : (tensor<1x4096xbf16>, tensor<4096x2048xbf16>) -> tensor<1x2048xbf16>
  // CHECK: %[[M2:.*]] = "ttir.matmul"(%[[M1]], %arg2) <{transpose_a = false, transpose_b = false}> : (tensor<1x2048xbf16>, tensor<2048x1024xbf16>) -> tensor<1x1024xbf16>
  // CHECK: "ttir.matmul"(%[[M2]], %arg3) <{transpose_a = false, transpose_b = false}> : (tensor<1x1024xbf16>, tensor<1024x512xbf16>) -> tensor<1x512xbf16>
  // CHECK-NOT: "ttir.slice_static"
  %m1 = "ttir.matmul"(%arg0, %arg1) : (tensor<1024x4096xbf16>, tensor<4096x2048xbf16>) -> tensor<1024x2048xbf16>
  %m2 = "ttir.matmul"(%m1, %arg2) : (tensor<1024x2048xbf16>, tensor<2048x1024xbf16>) -> tensor<1024x1024xbf16>
  %m3 = "ttir.matmul"(%m2, %arg3) : (tensor<1024x1024xbf16>, tensor<1024x512xbf16>) -> tensor<1024x512xbf16>
  %s = "ttir.slice_static"(%m3) <{begins = [1023 : i32, 0 : i32], ends = [1024 : i32, 512 : i32], step = [1 : i32, 1 : i32]}> : (tensor<1024x512xbf16>) -> tensor<1x512xbf16>
  return %s : tensor<1x512xbf16>
}
