// RUN: ttmlir-opt --ttir-fusing %s -o %t.mlir
// RUN: FileCheck %s --input-file=%t.mlir

// An output-row slice of a matmul is pushed up into the LHS operand, so the
// matmul only computes the rows that are actually used.

// Direct slice of the last row of a 2D matmul output (greedy-decode lm_head
// after the rank-3 logits tensor has been collapsed to 2D).
// CHECK-LABEL: func.func @last_row_direct
func.func @last_row_direct(%arg0: tensor<1024x4096xbf16>, %arg1: tensor<4096x128256xbf16>) -> tensor<1x128256xbf16> {
  // The LHS is sliced to its last row first, then the matmul is narrowed.
  // CHECK: %[[A:.*]] = "ttir.slice_static"(%arg0) <{begins = [1023 : i32, 0 : i32], ends = [1024 : i32, 4096 : i32], step = [1 : i32, 1 : i32]}> : (tensor<1024x4096xbf16>) -> tensor<1x4096xbf16>
  // CHECK: "ttir.matmul"(%[[A]], %arg1){{.*}} : (tensor<1x4096xbf16>, tensor<4096x128256xbf16>) -> tensor<1x128256xbf16>
  // CHECK-NOT: tensor<1024x128256xbf16>
  %0 = "ttir.matmul"(%arg0, %arg1) : (tensor<1024x4096xbf16>, tensor<4096x128256xbf16>) -> tensor<1024x128256xbf16>
  %1 = "ttir.slice_static"(%0) <{begins = [1023 : i32, 0 : i32], ends = [1024 : i32, 128256 : i32], step = [1 : i32, 1 : i32]}> : (tensor<1024x128256xbf16>) -> tensor<1x128256xbf16>
  return %1 : tensor<1x128256xbf16>
}

// Same, but with a leading-unit-dim reshape between matmul and slice (the
// rank-3 logits tensor [1, seq, vocab] produced by HF causal-LM heads).
// CHECK-LABEL: func.func @last_row_through_reshape
func.func @last_row_through_reshape(%arg0: tensor<1024x4096xbf16>, %arg1: tensor<4096x128256xbf16>) -> tensor<1x1x128256xbf16> {
  // CHECK: %[[A:.*]] = "ttir.slice_static"(%arg0) <{begins = [1023 : i32, 0 : i32], ends = [1024 : i32, 4096 : i32], step = [1 : i32, 1 : i32]}> : (tensor<1024x4096xbf16>) -> tensor<1x4096xbf16>
  // CHECK: %[[M:.*]] = "ttir.matmul"(%[[A]], %arg1){{.*}} : (tensor<1x4096xbf16>, tensor<4096x128256xbf16>) -> tensor<1x128256xbf16>
  // CHECK: "ttir.reshape"(%[[M]]) <{shape = [1 : i32, 1 : i32, 128256 : i32]}>
  // CHECK-NOT: tensor<1024x128256xbf16>
  %0 = "ttir.matmul"(%arg0, %arg1) : (tensor<1024x4096xbf16>, tensor<4096x128256xbf16>) -> tensor<1024x128256xbf16>
  %1 = "ttir.reshape"(%0) <{shape = [1 : i32, 1024 : i32, 128256 : i32]}> : (tensor<1024x128256xbf16>) -> tensor<1x1024x128256xbf16>
  %2 = "ttir.slice_static"(%1) <{begins = [0 : i32, 1023 : i32, 0 : i32], ends = [1 : i32, 1024 : i32, 128256 : i32], step = [1 : i32, 1 : i32, 1 : i32]}> : (tensor<1x1024x128256xbf16>) -> tensor<1x1x128256xbf16>
  return %2 : tensor<1x1x128256xbf16>
}

// Negative: the matmul result has another user, so narrowing would force the
// full matmul to be recomputed. Leave it alone.
// CHECK-LABEL: func.func @multiple_uses
func.func @multiple_uses(%arg0: tensor<1024x4096xbf16>, %arg1: tensor<4096x128256xbf16>) -> (tensor<1x128256xbf16>, tensor<1024x128256xbf16>) {
  // CHECK: "ttir.matmul"(%arg0, %arg1){{.*}} : (tensor<1024x4096xbf16>, tensor<4096x128256xbf16>) -> tensor<1024x128256xbf16>
  %0 = "ttir.matmul"(%arg0, %arg1) : (tensor<1024x4096xbf16>, tensor<4096x128256xbf16>) -> tensor<1024x128256xbf16>
  %1 = "ttir.slice_static"(%0) <{begins = [1023 : i32, 0 : i32], ends = [1024 : i32, 128256 : i32], step = [1 : i32, 1 : i32]}> : (tensor<1024x128256xbf16>) -> tensor<1x128256xbf16>
  return %1, %0 : tensor<1x128256xbf16>, tensor<1024x128256xbf16>
}

// Negative: the slice narrows the column (N) dim, not the row dim. This would
// be an RHS narrowing, which this pattern does not handle.
// CHECK-LABEL: func.func @col_slice_untouched
func.func @col_slice_untouched(%arg0: tensor<1024x4096xbf16>, %arg1: tensor<4096x128256xbf16>) -> tensor<1024x64xbf16> {
  // CHECK: "ttir.matmul"(%arg0, %arg1){{.*}} : (tensor<1024x4096xbf16>, tensor<4096x128256xbf16>) -> tensor<1024x128256xbf16>
  %0 = "ttir.matmul"(%arg0, %arg1) : (tensor<1024x4096xbf16>, tensor<4096x128256xbf16>) -> tensor<1024x128256xbf16>
  %1 = "ttir.slice_static"(%0) <{begins = [0 : i32, 0 : i32], ends = [1024 : i32, 64 : i32], step = [1 : i32, 1 : i32]}> : (tensor<1024x128256xbf16>) -> tensor<1024x64xbf16>
  return %1 : tensor<1024x64xbf16>
}

// A contiguous block of rows in the middle of the output ([5:17]) is pushed up
// just like a single trailing row: only the selected rows feed the matmul.
// CHECK-LABEL: func.func @middle_rows_direct
func.func @middle_rows_direct(%arg0: tensor<1024x4096xbf16>, %arg1: tensor<4096x128256xbf16>) -> tensor<12x128256xbf16> {
  // CHECK: %[[A:.*]] = "ttir.slice_static"(%arg0) <{begins = [5 : i32, 0 : i32], ends = [17 : i32, 4096 : i32], step = [1 : i32, 1 : i32]}> : (tensor<1024x4096xbf16>) -> tensor<12x4096xbf16>
  // CHECK: "ttir.matmul"(%[[A]], %arg1){{.*}} : (tensor<12x4096xbf16>, tensor<4096x128256xbf16>) -> tensor<12x128256xbf16>
  // CHECK-NOT: tensor<1024x128256xbf16>
  %0 = "ttir.matmul"(%arg0, %arg1) : (tensor<1024x4096xbf16>, tensor<4096x128256xbf16>) -> tensor<1024x128256xbf16>
  %1 = "ttir.slice_static"(%0) <{begins = [5 : i32, 0 : i32], ends = [17 : i32, 128256 : i32], step = [1 : i32, 1 : i32]}> : (tensor<1024x128256xbf16>) -> tensor<12x128256xbf16>
  return %1 : tensor<12x128256xbf16>
}

// transpose_a=true: A is [K, M], so the output rows come from A's *last* dim.
// The row slice is pushed into that trailing dim, and transpose_a is preserved.
// CHECK-LABEL: func.func @transpose_a_last_row
func.func @transpose_a_last_row(%arg0: tensor<4096x1024xbf16>, %arg1: tensor<4096x128256xbf16>) -> tensor<1x128256xbf16> {
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
// CHECK-LABEL: func.func @real_reshape_untouched
func.func @real_reshape_untouched(%arg0: tensor<1024x4096xbf16>, %arg1: tensor<4096x128256xbf16>) -> tensor<1x1x64128xbf16> {
  // CHECK: "ttir.matmul"(%arg0, %arg1){{.*}} : (tensor<1024x4096xbf16>, tensor<4096x128256xbf16>) -> tensor<1024x128256xbf16>
  %0 = "ttir.matmul"(%arg0, %arg1) : (tensor<1024x4096xbf16>, tensor<4096x128256xbf16>) -> tensor<1024x128256xbf16>
  %1 = "ttir.reshape"(%0) <{shape = [1 : i32, 2048 : i32, 64128 : i32]}> : (tensor<1024x128256xbf16>) -> tensor<1x2048x64128xbf16>
  %2 = "ttir.slice_static"(%1) <{begins = [0 : i32, 2047 : i32, 0 : i32], ends = [1 : i32, 2048 : i32, 64128 : i32], step = [1 : i32, 1 : i32, 1 : i32]}> : (tensor<1x2048x64128xbf16>) -> tensor<1x1x64128xbf16>
  return %2 : tensor<1x1x64128xbf16>
}

// Negative: the reshape inflates the row dim into a leading non-unit dim
// ([1024, 128256] -> [2, 512, 128256]). The leading pad dim is 2, not 1, so the
// slice dims cannot be mapped back to the matmul output and fusion is skipped.
// CHECK-LABEL: func.func @reshape_inflate_leading_untouched
func.func @reshape_inflate_leading_untouched(%arg0: tensor<1024x4096xbf16>, %arg1: tensor<4096x128256xbf16>) -> tensor<1x1x128256xbf16> {
  // CHECK: "ttir.matmul"(%arg0, %arg1){{.*}} : (tensor<1024x4096xbf16>, tensor<4096x128256xbf16>) -> tensor<1024x128256xbf16>
  %0 = "ttir.matmul"(%arg0, %arg1) : (tensor<1024x4096xbf16>, tensor<4096x128256xbf16>) -> tensor<1024x128256xbf16>
  %1 = "ttir.reshape"(%0) <{shape = [2 : i32, 512 : i32, 128256 : i32]}> : (tensor<1024x128256xbf16>) -> tensor<2x512x128256xbf16>
  %2 = "ttir.slice_static"(%1) <{begins = [0 : i32, 511 : i32, 0 : i32], ends = [1 : i32, 512 : i32, 128256 : i32], step = [1 : i32, 1 : i32, 1 : i32]}> : (tensor<2x512x128256xbf16>) -> tensor<1x1x128256xbf16>
  return %2 : tensor<1x1x128256xbf16>
}

// Negative: the reshape inflates a non-unit dim in the middle of the leading
// pad region ([1024, 128256] -> [1, 2, 512, 128256]). A pad dim (the 2) is not
// 1, so the slice dims do not line up with the matmul output and fusion is
// skipped.
// CHECK-LABEL: func.func @reshape_inflate_middle_untouched
func.func @reshape_inflate_middle_untouched(%arg0: tensor<1024x4096xbf16>, %arg1: tensor<4096x128256xbf16>) -> tensor<1x1x1x128256xbf16> {
  // CHECK: "ttir.matmul"(%arg0, %arg1){{.*}} : (tensor<1024x4096xbf16>, tensor<4096x128256xbf16>) -> tensor<1024x128256xbf16>
  %0 = "ttir.matmul"(%arg0, %arg1) : (tensor<1024x4096xbf16>, tensor<4096x128256xbf16>) -> tensor<1024x128256xbf16>
  %1 = "ttir.reshape"(%0) <{shape = [1 : i32, 2 : i32, 512 : i32, 128256 : i32]}> : (tensor<1024x128256xbf16>) -> tensor<1x2x512x128256xbf16>
  %2 = "ttir.slice_static"(%1) <{begins = [0 : i32, 0 : i32, 511 : i32, 0 : i32], ends = [1 : i32, 1 : i32, 512 : i32, 128256 : i32], step = [1 : i32, 1 : i32, 1 : i32, 1 : i32]}> : (tensor<1x2x512x128256xbf16>) -> tensor<1x1x1x128256xbf16>
  return %2 : tensor<1x1x1x128256xbf16>
}

// Negative: a leading-unit dim is *deflated* away by the reshape
// ([1, 1024, 128256] -> [1024, 128256]) so the slice has lower rank than the
// matmul output. The pattern only looks through a reshape that prepends size-1
// dims, not one that removes them, so this is left untouched.
// CHECK-LABEL: func.func @reshape_deflate_untouched
func.func @reshape_deflate_untouched(%arg0: tensor<1x1024x4096xbf16>, %arg1: tensor<1x4096x128256xbf16>) -> tensor<1x128256xbf16> {
  // CHECK: "ttir.matmul"(%arg0, %arg1){{.*}} : (tensor<1x1024x4096xbf16>, tensor<1x4096x128256xbf16>) -> tensor<1x1024x128256xbf16>
  %0 = "ttir.matmul"(%arg0, %arg1) : (tensor<1x1024x4096xbf16>, tensor<1x4096x128256xbf16>) -> tensor<1x1024x128256xbf16>
  %1 = "ttir.reshape"(%0) <{shape = [1024 : i32, 128256 : i32]}> : (tensor<1x1024x128256xbf16>) -> tensor<1024x128256xbf16>
  %2 = "ttir.slice_static"(%1) <{begins = [1023 : i32, 0 : i32], ends = [1024 : i32, 128256 : i32], step = [1 : i32, 1 : i32]}> : (tensor<1024x128256xbf16>) -> tensor<1x128256xbf16>
  return %2 : tensor<1x128256xbf16>
}
