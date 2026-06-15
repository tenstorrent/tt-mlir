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
  // CHECK: "ttir.matmul"(%[[A]], %arg1) : (tensor<1x4096xbf16>, tensor<4096x128256xbf16>) -> tensor<1x128256xbf16>
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
  // CHECK: %[[M:.*]] = "ttir.matmul"(%[[A]], %arg1) : (tensor<1x4096xbf16>, tensor<4096x128256xbf16>) -> tensor<1x128256xbf16>
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
  // CHECK: "ttir.matmul"(%arg0, %arg1) : (tensor<1024x4096xbf16>, tensor<4096x128256xbf16>) -> tensor<1024x128256xbf16>
  %0 = "ttir.matmul"(%arg0, %arg1) : (tensor<1024x4096xbf16>, tensor<4096x128256xbf16>) -> tensor<1024x128256xbf16>
  %1 = "ttir.slice_static"(%0) <{begins = [1023 : i32, 0 : i32], ends = [1024 : i32, 128256 : i32], step = [1 : i32, 1 : i32]}> : (tensor<1024x128256xbf16>) -> tensor<1x128256xbf16>
  return %1, %0 : tensor<1x128256xbf16>, tensor<1024x128256xbf16>
}

// Negative: the slice narrows the column (N) dim, not the row dim. This would
// be an RHS narrowing, which this pattern does not handle.
// CHECK-LABEL: func.func @col_slice_untouched
func.func @col_slice_untouched(%arg0: tensor<1024x4096xbf16>, %arg1: tensor<4096x128256xbf16>) -> tensor<1024x64xbf16> {
  // CHECK: "ttir.matmul"(%arg0, %arg1) : (tensor<1024x4096xbf16>, tensor<4096x128256xbf16>) -> tensor<1024x128256xbf16>
  %0 = "ttir.matmul"(%arg0, %arg1) : (tensor<1024x4096xbf16>, tensor<4096x128256xbf16>) -> tensor<1024x128256xbf16>
  %1 = "ttir.slice_static"(%0) <{begins = [0 : i32, 0 : i32], ends = [1024 : i32, 64 : i32], step = [1 : i32, 1 : i32]}> : (tensor<1024x128256xbf16>) -> tensor<1024x64xbf16>
  return %1 : tensor<1024x64xbf16>
}
