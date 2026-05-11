// RUN: ttmlir-opt --ttir-rank-normalization %s | FileCheck %s
// RUN: ttmlir-opt --ttir-rank-normalization --canonicalize %s | FileCheck %s --check-prefix=CHECK-CANON
//
// The ttir-rank-normalization pass promotes tensor types with rank < 2 to rank 2
// by prepending 1s to the shape (e.g. tensor<128xf32> -> tensor<1x128xf32>).
// After promotion, some reshapes become identity (same type in and out).
// The canonicalizer folds these away.
//
// Public functions form the module's external boundary (JAX entry points,
// PJRT-visible signatures). Their signatures are preserved; the pass instead
// inserts `ttir.reshape` ops at function entry/exit so the body operates on
// rank>=2 while the function arguments/results stay rank<2. The internal
// (private) tests below exercise the full signature-promotion path.

// =============================================================================
// Test 1: Single op with 1D input/output - types promoted in place, no new ops
// =============================================================================

// CHECK-LABEL: func.func private @single_op_1d
// CHECK-SAME: (%arg0: tensor<1x128xf32>) -> tensor<1x128xf32>
// CHECK-NOT: ttir.reshape
// CHECK: %[[ABS:.*]] = "ttir.abs"(%arg0) : (tensor<1x128xf32>) -> tensor<1x128xf32>
// CHECK: return %[[ABS]] : tensor<1x128xf32>
func.func private @single_op_1d(%arg0: tensor<128xf32>) -> tensor<128xf32> {
  %0 = "ttir.abs"(%arg0) : (tensor<128xf32>) -> tensor<128xf32>
  return %0 : tensor<128xf32>
}

// =============================================================================
// Test 2: Chained ops - all 1D types promoted to 2D throughout
// =============================================================================

// CHECK-LABEL: func.func private @chained_ops_1d
// CHECK-SAME: (%arg0: tensor<1x128xf32>) -> tensor<1x128xf32>
// CHECK-NOT: ttir.reshape
// CHECK: %[[ABS:.*]] = "ttir.abs"(%arg0) : (tensor<1x128xf32>) -> tensor<1x128xf32>
// CHECK: %[[NEG:.*]] = "ttir.neg"(%[[ABS]]) : (tensor<1x128xf32>) -> tensor<1x128xf32>
// CHECK: return %[[NEG]] : tensor<1x128xf32>
func.func private @chained_ops_1d(%arg0: tensor<128xf32>) -> tensor<128xf32> {
  %0 = "ttir.abs"(%arg0) : (tensor<128xf32>) -> tensor<128xf32>
  %1 = "ttir.neg"(%0) : (tensor<128xf32>) -> tensor<128xf32>
  return %1 : tensor<128xf32>
}

// =============================================================================
// Test 3: Function with no 1D tensors - unchanged (rank already >= 2)
// =============================================================================

// CHECK-LABEL: func.func private @no_1d_tensors
// CHECK-SAME: (%arg0: tensor<32x64xf32>) -> tensor<32x64xf32>
// CHECK-NOT: ttir.reshape
// CHECK: %[[ABS:.*]] = "ttir.abs"(%arg0) : (tensor<32x64xf32>) -> tensor<32x64xf32>
// CHECK: return %[[ABS]] : tensor<32x64xf32>
func.func private @no_1d_tensors(%arg0: tensor<32x64xf32>) -> tensor<32x64xf32> {
  %0 = "ttir.abs"(%arg0) : (tensor<32x64xf32>) -> tensor<32x64xf32>
  return %0 : tensor<32x64xf32>
}

// =============================================================================
// Test 4: Multiple 1D inputs - both args and results promoted
// =============================================================================

// CHECK-LABEL: func.func private @multiple_1d_inputs
// CHECK-SAME: (%arg0: tensor<1x128xf32>, %arg1: tensor<1x128xf32>) -> tensor<1x128xf32>
// CHECK-NOT: ttir.reshape
// CHECK: %[[ADD:.*]] = "ttir.add"(%arg0, %arg1) : (tensor<1x128xf32>, tensor<1x128xf32>) -> tensor<1x128xf32>
// CHECK: return %[[ADD]] : tensor<1x128xf32>
func.func private @multiple_1d_inputs(%arg0: tensor<128xf32>, %arg1: tensor<128xf32>) -> tensor<128xf32> {
  %0 = "ttir.add"(%arg0, %arg1) : (tensor<128xf32>, tensor<128xf32>) -> tensor<128xf32>
  return %0 : tensor<128xf32>
}

// =============================================================================
// Test 5: Mixed 1D and 2D - only 1D arg/result promoted
// =============================================================================

// CHECK-LABEL: func.func private @mixed_1d_and_2d
// CHECK-SAME: (%arg0: tensor<1x128xf32>, %arg1: tensor<32x64xf32>) -> tensor<1x128xf32>
// CHECK: %[[ABS:.*]] = "ttir.abs"(%arg0) : (tensor<1x128xf32>) -> tensor<1x128xf32>
// CHECK: return %[[ABS]] : tensor<1x128xf32>
func.func private @mixed_1d_and_2d(%arg0: tensor<128xf32>, %arg1: tensor<32x64xf32>) -> tensor<128xf32> {
  %0 = "ttir.abs"(%arg0) : (tensor<128xf32>) -> tensor<128xf32>
  return %0 : tensor<128xf32>
}

// =============================================================================
// Test 6: Reshape 1D->2D - types promoted, shape attr already matches.
// =============================================================================

// CHECK-LABEL: func.func private @input_already_reshaped
// CHECK-SAME: (%arg0: tensor<1x128xf32>) -> tensor<1x128xf32>
// CHECK: %[[R:.*]] = "ttir.reshape"(%arg0) <{shape = [1 : i32, 128 : i32]}> : (tensor<1x128xf32>) -> tensor<1x128xf32>
// CHECK: %[[ABS:.*]] = "ttir.abs"(%[[R]]) : (tensor<1x128xf32>) -> tensor<1x128xf32>
// CHECK: return %[[ABS]] : tensor<1x128xf32>
func.func private @input_already_reshaped(%arg0: tensor<128xf32>) -> tensor<1x128xf32> {
  %0 = "ttir.reshape"(%arg0) <{shape = [1 : i32, 128 : i32]}> : (tensor<128xf32>) -> tensor<1x128xf32>
  %1 = "ttir.abs"(%0) : (tensor<1x128xf32>) -> tensor<1x128xf32>
  return %1 : tensor<1x128xf32>
}

// =============================================================================
// Test 7: Reshape 2D->1D - result type promoted, shape attr promoted.
// =============================================================================

// CHECK-LABEL: func.func private @output_1d_only
// CHECK-SAME: (%arg0: tensor<1x128xf32>) -> tensor<1x128xf32>
// CHECK: %[[ABS:.*]] = "ttir.abs"(%arg0) : (tensor<1x128xf32>) -> tensor<1x128xf32>
// CHECK: %[[R:.*]] = "ttir.reshape"(%[[ABS]]) <{shape = [1 : i32, 128 : i32]}> : (tensor<1x128xf32>) -> tensor<1x128xf32>
// CHECK: return %[[R]] : tensor<1x128xf32>
func.func private @output_1d_only(%arg0: tensor<1x128xf32>) -> tensor<128xf32> {
  %0 = "ttir.abs"(%arg0) : (tensor<1x128xf32>) -> tensor<1x128xf32>
  %1 = "ttir.reshape"(%0) <{shape = [128 : i32]}> : (tensor<1x128xf32>) -> tensor<128xf32>
  return %1 : tensor<128xf32>
}

// =============================================================================
// Tests 8-9: Reshapes with types that need promotion.
// After promotion these become identity reshapes (same type in and out).
// =============================================================================

// CHECK-LABEL: func.func private @reshape_shape_attr_promoted
// CHECK-SAME: (%arg0: tensor<1x64xf32>) -> tensor<1x64xf32>
// CHECK-NOT: ttir.reshape
// CHECK: return %arg0 : tensor<1x64xf32>
func.func private @reshape_shape_attr_promoted(%arg0: tensor<64xf32>) -> tensor<64xf32> {
  %0 = "ttir.reshape"(%arg0) <{shape = [64 : i32]}> : (tensor<64xf32>) -> tensor<64xf32>
  return %0 : tensor<64xf32>
}

// CHECK-LABEL: func.func private @reshape_1d_to_1d
// CHECK-SAME: (%arg0: tensor<1x64xf32>) -> tensor<1x64xf32>
// CHECK-NOT: ttir.reshape
// CHECK: return %arg0 : tensor<1x64xf32>
func.func private @reshape_1d_to_1d(%arg0: tensor<64xf32>) -> tensor<64xf32> {
  %0 = "ttir.reshape"(%arg0) <{shape = [64 : i32]}> : (tensor<64xf32>) -> tensor<64xf32>
  return %0 : tensor<64xf32>
}

// CHECK-LABEL: func.func private @reshape_1d_to_2d
// CHECK-SAME: (%arg0: tensor<1x64xf32>) -> tensor<1x64xf32>
// CHECK: %[[R:.*]] = "ttir.reshape"(%arg0) <{shape = [1 : i32, 64 : i32]}> : (tensor<1x64xf32>) -> tensor<1x64xf32>
// CHECK: return %[[R]] : tensor<1x64xf32>
func.func private @reshape_1d_to_2d(%arg0: tensor<64xf32>) -> tensor<1x64xf32> {
  %0 = "ttir.reshape"(%arg0) <{shape = [1 : i32, 64 : i32]}> : (tensor<64xf32>) -> tensor<1x64xf32>
  return %0 : tensor<1x64xf32>
}

// CHECK-LABEL: func.func private @reshape_2d_to_1d
// CHECK-SAME: (%arg0: tensor<1x64xf32>) -> tensor<1x64xf32>
// CHECK: %[[R:.*]] = "ttir.reshape"(%arg0) <{shape = [1 : i32, 64 : i32]}> : (tensor<1x64xf32>) -> tensor<1x64xf32>
// CHECK: return %[[R]] : tensor<1x64xf32>
func.func private @reshape_2d_to_1d(%arg0: tensor<1x64xf32>) -> tensor<64xf32> {
  %0 = "ttir.reshape"(%arg0) <{shape = [64 : i32]}> : (tensor<1x64xf32>) -> tensor<64xf32>
  return %0 : tensor<64xf32>
}

// =============================================================================
// Test 10: 0D tensor promoted to rank 2 (1x1)
// =============================================================================

// After rank-normalization + canonicalize, identity reshapes are folded:
// CHECK-CANON: func.func private @input_already_reshaped
// CHECK-CANON-SAME: (%arg0: tensor<1x128xf32>) -> tensor<1x128xf32>
// CHECK-CANON-NOT: ttir.reshape
// CHECK-CANON: %[[ABS:.*]] = "ttir.abs"(%arg0)
// CHECK-CANON: return %[[ABS]] : tensor<1x128xf32>

// CHECK-CANON: func.func private @output_1d_only
// CHECK-CANON-SAME: (%arg0: tensor<1x128xf32>) -> tensor<1x128xf32>
// CHECK-CANON-NOT: ttir.reshape
// CHECK-CANON: %[[ABS:.*]] = "ttir.abs"(%arg0)
// CHECK-CANON: return %[[ABS]] : tensor<1x128xf32>

// CHECK-CANON: func.func private @reshape_shape_attr_promoted
// CHECK-CANON-SAME: (%arg0: tensor<1x64xf32>) -> tensor<1x64xf32>
// CHECK-CANON-NOT: ttir.reshape
// CHECK-CANON: return %arg0 : tensor<1x64xf32>

// CHECK-CANON: func.func private @reshape_1d_to_1d
// CHECK-CANON-SAME: (%arg0: tensor<1x64xf32>) -> tensor<1x64xf32>
// CHECK-CANON-NOT: ttir.reshape
// CHECK-CANON: return %arg0 : tensor<1x64xf32>

// CHECK-CANON: func.func private @reshape_1d_to_2d
// CHECK-CANON-SAME: (%arg0: tensor<1x64xf32>) -> tensor<1x64xf32>
// CHECK-CANON-NOT: ttir.reshape
// CHECK-CANON: return %arg0 : tensor<1x64xf32>

// CHECK-CANON: func.func private @reshape_2d_to_1d
// CHECK-CANON-SAME: (%arg0: tensor<1x64xf32>) -> tensor<1x64xf32>
// CHECK-CANON-NOT: ttir.reshape
// CHECK-CANON: return %arg0 : tensor<1x64xf32>

// CHECK-LABEL: func.func private @scalar_promoted
// CHECK-SAME: (%arg0: tensor<1x1xf32>) -> tensor<1x1xf32>
// CHECK: %[[ABS:.*]] = "ttir.abs"(%arg0) : (tensor<1x1xf32>) -> tensor<1x1xf32>
// CHECK: return %[[ABS]] : tensor<1x1xf32>
func.func private @scalar_promoted(%arg0: tensor<f32>) -> tensor<f32> {
  %0 = "ttir.abs"(%arg0) : (tensor<f32>) -> tensor<f32>
  return %0 : tensor<f32>
}

// =============================================================================
// Test 10b: 1D constant - result type and value attribute both promoted to 2D
// =============================================================================
// ttir.constant has AllShapesMatch<["value", "result"]>; the pass must update
// the value attribute to match the promoted result type (same data, new shape).

// CHECK-LABEL: func.func private @constant_1d_promoted
// CHECK-SAME: () -> tensor<1x4xi64>
// CHECK: %[[C:.*]] = "ttir.constant"() <{value = dense<{{.*}}> : tensor<1x4xi64>}> : () -> tensor<1x4xi64>
// CHECK: return %[[C]] : tensor<1x4xi64>
func.func private @constant_1d_promoted() -> tensor<4xi64> {
  %0 = "ttir.constant"() <{value = dense<[0, 1, 2, 3]> : tensor<4xi64>}> : () -> tensor<4xi64>
  return %0 : tensor<4xi64>
}

// =============================================================================
// 1D ttir.full - shape attr promoted to match result (verifier)
// =============================================================================
// ttir.full requires `shape` to match the result tensor shape; rank normalization
// promotes tensor<1xsi32> to tensor<1x1xsi32> and must prepend 1 to `shape`.

// CHECK-LABEL: func.func private @full_1d_with_1d_arg
// CHECK-SAME: (%arg0: tensor<1x1xsi32>) -> tensor<1x1xsi32>
// CHECK: %[[F:.*]] = "ttir.full"() <{fill_value = 128 : i32, shape = array<i32: 1, 1>}> : () -> tensor<1x1xsi32>
// CHECK: %[[ADD:.*]] = "ttir.add"(%arg0, %[[F]]) : (tensor<1x1xsi32>, tensor<1x1xsi32>) -> tensor<1x1xsi32>
// CHECK: return %[[ADD]] : tensor<1x1xsi32>
func.func private @full_1d_with_1d_arg(%arg0: tensor<1xsi32>) -> tensor<1xsi32> {
  %0 = "ttir.full"() <{fill_value = 128 : i32, shape = array<i32: 1>}> : () -> tensor<1xsi32>
  %1 = "ttir.add"(%arg0, %0) : (tensor<1xsi32>, tensor<1xsi32>) -> tensor<1xsi32>
  return %1 : tensor<1xsi32>
}

// =============================================================================
// 1D ttir.zeros - shape attr promoted to match result
// =============================================================================

// CHECK-LABEL: func.func private @zeros_1d_promoted
// CHECK-SAME: (%arg0: tensor<1x64xf32>) -> tensor<1x64xf32>
// CHECK: %[[Z:.*]] = "ttir.zeros"() <{shape = array<i32: 1, 64>}> : () -> tensor<1x64xf32>
// CHECK: %[[ADD:.*]] = "ttir.add"(%arg0, %[[Z]]) : (tensor<1x64xf32>, tensor<1x64xf32>) -> tensor<1x64xf32>
// CHECK: return %[[ADD]] : tensor<1x64xf32>
func.func private @zeros_1d_promoted(%arg0: tensor<64xf32>) -> tensor<64xf32> {
  %0 = "ttir.zeros"() <{shape = array<i32: 64>}> : () -> tensor<64xf32>
  %1 = "ttir.add"(%arg0, %0) : (tensor<64xf32>, tensor<64xf32>) -> tensor<64xf32>
  return %1 : tensor<64xf32>
}

// =============================================================================
// 1D ttir.ones - shape attr promoted to match result
// =============================================================================

// CHECK-LABEL: func.func private @ones_1d_promoted
// CHECK-SAME: (%arg0: tensor<1x128xbf16>) -> tensor<1x128xbf16>
// CHECK: %[[O:.*]] = "ttir.ones"() <{shape = array<i32: 1, 128>}> : () -> tensor<1x128xbf16>
// CHECK: %[[ADD:.*]] = "ttir.add"(%arg0, %[[O]]) : (tensor<1x128xbf16>, tensor<1x128xbf16>) -> tensor<1x128xbf16>
// CHECK: return %[[ADD]] : tensor<1x128xbf16>
func.func private @ones_1d_promoted(%arg0: tensor<128xbf16>) -> tensor<128xbf16> {
  %0 = "ttir.ones"() <{shape = array<i32: 128>}> : () -> tensor<128xbf16>
  %1 = "ttir.add"(%arg0, %0) : (tensor<128xbf16>, tensor<128xbf16>) -> tensor<128xbf16>
  return %1 : tensor<128xbf16>
}

// =============================================================================
// Test 10c: 1D arange - result type promoted to 2D, arange_dimension 0 -> 1
// =============================================================================
// ttir.arange verifies result shape at arange_dimension equals (end-start)/step;
// after promoting [128] to [1, 128], the range dim is now at index 1.

// CHECK-LABEL: func.func private @arange_1d_promoted
// CHECK-SAME: () -> tensor<1x8xi64>
// CHECK: %[[A:.*]] = "ttir.arange"() <{arange_dimension = 1 : i64, end = 8 : si64, start = 0 : si64, step = 1 : si64}> : () -> tensor<1x8xi64>
// CHECK: return %[[A]] : tensor<1x8xi64>
func.func private @arange_1d_promoted() -> tensor<8xi64> {
  %0 = "ttir.arange"() <{arange_dimension = 0 : i64, end = 8 : si64, start = 0 : si64, step = 1 : si64}> : () -> tensor<8xi64>
  return %0 : tensor<8xi64>
}

// =============================================================================
// Test 11: Different element types - promotion is shape-only
// =============================================================================

// CHECK-LABEL: func.func private @different_dtype_bf16
// CHECK-SAME: (%arg0: tensor<1x256xbf16>) -> tensor<1x256xbf16>
// CHECK: %[[ABS:.*]] = "ttir.abs"(%arg0) : (tensor<1x256xbf16>) -> tensor<1x256xbf16>
// CHECK: return %[[ABS]] : tensor<1x256xbf16>
func.func private @different_dtype_bf16(%arg0: tensor<256xbf16>) -> tensor<256xbf16> {
  %0 = "ttir.abs"(%arg0) : (tensor<256xbf16>) -> tensor<256xbf16>
  return %0 : tensor<256xbf16>
}

// =============================================================================
// Test 12: Higher rank (3D) unchanged
// =============================================================================

// CHECK-LABEL: func.func private @higher_rank_unchanged
// CHECK-SAME: (%arg0: tensor<2x32x64xf32>) -> tensor<2x32x64xf32>
// CHECK-NOT: ttir.reshape
// CHECK: %[[ABS:.*]] = "ttir.abs"(%arg0) : (tensor<2x32x64xf32>) -> tensor<2x32x64xf32>
// CHECK: return %[[ABS]] : tensor<2x32x64xf32>
func.func private @higher_rank_unchanged(%arg0: tensor<2x32x64xf32>) -> tensor<2x32x64xf32> {
  %0 = "ttir.abs"(%arg0) : (tensor<2x32x64xf32>) -> tensor<2x32x64xf32>
  return %0 : tensor<2x32x64xf32>
}

// =============================================================================
// Test 13: Multiple returns with 1D - all promoted
// =============================================================================

// CHECK-LABEL: func.func private @multiple_returns
// CHECK-SAME: (%arg0: tensor<1x128xf32>) -> (tensor<1x128xf32>, tensor<1x128xf32>)
// CHECK: %[[ABS:.*]] = "ttir.abs"(%arg0) : (tensor<1x128xf32>) -> tensor<1x128xf32>
// CHECK: %[[NEG:.*]] = "ttir.neg"(%arg0) : (tensor<1x128xf32>) -> tensor<1x128xf32>
// CHECK: return %[[ABS]], %[[NEG]] : tensor<1x128xf32>, tensor<1x128xf32>
func.func private @multiple_returns(%arg0: tensor<128xf32>) -> (tensor<128xf32>, tensor<128xf32>) {
  %0 = "ttir.abs"(%arg0) : (tensor<128xf32>) -> tensor<128xf32>
  %1 = "ttir.neg"(%arg0) : (tensor<128xf32>) -> tensor<128xf32>
  return %0, %1 : tensor<128xf32>, tensor<128xf32>
}

// =============================================================================
// Test 14: Function with no promotion needed - early exit, body unchanged
// =============================================================================

// CHECK-LABEL: func.func private @all_2d_no_promotion
// CHECK-SAME: (%arg0: tensor<8x16xf32>) -> tensor<8x16xf32>
// CHECK: %0 = "ttir.abs"(%arg0) : (tensor<8x16xf32>) -> tensor<8x16xf32>
// CHECK: return %0 : tensor<8x16xf32>
func.func private @all_2d_no_promotion(%arg0: tensor<8x16xf32>) -> tensor<8x16xf32> {
  %0 = "ttir.abs"(%arg0) : (tensor<8x16xf32>) -> tensor<8x16xf32>
  return %0 : tensor<8x16xf32>
}

// =============================================================================
// Test 15: Implicit broadcast - 1D tensor added to 3D tensor
// =============================================================================
// The 1D operand is promoted to 2D; the 3D operand and result are unchanged.

// CHECK-LABEL: func.func private @implicit_broadcast_1d_3d
// CHECK-SAME: (%arg0: tensor<1x64xf32>, %arg1: tensor<2x32x64xf32>) -> tensor<2x32x64xf32>
// CHECK: %[[ADD:.*]] = "ttir.add"(%arg0, %arg1) : (tensor<1x64xf32>, tensor<2x32x64xf32>) -> tensor<2x32x64xf32>
// CHECK: return %[[ADD]] : tensor<2x32x64xf32>
func.func private @implicit_broadcast_1d_3d(%arg0: tensor<64xf32>, %arg1: tensor<2x32x64xf32>) -> tensor<2x32x64xf32> {
  %0 = "ttir.add"(%arg0, %arg1) : (tensor<64xf32>, tensor<2x32x64xf32>) -> tensor<2x32x64xf32>
  return %0 : tensor<2x32x64xf32>
}

// =============================================================================
// Test 16: Slice - 1D input
// =============================================================================

// CHECK-LABEL: func.func private @slice_static_1d
// CHECK-SAME: (%arg0: tensor<1x128xf32>) -> tensor<1x64xf32>
// CHECK: %[[SLICE:.*]] = "ttir.slice_static"(%arg0) <{begins = [0 : i32, 1 : i32], ends = [1 : i32, 128 : i32], step = [1 : i32, 2 : i32]}> : (tensor<1x128xf32>) -> tensor<1x64xf32>
// CHECK: return %[[SLICE]] : tensor<1x64xf32>
func.func private @slice_static_1d(%arg0: tensor<128xf32>) -> tensor<64xf32> {
  %0 = "ttir.slice_static"(%arg0) <{begins = [1 : i32], ends = [128 : i32], step = [2 : i32]}> : (tensor<128xf32>) -> tensor<64xf32>
  return %0 : tensor<64xf32>
}

// =============================================================================
// Test 17: Broadcast - 1D input and output
// =============================================================================

// CHECK-LABEL: func.func private @broadcast_1d_promoted
// CHECK-SAME: (%arg0: tensor<1x1xf32>) -> tensor<1x1200xf32>
// CHECK: %[[R:.*]] = "ttir.reshape"(%arg0) <{shape = [1 : i32, 1 : i32]}> : (tensor<1x1xf32>) -> tensor<1x1xf32>
// CHECK: %[[B:.*]] = "ttir.broadcast"(%[[R]]) <{broadcast_dimensions = array<i64: 1, 1200>}> : (tensor<1x1xf32>) -> tensor<1x1200xf32>
// CHECK: return %[[B]] : tensor<1x1200xf32>
func.func private @broadcast_1d_promoted(%arg0: tensor<f32>) -> tensor<1200xf32> {
  %0 = "ttir.reshape"(%arg0) <{shape = [1 : i32]}> : (tensor<f32>) -> tensor<1xf32>
  %1 = "ttir.broadcast"(%0) <{broadcast_dimensions = array<i64: 1200>}> : (tensor<1xf32>) -> tensor<1200xf32>
  return %1 : tensor<1200xf32>
}

// =============================================================================
// Public (boundary) function tests: signature is preserved, reshape ops are
// inserted at entry and exit so the body can still operate on rank>=2.
// =============================================================================

// Scalar identity: signature stays rank-0; entry reshape promotes to rank-2,
// exit reshape squeezes back.
// CHECK-LABEL: func.func public @public_scalar_identity
// CHECK-SAME: (%arg0: tensor<f32>) -> tensor<f32>
// CHECK: %[[E:.*]] = "ttir.reshape"(%arg0) <{shape = [1 : i32, 1 : i32]}> {ttir.boundary_reshape} : (tensor<f32>) -> tensor<1x1xf32>
// CHECK: %[[X:.*]] = "ttir.reshape"(%[[E]]) <{shape = []}> {ttir.boundary_reshape} : (tensor<1x1xf32>) -> tensor<f32>
// CHECK: return %[[X]] : tensor<f32>
func.func public @public_scalar_identity(%arg0: tensor<f32>) -> tensor<f32> {
  return %arg0 : tensor<f32>
}

// Scalar add: function signature stays rank-0, but `ttir.add` now runs on
// rank-2 tensors (this is what unblocks the rank-0 binary_ng issue).
// CHECK-LABEL: func.func public @public_scalar_add
// CHECK-SAME: (%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32>
// CHECK: %[[E0:.*]] = "ttir.reshape"(%arg0) <{shape = [1 : i32, 1 : i32]}> {ttir.boundary_reshape} : (tensor<f32>) -> tensor<1x1xf32>
// CHECK: %[[E1:.*]] = "ttir.reshape"(%arg1) <{shape = [1 : i32, 1 : i32]}> {ttir.boundary_reshape} : (tensor<f32>) -> tensor<1x1xf32>
// CHECK: %[[A:.*]] = "ttir.add"(%[[E0]], %[[E1]]) : (tensor<1x1xf32>, tensor<1x1xf32>) -> tensor<1x1xf32>
// CHECK: %[[X:.*]] = "ttir.reshape"(%[[A]]) <{shape = []}> {ttir.boundary_reshape} : (tensor<1x1xf32>) -> tensor<f32>
// CHECK: return %[[X]] : tensor<f32>
func.func public @public_scalar_add(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
  %0 = "ttir.add"(%arg0, %arg1) : (tensor<f32>, tensor<f32>) -> tensor<f32>
  return %0 : tensor<f32>
}

// Rank-1 boundary: signature stays rank-1.
// CHECK-LABEL: func.func public @public_rank1
// CHECK-SAME: (%arg0: tensor<5xf32>) -> tensor<5xf32>
// CHECK: %[[E:.*]] = "ttir.reshape"(%arg0) <{shape = [1 : i32, 5 : i32]}> {ttir.boundary_reshape} : (tensor<5xf32>) -> tensor<1x5xf32>
// CHECK: %[[X:.*]] = "ttir.reshape"(%[[E]]) <{shape = [5 : i32]}> {ttir.boundary_reshape} : (tensor<1x5xf32>) -> tensor<5xf32>
// CHECK: return %[[X]] : tensor<5xf32>
func.func public @public_rank1(%arg0: tensor<5xf32>) -> tensor<5xf32> {
  return %arg0 : tensor<5xf32>
}

// Rank>=2 public function is untouched (no reshapes inserted, no signature
// change).
// CHECK-LABEL: func.func public @public_rank2_unchanged
// CHECK-SAME: (%arg0: tensor<2x3xf32>) -> tensor<2x3xf32>
// CHECK-NOT: ttir.reshape
// CHECK: return %arg0 : tensor<2x3xf32>
func.func public @public_rank2_unchanged(%arg0: tensor<2x3xf32>) -> tensor<2x3xf32> {
  return %arg0 : tensor<2x3xf32>
}

// =============================================================================
// Test: rank-1 ttir.mesh_shard is preserved at the public boundary.
// =============================================================================
//
// `ttir.mesh_shard` has a folder that returns its input when the shard volume
// is 1. If the pass marked it illegal, MLIR's conversion framework would call
// the folder during legalizeWithFold, and the rank promotion of the operand
// (without an accompanying result-type rewrite) would trigger an "incorrect
// fold result type" assertion. The pass therefore keeps mesh_shard legal and
// wraps it with explicit demote/promote reshapes so the rest of the body can
// still use rank>=2 values around it. This mirrors the JAX/sdy entry point
// where mesh_shard ops live in the public main function.

// CHECK-LABEL: func.func public @public_mesh_shard_rank1
// CHECK-SAME: (%arg0: tensor<2560xf32>) -> tensor<2560xf32>
// Entry boundary reshape promotes the public arg to rank-2 for the body.
// CHECK: %[[E:.*]] = "ttir.reshape"(%arg0) <{shape = [1 : i32, 2560 : i32]}> {ttir.boundary_reshape} : (tensor<2560xf32>) -> tensor<1x2560xf32>
// Phase 1.5 demotes back to rank-1 right before the mesh_shard.
// CHECK: %[[D:.*]] = "ttir.reshape"(%[[E]]) <{shape = [2560 : i32]}> {ttir.boundary_reshape} : (tensor<1x2560xf32>) -> tensor<2560xf32>
// Mesh_shard stays rank-1 -> rank-1, untouched by the pass.
// CHECK: %[[M:.*]] = "ttir.mesh_shard"(%[[D]])
// CHECK-SAME: (tensor<2560xf32>) -> tensor<2560xf32>
// Phase 1.5 promotes the result to rank-2 for downstream consumers.
// CHECK: %[[P:.*]] = "ttir.reshape"(%[[M]]) <{shape = [1 : i32, 2560 : i32]}> {ttir.boundary_reshape} : (tensor<2560xf32>) -> tensor<1x2560xf32>
// Exit boundary reshape squeezes back to rank-1 for the public return.
// CHECK: %[[X:.*]] = "ttir.reshape"(%[[P]]) <{shape = [2560 : i32]}> {ttir.boundary_reshape} : (tensor<1x2560xf32>) -> tensor<2560xf32>
// CHECK: return %[[X]] : tensor<2560xf32>
func.func public @public_mesh_shard_rank1(%arg0: tensor<2560xf32>) -> tensor<2560xf32> {
  %0 = "ttir.mesh_shard"(%arg0) <{
    shard_direction = #ttcore.shard_direction<full_to_shard>,
    shard_dims = array<i64: -1>,
    shard_shape = array<i64: 1>,
    shard_type = #ttcore.shard_type<replicate>}> : (tensor<2560xf32>) -> tensor<2560xf32>
  return %0 : tensor<2560xf32>
}

// Rank>=2 mesh_shard at the public boundary is untouched (no inserted
// boundary reshapes since neither operand nor result is rank<minRank).
// CHECK-LABEL: func.func public @public_mesh_shard_rank2_unchanged
// CHECK-NOT: ttir.boundary_reshape
// CHECK: %[[M:.*]] = "ttir.mesh_shard"(%arg0)
// CHECK-SAME: (tensor<2560x32xf32>) -> tensor<2560x32xf32>
// CHECK: return %[[M]] : tensor<2560x32xf32>
func.func public @public_mesh_shard_rank2_unchanged(%arg0: tensor<2560x32xf32>) -> tensor<2560x32xf32> {
  %0 = "ttir.mesh_shard"(%arg0) <{
    shard_direction = #ttcore.shard_direction<full_to_shard>,
    shard_dims = array<i64: -1>,
    shard_shape = array<i64: 1>,
    shard_type = #ttcore.shard_type<replicate>}> : (tensor<2560x32xf32>) -> tensor<2560x32xf32>
  return %0 : tensor<2560x32xf32>
}

// =============================================================================
// Reduction op tests: result rank is determined by input rank, dim_arg, and
// keep_dim. The pass must NOT promote the reduction's result type in place,
// because that would leave dim_arg/keep_dim out of sync with the new rank
// and trigger the verifier's "Expected output shape (...), got (...)" error.
// Instead, the reduction op is wrapped with boundary reshapes (demote on any
// rank<minRank operand, promote on any rank<minRank result) and left
// untouched, like other rank-strict ops.
// =============================================================================

// Reducing all dims of a rank-2 input produces a rank-0 result. The op stays
// rank-2 -> rank-0; a promote reshape rewires downstream consumers.
// This is the case that broke the SHLO->TTIR pipeline before the fix.
// CHECK-LABEL: func.func private @sum_rank2_to_scalar
// CHECK-SAME: (%arg0: tensor<2x4xf32>) -> tensor<1x1xf32>
// Reduction op is untouched (operand and dim_arg/keep_dim/result type intact).
// CHECK: %[[S:.*]] = "ttir.sum"(%arg0) <{dim_arg = [0 : i32, 1 : i32], keep_dim = false}> : (tensor<2x4xf32>) -> tensor<f32>
// Promote reshape lifts the rank-0 result to rank-2 for downstream/return.
// CHECK: %[[P:.*]] = "ttir.reshape"(%[[S]]) <{shape = [1 : i32, 1 : i32]}> {ttir.boundary_reshape} : (tensor<f32>) -> tensor<1x1xf32>
// CHECK: return %[[P]] : tensor<1x1xf32>
func.func private @sum_rank2_to_scalar(%arg0: tensor<2x4xf32>) -> tensor<f32> {
  %0 = "ttir.sum"(%arg0) <{dim_arg = [0 : i32, 1 : i32], keep_dim = false}> : (tensor<2x4xf32>) -> tensor<f32>
  return %0 : tensor<f32>
}

// Reducing the only dim of a rank-1 input via a public boundary func. The
// public signature stays rank-1, the entry boundary reshape promotes the arg
// to rank-2 for the body, the demote reshape squeezes back to rank-1 for the
// reduction (which stays rank-1 -> rank-0), and the promote reshape lifts the
// scalar result back to rank-2 for the body's exit reshape.
// CHECK-LABEL: func.func public @public_sum_rank1_to_scalar
// CHECK-SAME: (%arg0: tensor<8xf32>) -> tensor<f32>
// CHECK: %[[E:.*]] = "ttir.reshape"(%arg0) <{shape = [1 : i32, 8 : i32]}> {ttir.boundary_reshape} : (tensor<8xf32>) -> tensor<1x8xf32>
// CHECK: %[[D:.*]] = "ttir.reshape"(%[[E]]) <{shape = [8 : i32]}> {ttir.boundary_reshape} : (tensor<1x8xf32>) -> tensor<8xf32>
// CHECK: %[[S:.*]] = "ttir.sum"(%[[D]]) <{dim_arg = [0 : i32], keep_dim = false}> : (tensor<8xf32>) -> tensor<f32>
// CHECK: %[[P:.*]] = "ttir.reshape"(%[[S]]) <{shape = [1 : i32, 1 : i32]}> {ttir.boundary_reshape} : (tensor<f32>) -> tensor<1x1xf32>
// CHECK: %[[X:.*]] = "ttir.reshape"(%[[P]]) <{shape = []}> {ttir.boundary_reshape} : (tensor<1x1xf32>) -> tensor<f32>
// CHECK: return %[[X]] : tensor<f32>
func.func public @public_sum_rank1_to_scalar(%arg0: tensor<8xf32>) -> tensor<f32> {
  %0 = "ttir.sum"(%arg0) <{dim_arg = [0 : i32], keep_dim = false}> : (tensor<8xf32>) -> tensor<f32>
  return %0 : tensor<f32>
}

// Partial reduction: rank-3 -> rank-2. Both operand and result are already
// rank>=minRank, so the reduction op is left fully unchanged and no
// boundary reshapes are inserted around it.
// CHECK-LABEL: func.func private @sum_rank3_partial_unchanged
// CHECK-SAME: (%arg0: tensor<2x3x4xf32>) -> tensor<2x4xf32>
// CHECK-NOT: ttir.boundary_reshape
// CHECK: %[[S:.*]] = "ttir.sum"(%arg0) <{dim_arg = [1 : i32], keep_dim = false}> : (tensor<2x3x4xf32>) -> tensor<2x4xf32>
// CHECK: return %[[S]] : tensor<2x4xf32>
func.func private @sum_rank3_partial_unchanged(%arg0: tensor<2x3x4xf32>) -> tensor<2x4xf32> {
  %0 = "ttir.sum"(%arg0) <{dim_arg = [1 : i32], keep_dim = false}> : (tensor<2x3x4xf32>) -> tensor<2x4xf32>
  return %0 : tensor<2x4xf32>
}

// Partial reduction with rank-1 result: rank-2 -> rank-1. The result is
// rank<minRank, so a promote reshape is inserted after the op. The operand
// stays rank-2 (already >= minRank).
// CHECK-LABEL: func.func private @mean_rank2_to_rank1
// CHECK-SAME: (%arg0: tensor<3x5xf32>) -> tensor<1x5xf32>
// CHECK: %[[M:.*]] = "ttir.mean"(%arg0) <{dim_arg = [0 : i32], keep_dim = false}> : (tensor<3x5xf32>) -> tensor<5xf32>
// CHECK: %[[P:.*]] = "ttir.reshape"(%[[M]]) <{shape = [1 : i32, 5 : i32]}> {ttir.boundary_reshape} : (tensor<5xf32>) -> tensor<1x5xf32>
// CHECK: return %[[P]] : tensor<1x5xf32>
func.func private @mean_rank2_to_rank1(%arg0: tensor<3x5xf32>) -> tensor<5xf32> {
  %0 = "ttir.mean"(%arg0) <{dim_arg = [0 : i32], keep_dim = false}> : (tensor<3x5xf32>) -> tensor<5xf32>
  return %0 : tensor<5xf32>
}

// Reducing all dims of a rank-3 input -> rank-0 result. Operand is already
// rank>=minRank so no demote; only a promote reshape after the op.
// CHECK-LABEL: func.func private @sum_rank3_to_scalar
// CHECK-SAME: (%arg0: tensor<2x3x4xf32>) -> tensor<1x1xf32>
// CHECK-NOT: "ttir.reshape"(%arg0)
// CHECK: %[[S:.*]] = "ttir.sum"(%arg0) <{dim_arg = [0 : i32, 1 : i32, 2 : i32], keep_dim = false}> : (tensor<2x3x4xf32>) -> tensor<f32>
// CHECK: %[[P:.*]] = "ttir.reshape"(%[[S]]) <{shape = [1 : i32, 1 : i32]}> {ttir.boundary_reshape} : (tensor<f32>) -> tensor<1x1xf32>
// CHECK: return %[[P]] : tensor<1x1xf32>
func.func private @sum_rank3_to_scalar(%arg0: tensor<2x3x4xf32>) -> tensor<f32> {
  %0 = "ttir.sum"(%arg0) <{dim_arg = [0 : i32, 1 : i32, 2 : i32], keep_dim = false}> : (tensor<2x3x4xf32>) -> tensor<f32>
  return %0 : tensor<f32>
}

// max with keep_dim=true: result is already rank>=minRank, op is left
// untouched, and no boundary reshapes are inserted.
// CHECK-LABEL: func.func private @max_keep_dim_unchanged
// CHECK-SAME: (%arg0: tensor<2x3x4xf32>) -> tensor<2x3x1xf32>
// CHECK-NOT: ttir.boundary_reshape
// CHECK: %[[M:.*]] = "ttir.max"(%arg0) <{dim_arg = [2 : i32], keep_dim = true}> : (tensor<2x3x4xf32>) -> tensor<2x3x1xf32>
// CHECK: return %[[M]] : tensor<2x3x1xf32>
func.func private @max_keep_dim_unchanged(%arg0: tensor<2x3x4xf32>) -> tensor<2x3x1xf32> {
  %0 = "ttir.max"(%arg0) <{dim_arg = [2 : i32], keep_dim = true}> : (tensor<2x3x4xf32>) -> tensor<2x3x1xf32>
  return %0 : tensor<2x3x1xf32>
}

// Reduction sandwiched between rank-promoted producer and consumer: producer
// (`ttir.abs`) and consumer (`ttir.neg`) both run on rank-2 tensors. The
// reduction stays rank-2 -> rank-0; only a promote reshape is needed (the
// operand is already rank>=minRank).
// CHECK-LABEL: func.func private @sum_with_neighbors
// CHECK-SAME: (%arg0: tensor<2x4xf32>) -> tensor<1x1xf32>
// CHECK: %[[A:.*]] = "ttir.abs"(%arg0) : (tensor<2x4xf32>) -> tensor<2x4xf32>
// CHECK: %[[S:.*]] = "ttir.sum"(%[[A]]) <{dim_arg = [0 : i32, 1 : i32], keep_dim = false}> : (tensor<2x4xf32>) -> tensor<f32>
// CHECK: %[[P:.*]] = "ttir.reshape"(%[[S]]) <{shape = [1 : i32, 1 : i32]}> {ttir.boundary_reshape} : (tensor<f32>) -> tensor<1x1xf32>
// CHECK: %[[N:.*]] = "ttir.neg"(%[[P]]) : (tensor<1x1xf32>) -> tensor<1x1xf32>
// CHECK: return %[[N]] : tensor<1x1xf32>
func.func private @sum_with_neighbors(%arg0: tensor<2x4xf32>) -> tensor<f32> {
  %0 = "ttir.abs"(%arg0) : (tensor<2x4xf32>) -> tensor<2x4xf32>
  %1 = "ttir.sum"(%0) <{dim_arg = [0 : i32, 1 : i32], keep_dim = false}> : (tensor<2x4xf32>) -> tensor<f32>
  %2 = "ttir.neg"(%1) : (tensor<f32>) -> tensor<f32>
  return %2 : tensor<f32>
}

// Public boundary with a reduction to scalar inside: the public signature is
// preserved (rank-0 in / rank-0 out), the reduction is left rank-2 -> rank-0,
// and boundary reshapes glue the rank-0 op to the rank-2 body.
// CHECK-LABEL: func.func public @public_sum_to_scalar
// CHECK-SAME: (%arg0: tensor<2x4xf32>) -> tensor<f32>
// Reduction op stays on its declared types.
// CHECK: %[[S:.*]] = "ttir.sum"(%arg0) <{dim_arg = [0 : i32, 1 : i32], keep_dim = false}> : (tensor<2x4xf32>) -> tensor<f32>
// Promote reshape lifts rank-0 -> rank-2 for the rank-2 body world.
// CHECK: %[[P:.*]] = "ttir.reshape"(%[[S]]) <{shape = [1 : i32, 1 : i32]}> {ttir.boundary_reshape} : (tensor<f32>) -> tensor<1x1xf32>
// Exit boundary reshape squeezes back to rank-0 for the unchanged signature.
// CHECK: %[[X:.*]] = "ttir.reshape"(%[[P]]) <{shape = []}> {ttir.boundary_reshape} : (tensor<1x1xf32>) -> tensor<f32>
// CHECK: return %[[X]] : tensor<f32>
func.func public @public_sum_to_scalar(%arg0: tensor<2x4xf32>) -> tensor<f32> {
  %0 = "ttir.sum"(%arg0) <{dim_arg = [0 : i32, 1 : i32], keep_dim = false}> : (tensor<2x4xf32>) -> tensor<f32>
  return %0 : tensor<f32>
}

// =============================================================================
// Materialization-cast regression tests: when a rank-strict reduction result
// and a rank-0 constant both feed a non-rank-strict op, the conversion
// framework must bridge the rank gap. Previously it inserted an
// `unrealized_conversion_cast` which survived past legalization and caused
// "failed to legalize operation 'builtin.unrealized_conversion_cast'".
// After the fix, `materializeCast` emits a real `ttir.reshape` instead.
// =============================================================================

// Test: masked_cross_entropy_chain (private)
// Mirrors the failing JAX pattern: two ttir.sum reductions -> ttir.maximum
// (with a rank-0 constant) -> ttir.div. All non-rank-strict ops must be
// promoted to rank-2 with no leftover casts.
// CHECK-LABEL: func.func private @masked_cross_entropy_chain
// CHECK-SAME: (%arg0: tensor<2x4xf32>, %arg1: tensor<2x4xf32>) -> tensor<1x1xf32>
// CHECK-NOT: unrealized_conversion_cast
// Sum results are rank-0 with promote reshapes to rank-2.
// CHECK: %[[S1:.*]] = "ttir.sum"(%arg0) <{dim_arg = [0 : i32, 1 : i32], keep_dim = false}> : (tensor<2x4xf32>) -> tensor<f32>
// CHECK: %[[P1:.*]] = "ttir.reshape"(%[[S1]]) <{shape = [1 : i32, 1 : i32]}> {ttir.boundary_reshape} : (tensor<f32>) -> tensor<1x1xf32>
// CHECK: %[[S2:.*]] = "ttir.sum"(%arg1) <{dim_arg = [0 : i32, 1 : i32], keep_dim = false}> : (tensor<2x4xf32>) -> tensor<f32>
// CHECK: %[[P2:.*]] = "ttir.reshape"(%[[S2]]) <{shape = [1 : i32, 1 : i32]}> {ttir.boundary_reshape} : (tensor<f32>) -> tensor<1x1xf32>
// Constant is promoted to rank-2.
// CHECK: %[[C:.*]] = "ttir.constant"() <{value = dense<1.000000e+00> : tensor<1x1xf32>}> : () -> tensor<1x1xf32>
// maximum and div operate on rank-2 tensors.
// CHECK: %[[MAX:.*]] = "ttir.maximum"(%[[P2]], %[[C]]) : (tensor<1x1xf32>, tensor<1x1xf32>) -> tensor<1x1xf32>
// CHECK: %[[DIV:.*]] = "ttir.div"(%[[P1]], %[[MAX]]) : (tensor<1x1xf32>, tensor<1x1xf32>) -> tensor<1x1xf32>
// CHECK: return %[[DIV]] : tensor<1x1xf32>
func.func private @masked_cross_entropy_chain(%arg0: tensor<2x4xf32>, %arg1: tensor<2x4xf32>) -> tensor<f32> {
  %0 = "ttir.sum"(%arg0) <{dim_arg = [0 : i32, 1 : i32], keep_dim = false}> : (tensor<2x4xf32>) -> tensor<f32>
  %1 = "ttir.sum"(%arg1) <{dim_arg = [0 : i32, 1 : i32], keep_dim = false}> : (tensor<2x4xf32>) -> tensor<f32>
  %c = "ttir.constant"() <{value = dense<1.0> : tensor<f32>}> : () -> tensor<f32>
  %2 = "ttir.maximum"(%1, %c) : (tensor<f32>, tensor<f32>) -> tensor<f32>
  %3 = "ttir.div"(%0, %2) : (tensor<f32>, tensor<f32>) -> tensor<f32>
  return %3 : tensor<f32>
}

// Test: materialize_const_into_promoted_op (private)
// Minimal repro: rank-0 constant + rank-strict reduction result feeding a
// non-rank-strict op (ttir.add). Ensures the constant is promoted via a
// reshape rather than an unrealized_conversion_cast.
// CHECK-LABEL: func.func private @materialize_const_into_promoted_op
// CHECK-SAME: (%arg0: tensor<3x4xf32>) -> tensor<1x1xf32>
// CHECK-NOT: unrealized_conversion_cast
// CHECK: %[[S:.*]] = "ttir.sum"(%arg0) <{dim_arg = [0 : i32, 1 : i32], keep_dim = false}> : (tensor<3x4xf32>) -> tensor<f32>
// CHECK: %[[P:.*]] = "ttir.reshape"(%[[S]]) <{shape = [1 : i32, 1 : i32]}> {ttir.boundary_reshape} : (tensor<f32>) -> tensor<1x1xf32>
// CHECK: %[[C:.*]] = "ttir.constant"() <{value = dense<2.000000e+00> : tensor<1x1xf32>}> : () -> tensor<1x1xf32>
// CHECK: %[[A:.*]] = "ttir.add"(%[[P]], %[[C]]) : (tensor<1x1xf32>, tensor<1x1xf32>) -> tensor<1x1xf32>
// CHECK: return %[[A]] : tensor<1x1xf32>
func.func private @materialize_const_into_promoted_op(%arg0: tensor<3x4xf32>) -> tensor<f32> {
  %0 = "ttir.sum"(%arg0) <{dim_arg = [0 : i32, 1 : i32], keep_dim = false}> : (tensor<3x4xf32>) -> tensor<f32>
  %c = "ttir.constant"() <{value = dense<2.0> : tensor<f32>}> : () -> tensor<f32>
  %1 = "ttir.add"(%0, %c) : (tensor<f32>, tensor<f32>) -> tensor<f32>
  return %1 : tensor<f32>
}

// Test: boundary_func_masked_cross_entropy_chain (public)
// Same masked_cross_entropy chain inside a public/JIT-style function.
// Confirms public-boundary path also stays cast-free with the exit reshape
// inserted exactly once.
// CHECK-LABEL: func.func public @boundary_func_masked_cross_entropy_chain
// CHECK-SAME: (%arg0: tensor<2x4xf32>, %arg1: tensor<2x4xf32>) -> tensor<f32>
// CHECK-NOT: unrealized_conversion_cast
// Sum results with promote reshapes.
// CHECK: %[[S1:.*]] = "ttir.sum"(%arg0) <{dim_arg = [0 : i32, 1 : i32], keep_dim = false}> : (tensor<2x4xf32>) -> tensor<f32>
// CHECK: %[[P1:.*]] = "ttir.reshape"(%[[S1]]) <{shape = [1 : i32, 1 : i32]}> {ttir.boundary_reshape} : (tensor<f32>) -> tensor<1x1xf32>
// CHECK: %[[S2:.*]] = "ttir.sum"(%arg1) <{dim_arg = [0 : i32, 1 : i32], keep_dim = false}> : (tensor<2x4xf32>) -> tensor<f32>
// CHECK: %[[P2:.*]] = "ttir.reshape"(%[[S2]]) <{shape = [1 : i32, 1 : i32]}> {ttir.boundary_reshape} : (tensor<f32>) -> tensor<1x1xf32>
// Constant promoted to rank-2.
// CHECK: %[[C:.*]] = "ttir.constant"() <{value = dense<1.000000e+00> : tensor<1x1xf32>}> : () -> tensor<1x1xf32>
// maximum and div on rank-2.
// CHECK: %[[MAX:.*]] = "ttir.maximum"(%[[P2]], %[[C]]) : (tensor<1x1xf32>, tensor<1x1xf32>) -> tensor<1x1xf32>
// CHECK: %[[DIV:.*]] = "ttir.div"(%[[P1]], %[[MAX]]) : (tensor<1x1xf32>, tensor<1x1xf32>) -> tensor<1x1xf32>
// Exit boundary reshape squeezes back to rank-0 for the public return.
// CHECK: %[[X:.*]] = "ttir.reshape"(%[[DIV]]) <{shape = []}> {ttir.boundary_reshape} : (tensor<1x1xf32>) -> tensor<f32>
// CHECK: return %[[X]] : tensor<f32>
func.func public @boundary_func_masked_cross_entropy_chain(%arg0: tensor<2x4xf32>, %arg1: tensor<2x4xf32>) -> tensor<f32> {
  %0 = "ttir.sum"(%arg0) <{dim_arg = [0 : i32, 1 : i32], keep_dim = false}> : (tensor<2x4xf32>) -> tensor<f32>
  %1 = "ttir.sum"(%arg1) <{dim_arg = [0 : i32, 1 : i32], keep_dim = false}> : (tensor<2x4xf32>) -> tensor<f32>
  %c = "ttir.constant"() <{value = dense<1.0> : tensor<f32>}> : () -> tensor<f32>
  %2 = "ttir.maximum"(%1, %c) : (tensor<f32>, tensor<f32>) -> tensor<f32>
  %3 = "ttir.div"(%0, %2) : (tensor<f32>, tensor<f32>) -> tensor<f32>
  return %3 : tensor<f32>
}

// =============================================================================
// dot_general rank-strict regression tests
//
// Background: `ttir.dot_general`'s result rank is
//   lhs_rank + rhs_rank - 2*|contract| - |batch|.
// If RankNormalization promotes the operands without also adjusting the result
// type / contract / batch attributes, the op's declared result type stops
// matching what its operands actually produce. The downstream
// TTIRToTTIRDecomposition pass then rebuilds the shape from the post-promotion
// operand ranks (e.g. (1x40960) x (1x64) -> 1x40960x1x64) and the dialect
// conversion framework leaves behind an `unrealized_conversion_cast` that
// TTIRToTTNNCommon's 1:1 type converter cannot resolve. Adding DotGeneralOp
// to `hasRankStrictOperandInvariants` keeps its operand/result types exactly
// as SHLO->TTIR declared them by wrapping operands with boundary reshapes.
//
// The pattern below mirrors EasyDeL's rotary `compute_basic_frequencies`:
//   einsum("i,j->ij", positions, inv_freqs) -> cos / sin / concat.
// =============================================================================

// Test: dot_general_outer_product_rank1_rank1 (private)
// Both operands are rank-1, result is rank-2. The pass promotes the function
// signature to rank-2 and wraps the rank-1 dot_general with boundary demote
// reshapes, leaving the dot_general itself with rank-1 operands and a rank-2
// result. After canonicalization the duplicate boundary reshapes (one pair
// from Phase 1a, one pair materialized by the type converter) fold to a
// single demote per operand.
// CHECK-LABEL: func.func private @dot_general_outer_product_rank1_rank1
// CHECK-SAME: (%arg0: tensor<1x40960xf32>, %arg1: tensor<1x64xf32>) -> tensor<40960x64xf32>
// CHECK-NOT: unrealized_conversion_cast
// CHECK: "ttir.dot_general"
// CHECK-SAME: (tensor<40960xf32>, tensor<64xf32>) -> tensor<40960x64xf32>
//
// CHECK-CANON-LABEL: func.func private @dot_general_outer_product_rank1_rank1
// CHECK-CANON-SAME: (%arg0: tensor<1x40960xf32>, %arg1: tensor<1x64xf32>) -> tensor<40960x64xf32>
// CHECK-CANON-NOT: unrealized_conversion_cast
// After canonicalization there is exactly one demote reshape per operand and
// the dot_general retains its rank-1 operand / rank-2 result signature.
// CHECK-CANON-DAG: %[[L:.*]] = "ttir.reshape"(%arg0) <{shape = [40960 : i32]}> {ttir.boundary_reshape} : (tensor<1x40960xf32>) -> tensor<40960xf32>
// CHECK-CANON-DAG: %[[R:.*]] = "ttir.reshape"(%arg1) <{shape = [64 : i32]}> {ttir.boundary_reshape} : (tensor<1x64xf32>) -> tensor<64xf32>
// CHECK-CANON: %[[D:.*]] = "ttir.dot_general"(%[[L]], %[[R]])
// CHECK-CANON-SAME: <{batch_dims_lhs = array<i64>, batch_dims_rhs = array<i64>, contract_dims_lhs = array<i64>, contract_dims_rhs = array<i64>}>
// CHECK-CANON-SAME: (tensor<40960xf32>, tensor<64xf32>) -> tensor<40960x64xf32>
// CHECK-CANON: return %[[D]] : tensor<40960x64xf32>
func.func private @dot_general_outer_product_rank1_rank1(%arg0: tensor<40960xf32>, %arg1: tensor<64xf32>) -> tensor<40960x64xf32> {
  %0 = "ttir.dot_general"(%arg0, %arg1) <{batch_dims_lhs = array<i64>, batch_dims_rhs = array<i64>, contract_dims_lhs = array<i64>, contract_dims_rhs = array<i64>}> : (tensor<40960xf32>, tensor<64xf32>) -> tensor<40960x64xf32>
  return %0 : tensor<40960x64xf32>
}

// Test: dot_general_already_rank2_unchanged (private)
// Regression guard: when both operands are already rank>=2, the rank-strict
// classification must be a complete no-op. No boundary reshapes anywhere.
// CHECK-LABEL: func.func private @dot_general_already_rank2_unchanged
// CHECK-SAME: (%arg0: tensor<16x32xf32>, %arg1: tensor<32x64xf32>) -> tensor<16x64xf32>
// CHECK-NOT: ttir.reshape
// CHECK-NOT: unrealized_conversion_cast
// CHECK: %[[D:.*]] = "ttir.dot_general"(%arg0, %arg1)
// CHECK-SAME: <{batch_dims_lhs = array<i64>, batch_dims_rhs = array<i64>, contract_dims_lhs = array<i64: 1>, contract_dims_rhs = array<i64: 0>}>
// CHECK-SAME: (tensor<16x32xf32>, tensor<32x64xf32>) -> tensor<16x64xf32>
// CHECK: return %[[D]] : tensor<16x64xf32>
//
// CHECK-CANON-LABEL: func.func private @dot_general_already_rank2_unchanged
// CHECK-CANON-NOT: ttir.reshape
// CHECK-CANON-NOT: unrealized_conversion_cast
// CHECK-CANON: "ttir.dot_general"(%arg0, %arg1)
// CHECK-CANON-SAME: (tensor<16x32xf32>, tensor<32x64xf32>) -> tensor<16x64xf32>
func.func private @dot_general_already_rank2_unchanged(%arg0: tensor<16x32xf32>, %arg1: tensor<32x64xf32>) -> tensor<16x64xf32> {
  %0 = "ttir.dot_general"(%arg0, %arg1) <{batch_dims_lhs = array<i64>, batch_dims_rhs = array<i64>, contract_dims_lhs = array<i64: 1>, contract_dims_rhs = array<i64: 0>}> : (tensor<16x32xf32>, tensor<32x64xf32>) -> tensor<16x64xf32>
  return %0 : tensor<16x64xf32>
}

// Test: boundary_func_dot_general_outer_product_rank1_rank1 (public)
// Same outer-product pattern but on a public boundary function. The function
// signature stays rank-1; the entry reshape that would promote each arg to
// rank-2 cancels with the rank-strict demote reshape after canonicalization,
// leaving the dot_general operating directly on the rank-1 block arguments.
// CHECK-LABEL: func.func public @boundary_func_dot_general_outer_product_rank1_rank1
// CHECK-SAME: (%arg0: tensor<40960xf32>, %arg1: tensor<64xf32>) -> tensor<40960x64xf32>
// CHECK-NOT: unrealized_conversion_cast
// CHECK: "ttir.dot_general"
// CHECK-SAME: (tensor<40960xf32>, tensor<64xf32>) -> tensor<40960x64xf32>
//
// CHECK-CANON-LABEL: func.func public @boundary_func_dot_general_outer_product_rank1_rank1
// CHECK-CANON-SAME: (%arg0: tensor<40960xf32>, %arg1: tensor<64xf32>) -> tensor<40960x64xf32>
// CHECK-CANON-NOT: ttir.reshape
// CHECK-CANON-NOT: unrealized_conversion_cast
// CHECK-CANON: %[[D:.*]] = "ttir.dot_general"(%arg0, %arg1)
// CHECK-CANON-SAME: (tensor<40960xf32>, tensor<64xf32>) -> tensor<40960x64xf32>
// CHECK-CANON: return %[[D]] : tensor<40960x64xf32>
func.func public @boundary_func_dot_general_outer_product_rank1_rank1(%arg0: tensor<40960xf32>, %arg1: tensor<64xf32>) -> tensor<40960x64xf32> {
  %0 = "ttir.dot_general"(%arg0, %arg1) <{batch_dims_lhs = array<i64>, batch_dims_rhs = array<i64>, contract_dims_lhs = array<i64>, contract_dims_rhs = array<i64>}> : (tensor<40960xf32>, tensor<64xf32>) -> tensor<40960x64xf32>
  return %0 : tensor<40960x64xf32>
}

// =============================================================================
// mesh_partition rank-strict regression tests
//
// Background: `ttir.mesh_partition` carries a `dim : si32` attribute that
// indexes into the operand's shape. The op verifier accepts any `dim` in
// `[-rank, rank)`, so RankNormalization promoting a rank-1 operand to rank-2
// by prepending `1` would silently shift what `dim = 0` refers to (it would
// then point at the prepended unit dim instead of the original data dim).
// The TTNN runtime catches this only at execution time with
//   "input shape Shape([1, N]) must be divisible by cluster axis size K"
// from MeshPartitionDeviceOperation::validate_on_program_cache_miss.
//
// EasyDeL/Shardy emits `sdy.all_slice` composites on rank-1 norm/scale
// parameters (e.g. RMSNorm gamma of shape <2560>); the SHLO->TTIR composite
// legalizer lowers these to `ttir.mesh_partition %x {dim = 0}`. The tests
// below lock in that the pass wraps mesh_partition with boundary reshapes
// (preserving its original rank-1 operand and `dim` value) and leaves any
// already-rank>=2 mesh_partition untouched.
// =============================================================================

// Test: mesh_partition_rank1_dim0 (private)
// Rank-1 operand, dim=0 partitions the only data axis across cluster_axis=0.
// The pass must keep the op's rank-1 operand/result types and dim/cluster_axis
// attributes intact, with demote/promote boundary reshapes around it.
// CHECK-LABEL: func.func private @mesh_partition_rank1_dim0
// CHECK-SAME: (%arg0: tensor<1x2560xf32>) -> tensor<1x320xf32>
// CHECK-NOT: unrealized_conversion_cast
// CHECK: "ttir.mesh_partition"
// CHECK-SAME: <{cluster_axis = 0 : ui32, dim = 0 : si32}> : (tensor<2560xf32>) -> tensor<320xf32>
//
// CHECK-CANON-LABEL: func.func private @mesh_partition_rank1_dim0
// CHECK-CANON-SAME: (%arg0: tensor<1x2560xf32>) -> tensor<1x320xf32>
// CHECK-CANON-NOT: unrealized_conversion_cast
// CHECK-CANON: %[[D:.*]] = "ttir.reshape"(%arg0) <{shape = [2560 : i32]}> {ttir.boundary_reshape} : (tensor<1x2560xf32>) -> tensor<2560xf32>
// CHECK-CANON: %[[M:.*]] = "ttir.mesh_partition"(%[[D]])
// CHECK-CANON-SAME: <{cluster_axis = 0 : ui32, dim = 0 : si32}> : (tensor<2560xf32>) -> tensor<320xf32>
// CHECK-CANON: %[[P:.*]] = "ttir.reshape"(%[[M]]) <{shape = [1 : i32, 320 : i32]}> {ttir.boundary_reshape} : (tensor<320xf32>) -> tensor<1x320xf32>
// CHECK-CANON: return %[[P]] : tensor<1x320xf32>
func.func private @mesh_partition_rank1_dim0(%arg0: tensor<2560xf32>) -> tensor<320xf32> {
  %0 = "ttir.mesh_partition"(%arg0) <{cluster_axis = 0 : ui32, dim = 0 : si32}> : (tensor<2560xf32>) -> tensor<320xf32>
  return %0 : tensor<320xf32>
}

// Test: mesh_partition_already_rank2_unchanged (private)
// Regression guard: rank>=2 operands must not trigger any boundary reshape.
// CHECK-LABEL: func.func private @mesh_partition_already_rank2_unchanged
// CHECK-SAME: (%arg0: tensor<4x32xbf16>) -> tensor<4x16xbf16>
// CHECK-NOT: ttir.reshape
// CHECK-NOT: unrealized_conversion_cast
// CHECK: %[[M:.*]] = "ttir.mesh_partition"(%arg0)
// CHECK-SAME: <{cluster_axis = 0 : ui32, dim = 1 : si32}> : (tensor<4x32xbf16>) -> tensor<4x16xbf16>
// CHECK: return %[[M]] : tensor<4x16xbf16>
//
// CHECK-CANON-LABEL: func.func private @mesh_partition_already_rank2_unchanged
// CHECK-CANON-NOT: ttir.reshape
// CHECK-CANON-NOT: unrealized_conversion_cast
// CHECK-CANON: "ttir.mesh_partition"(%arg0)
// CHECK-CANON-SAME: (tensor<4x32xbf16>) -> tensor<4x16xbf16>
func.func private @mesh_partition_already_rank2_unchanged(%arg0: tensor<4x32xbf16>) -> tensor<4x16xbf16> {
  %0 = "ttir.mesh_partition"(%arg0) <{cluster_axis = 0 : ui32, dim = 1 : si32}> : (tensor<4x32xbf16>) -> tensor<4x16xbf16>
  return %0 : tensor<4x16xbf16>
}

// Test: boundary_func_mesh_partition_rank1 (public)
// JIT-style public boundary: rank-1 mesh_partition inside a public function.
// The signature stays rank-1; the entry-promote / Phase 1a-demote pair
// cancels in canonicalize, leaving the partition directly on the rank-1
// block argument with dim=0 still pointing at the 2560 axis.
// CHECK-LABEL: func.func public @boundary_func_mesh_partition_rank1
// CHECK-SAME: (%arg0: tensor<2560xf32>) -> tensor<320xf32>
// CHECK-NOT: unrealized_conversion_cast
// CHECK: "ttir.mesh_partition"
// CHECK-SAME: <{cluster_axis = 0 : ui32, dim = 0 : si32}> : (tensor<2560xf32>) -> tensor<320xf32>
//
// CHECK-CANON-LABEL: func.func public @boundary_func_mesh_partition_rank1
// CHECK-CANON-SAME: (%arg0: tensor<2560xf32>) -> tensor<320xf32>
// CHECK-CANON-NOT: ttir.reshape
// CHECK-CANON-NOT: unrealized_conversion_cast
// CHECK-CANON: %[[M:.*]] = "ttir.mesh_partition"(%arg0)
// CHECK-CANON-SAME: <{cluster_axis = 0 : ui32, dim = 0 : si32}> : (tensor<2560xf32>) -> tensor<320xf32>
// CHECK-CANON: return %[[M]] : tensor<320xf32>
func.func public @boundary_func_mesh_partition_rank1(%arg0: tensor<2560xf32>) -> tensor<320xf32> {
  %0 = "ttir.mesh_partition"(%arg0) <{cluster_axis = 0 : ui32, dim = 0 : si32}> : (tensor<2560xf32>) -> tensor<320xf32>
  return %0 : tensor<320xf32>
}
