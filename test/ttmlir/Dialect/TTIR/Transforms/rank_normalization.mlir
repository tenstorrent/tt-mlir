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
