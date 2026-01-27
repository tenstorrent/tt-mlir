// RUN: ttmlir-opt --d2m-promote-1d-to-2d %s | FileCheck %s

// =============================================================================
// Test 1: Basic single op with 1D input/output
// =============================================================================

// CHECK-LABEL: func.func @single_op_1d
// CHECK-SAME: (%arg0: tensor<128xf32>) -> tensor<128xf32>
// CHECK: %[[RESHAPE_IN:.*]] = "ttir.reshape"(%arg0) <{shape = [1 : i32, 128 : i32]}> : (tensor<128xf32>) -> tensor<1x128xf32>
// CHECK: %[[ABS:.*]] = "ttir.abs"(%[[RESHAPE_IN]]) : (tensor<1x128xf32>) -> tensor<1x128xf32>
// CHECK: %[[RESHAPE_OUT:.*]] = "ttir.reshape"(%[[ABS]]) <{shape = [128 : i32]}> : (tensor<1x128xf32>) -> tensor<128xf32>
// CHECK: return %[[RESHAPE_OUT]] : tensor<128xf32>
func.func @single_op_1d(%arg0: tensor<128xf32>) -> tensor<128xf32> {
  %0 = "ttir.abs"(%arg0) : (tensor<128xf32>) -> tensor<128xf32>
  return %0 : tensor<128xf32>
}

// =============================================================================
// Test 2: Multiple chained ops - verifies downstream propagation
// =============================================================================

// CHECK-LABEL: func.func @chained_ops_1d
// CHECK-SAME: (%arg0: tensor<128xf32>) -> tensor<128xf32>
// CHECK: %[[RESHAPE_IN:.*]] = "ttir.reshape"(%arg0) <{shape = [1 : i32, 128 : i32]}> : (tensor<128xf32>) -> tensor<1x128xf32>
// CHECK: %[[ABS:.*]] = "ttir.abs"(%[[RESHAPE_IN]]) : (tensor<1x128xf32>) -> tensor<1x128xf32>
// CHECK: %[[NEG:.*]] = "ttir.neg"(%[[ABS]]) : (tensor<1x128xf32>) -> tensor<1x128xf32>
// CHECK: %[[RESHAPE_OUT:.*]] = "ttir.reshape"(%[[NEG]]) <{shape = [128 : i32]}> : (tensor<1x128xf32>) -> tensor<128xf32>
// CHECK: return %[[RESHAPE_OUT]] : tensor<128xf32>
func.func @chained_ops_1d(%arg0: tensor<128xf32>) -> tensor<128xf32> {
  %0 = "ttir.abs"(%arg0) : (tensor<128xf32>) -> tensor<128xf32>
  %1 = "ttir.neg"(%0) : (tensor<128xf32>) -> tensor<128xf32>
  return %1 : tensor<128xf32>
}

// =============================================================================
// Test 3: Long chain of ops - verifies deep propagation
// =============================================================================

// CHECK-LABEL: func.func @long_chain_1d
// CHECK-SAME: (%arg0: tensor<64xf32>) -> tensor<64xf32>
// CHECK: %[[RESHAPE_IN:.*]] = "ttir.reshape"(%arg0) <{shape = [1 : i32, 64 : i32]}> : (tensor<64xf32>) -> tensor<1x64xf32>
// CHECK: %[[OP1:.*]] = "ttir.abs"(%[[RESHAPE_IN]]) : (tensor<1x64xf32>) -> tensor<1x64xf32>
// CHECK: %[[OP2:.*]] = "ttir.neg"(%[[OP1]]) : (tensor<1x64xf32>) -> tensor<1x64xf32>
// CHECK: %[[OP3:.*]] = "ttir.exp"(%[[OP2]]) : (tensor<1x64xf32>) -> tensor<1x64xf32>
// CHECK: %[[OP4:.*]] = "ttir.log"(%[[OP3]]) : (tensor<1x64xf32>) -> tensor<1x64xf32>
// CHECK: %[[RESHAPE_OUT:.*]] = "ttir.reshape"(%[[OP4]]) <{shape = [64 : i32]}> : (tensor<1x64xf32>) -> tensor<64xf32>
// CHECK: return %[[RESHAPE_OUT]] : tensor<64xf32>
func.func @long_chain_1d(%arg0: tensor<64xf32>) -> tensor<64xf32> {
  %0 = "ttir.abs"(%arg0) : (tensor<64xf32>) -> tensor<64xf32>
  %1 = "ttir.neg"(%0) : (tensor<64xf32>) -> tensor<64xf32>
  %2 = "ttir.exp"(%1) : (tensor<64xf32>) -> tensor<64xf32>
  %3 = "ttir.log"(%2) : (tensor<64xf32>) -> tensor<64xf32>
  return %3 : tensor<64xf32>
}

// =============================================================================
// Test 4: Function with no 1D tensors - should be unchanged
// =============================================================================

// CHECK-LABEL: func.func @no_1d_tensors
// CHECK-SAME: (%arg0: tensor<32x64xf32>) -> tensor<32x64xf32>
// CHECK-NOT: ttir.reshape
// CHECK: %[[ABS:.*]] = "ttir.abs"(%arg0) : (tensor<32x64xf32>) -> tensor<32x64xf32>
// CHECK: return %[[ABS]] : tensor<32x64xf32>
func.func @no_1d_tensors(%arg0: tensor<32x64xf32>) -> tensor<32x64xf32> {
  %0 = "ttir.abs"(%arg0) : (tensor<32x64xf32>) -> tensor<32x64xf32>
  return %0 : tensor<32x64xf32>
}

// =============================================================================
// Test 5: Multiple 1D inputs
// =============================================================================

// CHECK-LABEL: func.func @multiple_1d_inputs
// CHECK-SAME: (%arg0: tensor<128xf32>, %arg1: tensor<128xf32>) -> tensor<128xf32>
// CHECK-DAG: %[[RESHAPE_A:.*]] = "ttir.reshape"(%arg0) <{shape = [1 : i32, 128 : i32]}> : (tensor<128xf32>) -> tensor<1x128xf32>
// CHECK-DAG: %[[RESHAPE_B:.*]] = "ttir.reshape"(%arg1) <{shape = [1 : i32, 128 : i32]}> : (tensor<128xf32>) -> tensor<1x128xf32>
// CHECK: %[[ADD:.*]] = "ttir.add"(%[[RESHAPE_A]], %[[RESHAPE_B]]) : (tensor<1x128xf32>, tensor<1x128xf32>) -> tensor<1x128xf32>
// CHECK: %[[RESHAPE_OUT:.*]] = "ttir.reshape"(%[[ADD]]) <{shape = [128 : i32]}> : (tensor<1x128xf32>) -> tensor<128xf32>
// CHECK: return %[[RESHAPE_OUT]] : tensor<128xf32>
func.func @multiple_1d_inputs(%arg0: tensor<128xf32>, %arg1: tensor<128xf32>) -> tensor<128xf32> {
  %0 = "ttir.add"(%arg0, %arg1) : (tensor<128xf32>, tensor<128xf32>) -> tensor<128xf32>
  return %0 : tensor<128xf32>
}

// =============================================================================
// Test 6: Mixed 1D and 2D inputs
// =============================================================================

// CHECK-LABEL: func.func @mixed_1d_and_2d
// CHECK-SAME: (%arg0: tensor<128xf32>, %arg1: tensor<32x64xf32>) -> tensor<128xf32>
// CHECK: %[[RESHAPE_1D:.*]] = "ttir.reshape"(%arg0) <{shape = [1 : i32, 128 : i32]}> : (tensor<128xf32>) -> tensor<1x128xf32>
// CHECK: %[[ABS:.*]] = "ttir.abs"(%[[RESHAPE_1D]]) : (tensor<1x128xf32>) -> tensor<1x128xf32>
// CHECK: %[[RESHAPE_OUT:.*]] = "ttir.reshape"(%[[ABS]]) <{shape = [128 : i32]}> : (tensor<1x128xf32>) -> tensor<128xf32>
// CHECK: return %[[RESHAPE_OUT]] : tensor<128xf32>
func.func @mixed_1d_and_2d(%arg0: tensor<128xf32>, %arg1: tensor<32x64xf32>) -> tensor<128xf32> {
  // Only use the 1D input - 2D input is just to test mixed args
  %0 = "ttir.abs"(%arg0) : (tensor<128xf32>) -> tensor<128xf32>
  return %0 : tensor<128xf32>
}

// =============================================================================
// Test 7: Input already used by reshape - should skip inserting promotion
// =============================================================================

// CHECK-LABEL: func.func @input_already_reshaped
// CHECK-SAME: (%arg0: tensor<128xf32>) -> tensor<1x128xf32>
// The pass should NOT insert an additional reshape since all uses are reshapes
// CHECK-NOT: "ttir.reshape"(%arg0) <{shape = [1 : i32, 128 : i32]}>
// CHECK: %[[EXISTING_RESHAPE:.*]] = "ttir.reshape"(%arg0) <{shape = [1 : i32, 128 : i32]}> : (tensor<128xf32>) -> tensor<1x128xf32>
// CHECK: %[[ABS:.*]] = "ttir.abs"(%[[EXISTING_RESHAPE]]) : (tensor<1x128xf32>) -> tensor<1x128xf32>
// CHECK: return %[[ABS]] : tensor<1x128xf32>
func.func @input_already_reshaped(%arg0: tensor<128xf32>) -> tensor<1x128xf32> {
  %0 = "ttir.reshape"(%arg0) <{shape = [1 : i32, 128 : i32]}> : (tensor<128xf32>) -> tensor<1x128xf32>
  %1 = "ttir.abs"(%0) : (tensor<1x128xf32>) -> tensor<1x128xf32>
  return %1 : tensor<1x128xf32>
}

// =============================================================================
// Test 8: 1D output only (input is 2D)
// =============================================================================

// CHECK-LABEL: func.func @output_1d_only
// CHECK-SAME: (%arg0: tensor<1x128xf32>) -> tensor<128xf32>
// CHECK-NOT: "ttir.reshape"(%arg0)
// CHECK: %[[ABS:.*]] = "ttir.abs"(%arg0) : (tensor<1x128xf32>) -> tensor<1x128xf32>
// CHECK: %[[RESHAPE_OUT:.*]] = "ttir.reshape"(%[[ABS]]) <{shape = [128 : i32]}> : (tensor<1x128xf32>) -> tensor<128xf32>
// CHECK: return %[[RESHAPE_OUT]] : tensor<128xf32>
func.func @output_1d_only(%arg0: tensor<1x128xf32>) -> tensor<128xf32> {
  %0 = "ttir.abs"(%arg0) : (tensor<1x128xf32>) -> tensor<1x128xf32>
  %1 = "ttir.reshape"(%0) <{shape = [128 : i32]}> : (tensor<1x128xf32>) -> tensor<128xf32>
  return %1 : tensor<128xf32>
}

// =============================================================================
// Test 9: Different data types
// =============================================================================

// CHECK-LABEL: func.func @different_dtype_bf16
// CHECK-SAME: (%arg0: tensor<256xbf16>) -> tensor<256xbf16>
// CHECK: %[[RESHAPE_IN:.*]] = "ttir.reshape"(%arg0) <{shape = [1 : i32, 256 : i32]}> : (tensor<256xbf16>) -> tensor<1x256xbf16>
// CHECK: %[[ABS:.*]] = "ttir.abs"(%[[RESHAPE_IN]]) : (tensor<1x256xbf16>) -> tensor<1x256xbf16>
// CHECK: %[[RESHAPE_OUT:.*]] = "ttir.reshape"(%[[ABS]]) <{shape = [256 : i32]}> : (tensor<1x256xbf16>) -> tensor<256xbf16>
// CHECK: return %[[RESHAPE_OUT]] : tensor<256xbf16>
func.func @different_dtype_bf16(%arg0: tensor<256xbf16>) -> tensor<256xbf16> {
  %0 = "ttir.abs"(%arg0) : (tensor<256xbf16>) -> tensor<256xbf16>
  return %0 : tensor<256xbf16>
}

// CHECK-LABEL: func.func @different_dtype_i32
// CHECK-SAME: (%arg0: tensor<512xi32>) -> tensor<512xi32>
// CHECK: %[[RESHAPE_IN:.*]] = "ttir.reshape"(%arg0) <{shape = [1 : i32, 512 : i32]}> : (tensor<512xi32>) -> tensor<1x512xi32>
// CHECK: %[[ABS:.*]] = "ttir.abs"(%[[RESHAPE_IN]]) : (tensor<1x512xi32>) -> tensor<1x512xi32>
// CHECK: %[[RESHAPE_OUT:.*]] = "ttir.reshape"(%[[ABS]]) <{shape = [512 : i32]}> : (tensor<1x512xi32>) -> tensor<512xi32>
// CHECK: return %[[RESHAPE_OUT]] : tensor<512xi32>
func.func @different_dtype_i32(%arg0: tensor<512xi32>) -> tensor<512xi32> {
  %0 = "ttir.abs"(%arg0) : (tensor<512xi32>) -> tensor<512xi32>
  return %0 : tensor<512xi32>
}

// =============================================================================
// Test 10: Small tensor size
// =============================================================================

// CHECK-LABEL: func.func @small_tensor
// CHECK-SAME: (%arg0: tensor<1xf32>) -> tensor<1xf32>
// CHECK: %[[RESHAPE_IN:.*]] = "ttir.reshape"(%arg0) <{shape = [1 : i32, 1 : i32]}> : (tensor<1xf32>) -> tensor<1x1xf32>
// CHECK: %[[ABS:.*]] = "ttir.abs"(%[[RESHAPE_IN]]) : (tensor<1x1xf32>) -> tensor<1x1xf32>
// CHECK: %[[RESHAPE_OUT:.*]] = "ttir.reshape"(%[[ABS]]) <{shape = [1 : i32]}> : (tensor<1x1xf32>) -> tensor<1xf32>
// CHECK: return %[[RESHAPE_OUT]] : tensor<1xf32>
func.func @small_tensor(%arg0: tensor<1xf32>) -> tensor<1xf32> {
  %0 = "ttir.abs"(%arg0) : (tensor<1xf32>) -> tensor<1xf32>
  return %0 : tensor<1xf32>
}

// =============================================================================
// Test 11: Multiple return values with 1D
// =============================================================================

// CHECK-LABEL: func.func @multiple_returns
// CHECK-SAME: (%arg0: tensor<128xf32>) -> (tensor<128xf32>, tensor<128xf32>)
// CHECK: %[[RESHAPE_IN:.*]] = "ttir.reshape"(%arg0) <{shape = [1 : i32, 128 : i32]}> : (tensor<128xf32>) -> tensor<1x128xf32>
// CHECK: %[[ABS:.*]] = "ttir.abs"(%[[RESHAPE_IN]]) : (tensor<1x128xf32>) -> tensor<1x128xf32>
// CHECK: %[[NEG:.*]] = "ttir.neg"(%[[RESHAPE_IN]]) : (tensor<1x128xf32>) -> tensor<1x128xf32>
// CHECK-DAG: %[[RESHAPE_OUT1:.*]] = "ttir.reshape"(%[[ABS]]) <{shape = [128 : i32]}> : (tensor<1x128xf32>) -> tensor<128xf32>
// CHECK-DAG: %[[RESHAPE_OUT2:.*]] = "ttir.reshape"(%[[NEG]]) <{shape = [128 : i32]}> : (tensor<1x128xf32>) -> tensor<128xf32>
// CHECK: return %[[RESHAPE_OUT1]], %[[RESHAPE_OUT2]] : tensor<128xf32>, tensor<128xf32>
func.func @multiple_returns(%arg0: tensor<128xf32>) -> (tensor<128xf32>, tensor<128xf32>) {
  %0 = "ttir.abs"(%arg0) : (tensor<128xf32>) -> tensor<128xf32>
  %1 = "ttir.neg"(%arg0) : (tensor<128xf32>) -> tensor<128xf32>
  return %0, %1 : tensor<128xf32>, tensor<128xf32>
}

// =============================================================================
// Test 12: Higher rank tensors (3D+) should be unchanged
// =============================================================================

// CHECK-LABEL: func.func @higher_rank_unchanged
// CHECK-SAME: (%arg0: tensor<2x32x64xf32>) -> tensor<2x32x64xf32>
// CHECK-NOT: ttir.reshape
// CHECK: %[[ABS:.*]] = "ttir.abs"(%arg0) : (tensor<2x32x64xf32>) -> tensor<2x32x64xf32>
// CHECK: return %[[ABS]] : tensor<2x32x64xf32>
func.func @higher_rank_unchanged(%arg0: tensor<2x32x64xf32>) -> tensor<2x32x64xf32> {
  %0 = "ttir.abs"(%arg0) : (tensor<2x32x64xf32>) -> tensor<2x32x64xf32>
  return %0 : tensor<2x32x64xf32>
}

// =============================================================================
// Test 13: Multiple functions in module
// =============================================================================

// CHECK-LABEL: func.func @multi_func_first
// CHECK: "ttir.reshape"(%arg0) <{shape = [1 : i32, 32 : i32]}>
func.func @multi_func_first(%arg0: tensor<32xf32>) -> tensor<32xf32> {
  %0 = "ttir.abs"(%arg0) : (tensor<32xf32>) -> tensor<32xf32>
  return %0 : tensor<32xf32>
}

// CHECK-LABEL: func.func @multi_func_second
// CHECK: "ttir.reshape"(%arg0) <{shape = [1 : i32, 64 : i32]}>
func.func @multi_func_second(%arg0: tensor<64xf32>) -> tensor<64xf32> {
  %0 = "ttir.neg"(%arg0) : (tensor<64xf32>) -> tensor<64xf32>
  return %0 : tensor<64xf32>
}
