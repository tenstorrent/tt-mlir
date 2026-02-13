// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// RUN: ttmlir-opt --cpu-hoist-const-eval --canonicalize -o %t %s
// RUN: FileCheck %s --input-file=%t

// Test const-eval CPU hoisting in isolation using the dedicated pass.
// This tests the CPUHoistConstEvalTransform pass directly, without running
// the full TTNN pipeline.

// ============================================================================
// Test 1: Basic const-eval hoisting.
// All non-creation ops in a const-eval function should be hoisted.
// ============================================================================

// CHECK: ttcore.device_module {
// CHECK: module {

// CHECK-LABEL: func.func private @basic_const_eval
// CHECK-SAME: attributes {tt.function_type = "const_eval"}
// CHECK: ttir.to_layout
// CHECK: ttir.to_layout
// CHECK: call @cpu_hoisted_const_eval_{{.*}}
// CHECK: ttir.to_layout
// CHECK: return

module {
  ttcore.device_module {
    module {
      func.func private @basic_const_eval(
          %arg0: tensor<32x32xbf16>, %arg1: tensor<32x32xbf16>
      ) -> tensor<32x32xbf16> attributes {tt.function_type = "const_eval"} {
        %0 = "ttir.add"(%arg0, %arg1) : (tensor<32x32xbf16>, tensor<32x32xbf16>) -> tensor<32x32xbf16>
        return %0 : tensor<32x32xbf16>
      }

// ============================================================================
// Test 2: Creation op between hoisted ops (bug fix validation).
// A creation op (zeros) is returned from the const-eval function and is
// sandwiched between hoisted ops. Before the fix, this would cause an SSA
// dominance violation because the ToLayout conversion for %zeros would be
// placed before %zeros was defined.
// ============================================================================

// CHECK-LABEL: func.func private @creation_op_between_hoisted_ops
// CHECK-SAME: attributes {tt.function_type = "const_eval"}
// The zeros op should remain (not hoisted).
// CHECK: ttir.zeros
// ToLayout conversions and call should come after zeros.
// CHECK: ttir.to_layout
// CHECK: call @cpu_hoisted_const_eval_{{.*}}
// CHECK: return

      func.func private @creation_op_between_hoisted_ops(
          %arg0: tensor<32x32xbf16>
      ) -> (tensor<32x32xbf16>, tensor<32x32xbf16>) attributes {tt.function_type = "const_eval"} {
        %add = "ttir.add"(%arg0, %arg0) : (tensor<32x32xbf16>, tensor<32x32xbf16>) -> tensor<32x32xbf16>
        %zeros = "ttir.zeros"() <{shape = array<i32: 32, 32>}> : () -> tensor<32x32xbf16>
        %mul = "ttir.multiply"(%add, %zeros) : (tensor<32x32xbf16>, tensor<32x32xbf16>) -> tensor<32x32xbf16>
        return %mul, %zeros : tensor<32x32xbf16>, tensor<32x32xbf16>
      }

// ============================================================================
// Test 3: Multiple creation ops interleaved with hoisted ops.
// Multiple non-hoisted creation ops are between hoisted ops, all used as
// inputs to hoisted ops. Tests that the fix handles multiple interleaved
// non-hoisted values correctly.
// ============================================================================

// CHECK-LABEL: func.func private @multiple_creation_ops_interleaved
// CHECK-SAME: attributes {tt.function_type = "const_eval"}
// Both creation ops should remain.
// CHECK: ttir.zeros
// CHECK: ttir.ones
// Conversions and call should follow.
// CHECK: ttir.to_layout
// CHECK: call @cpu_hoisted_const_eval_{{.*}}
// CHECK: return

      func.func private @multiple_creation_ops_interleaved(
          %arg0: tensor<32x32xbf16>
      ) -> (tensor<32x32xbf16>, tensor<32x32xbf16>, tensor<32x32xbf16>) attributes {tt.function_type = "const_eval"} {
        %add = "ttir.add"(%arg0, %arg0) : (tensor<32x32xbf16>, tensor<32x32xbf16>) -> tensor<32x32xbf16>
        %zeros = "ttir.zeros"() <{shape = array<i32: 32, 32>}> : () -> tensor<32x32xbf16>
        %mul = "ttir.multiply"(%add, %zeros) : (tensor<32x32xbf16>, tensor<32x32xbf16>) -> tensor<32x32xbf16>
        %ones = "ttir.ones"() <{shape = array<i32: 32, 32>}> : () -> tensor<32x32xbf16>
        %sub = "ttir.subtract"(%mul, %ones) : (tensor<32x32xbf16>, tensor<32x32xbf16>) -> tensor<32x32xbf16>
        return %sub, %zeros, %ones : tensor<32x32xbf16>, tensor<32x32xbf16>, tensor<32x32xbf16>
      }

// ============================================================================
// Test 4: Creation op at the end, not between hoisted ops.
// The creation op is after all hoisted ops and is not used by any hoisted op.
// This should work correctly with both old and new code.
// ============================================================================

// CHECK-LABEL: func.func private @creation_op_at_end
// CHECK-SAME: attributes {tt.function_type = "const_eval"}
// CHECK: ttir.to_layout
// CHECK: call @cpu_hoisted_const_eval_{{.*}}
// Zeros should still be present (returned, not hoisted).
// CHECK: ttir.zeros
// CHECK: return

      func.func private @creation_op_at_end(
          %arg0: tensor<32x32xbf16>, %arg1: tensor<32x32xbf16>
      ) -> (tensor<32x32xbf16>, tensor<32x32xbf16>) attributes {tt.function_type = "const_eval"} {
        %add = "ttir.add"(%arg0, %arg1) : (tensor<32x32xbf16>, tensor<32x32xbf16>) -> tensor<32x32xbf16>
        %zeros = "ttir.zeros"() <{shape = array<i32: 32, 32>}> : () -> tensor<32x32xbf16>
        return %add, %zeros : tensor<32x32xbf16>, tensor<32x32xbf16>
      }

// ============================================================================
// Test 5: All creation ops - nothing to hoist.
// When all ops in the const-eval function are creation ops (and returned),
// no hoisting should occur.
// ============================================================================

// CHECK-LABEL: func.func private @all_creation_ops
// CHECK-SAME: attributes {tt.function_type = "const_eval"}
// No call should be generated - all ops are creation ops.
// CHECK-NOT: call @cpu_hoisted
// CHECK: ttir.zeros
// CHECK: return

      func.func private @all_creation_ops() -> tensor<32x32xbf16> attributes {tt.function_type = "const_eval"} {
        %zeros = "ttir.zeros"() <{shape = array<i32: 32, 32>}> : () -> tensor<32x32xbf16>
        return %zeros : tensor<32x32xbf16>
      }

// ============================================================================
// Test 6: Creation op with transparent chain (reshape) leading to return.
// A creation op followed by a reshape, both returned - both should be skipped.
// ============================================================================

// CHECK-LABEL: func.func private @creation_with_transparent_chain
// CHECK-SAME: attributes {tt.function_type = "const_eval"}
// The add should be hoisted.
// CHECK: ttir.zeros
// CHECK: ttir.reshape
// CHECK: ttir.to_layout
// CHECK: call @cpu_hoisted_const_eval_{{.*}}
// CHECK: return

      func.func private @creation_with_transparent_chain(
          %arg0: tensor<32x32xbf16>
      ) -> (tensor<32x32xbf16>, tensor<1x1024xbf16>) attributes {tt.function_type = "const_eval"} {
        %add = "ttir.add"(%arg0, %arg0) : (tensor<32x32xbf16>, tensor<32x32xbf16>) -> tensor<32x32xbf16>
        %zeros = "ttir.zeros"() <{shape = array<i32: 32, 32>}> : () -> tensor<32x32xbf16>
        %reshaped = "ttir.reshape"(%zeros) <{shape = array<i32: 1, 1024>}> : (tensor<32x32xbf16>) -> tensor<1x1024xbf16>
        return %add, %reshaped : tensor<32x32xbf16>, tensor<1x1024xbf16>
      }

// ============================================================================
// Test 7: Non-const-eval function should not be affected.
// Functions without the const_eval function type should be left unchanged.
// ============================================================================

// CHECK-LABEL: func.func @non_const_eval_func
// CHECK-NOT: call @cpu_hoisted
// CHECK: ttir.add
// CHECK: return

      func.func @non_const_eval_func(
          %arg0: tensor<32x32xbf16>, %arg1: tensor<32x32xbf16>
      ) -> tensor<32x32xbf16> {
        %0 = "ttir.add"(%arg0, %arg1) : (tensor<32x32xbf16>, tensor<32x32xbf16>) -> tensor<32x32xbf16>
        return %0 : tensor<32x32xbf16>
      }

// ============================================================================
// Test 8: Const-eval with f32 types (no conversion needed).
// When input types are already f32, no ToLayout conversion should be needed.
// ============================================================================

// CHECK-LABEL: func.func private @const_eval_f32
// CHECK-SAME: attributes {tt.function_type = "const_eval"}
// CHECK: call @cpu_hoisted_const_eval_{{.*}}
// CHECK-NOT: ttir.to_layout
// CHECK: return

      func.func private @const_eval_f32(
          %arg0: tensor<32x32xf32>, %arg1: tensor<32x32xf32>
      ) -> tensor<32x32xf32> attributes {tt.function_type = "const_eval"} {
        %0 = "ttir.add"(%arg0, %arg1) : (tensor<32x32xf32>, tensor<32x32xf32>) -> tensor<32x32xf32>
        return %0 : tensor<32x32xf32>
      }

// ============================================================================
// Test 9: Multiple outputs from const-eval hoisting.
// Verify that multiple output values are correctly handled.
// ============================================================================

// CHECK-LABEL: func.func private @const_eval_multiple_outputs
// CHECK-SAME: attributes {tt.function_type = "const_eval"}
// CHECK: call @cpu_hoisted_const_eval_{{.*}}
// CHECK: return

      func.func private @const_eval_multiple_outputs(
          %arg0: tensor<32x32xbf16>, %arg1: tensor<32x32xbf16>
      ) -> (tensor<32x32xbf16>, tensor<32x32xbf16>) attributes {tt.function_type = "const_eval"} {
        %add = "ttir.add"(%arg0, %arg1) : (tensor<32x32xbf16>, tensor<32x32xbf16>) -> tensor<32x32xbf16>
        %sub = "ttir.subtract"(%arg0, %arg1) : (tensor<32x32xbf16>, tensor<32x32xbf16>) -> tensor<32x32xbf16>
        return %add, %sub : tensor<32x32xbf16>, tensor<32x32xbf16>
      }

    }
  }
}

// Verify CPU-hoisted function declarations exist in the device module.
// CHECK: func.func private @cpu_hoisted_const_eval_{{.*}}

// Verify the CPU module is created with hoisted function definitions.
// CHECK: ttcore.cpu_module {
// CHECK: module {
// CHECK: func.func @cpu_hoisted_const_eval_{{.*}}
