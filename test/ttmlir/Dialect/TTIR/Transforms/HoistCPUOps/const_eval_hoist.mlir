// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// RUN: ttmlir-opt --ttcore-wrap-device-module --cpu-hoist-const-eval --canonicalize -o %t %s
// RUN: FileCheck %s --input-file=%t

// CHECK: ttcore.device_module {
// CHECK: builtin.module {

// --- Test 1: Basic const-eval hoisting ---

// CHECK-LABEL: func.func private @basic_const_eval
// CHECK: ttir.to_layout
// CHECK: ttir.to_layout
// CHECK: call @cpu_hoisted_const_eval_{{.*}}
// CHECK: ttir.to_layout
// CHECK: return
func.func private @basic_const_eval(
    %arg0: tensor<32x32xbf16>, %arg1: tensor<32x32xbf16>
) -> tensor<32x32xbf16> attributes {tt.function_type = "const_eval"} {
  %0 = "ttir.add"(%arg0, %arg1) : (tensor<32x32xbf16>, tensor<32x32xbf16>) -> tensor<32x32xbf16>
  return %0 : tensor<32x32xbf16>
}

// --- Test 2: Creation op between hoisted ops ---
// Zeros is returned and sandwiched between hoisted ops. It should not be hoisted.

// CHECK-LABEL: func.func private @creation_op_between_hoisted_ops
// CHECK: ttir.zeros
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

// --- Test 3: Multiple creation ops interleaved ---

// CHECK-LABEL: func.func private @multiple_creation_ops_interleaved
// CHECK: ttir.zeros
// CHECK: ttir.ones
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

// --- Test 4: Creation op at the end ---

// CHECK-LABEL: func.func private @creation_op_at_end
// CHECK: ttir.to_layout
// CHECK: call @cpu_hoisted_const_eval_{{.*}}
// CHECK: ttir.zeros
// CHECK: return
func.func private @creation_op_at_end(
    %arg0: tensor<32x32xbf16>, %arg1: tensor<32x32xbf16>
) -> (tensor<32x32xbf16>, tensor<32x32xbf16>) attributes {tt.function_type = "const_eval"} {
  %add = "ttir.add"(%arg0, %arg1) : (tensor<32x32xbf16>, tensor<32x32xbf16>) -> tensor<32x32xbf16>
  %zeros = "ttir.zeros"() <{shape = array<i32: 32, 32>}> : () -> tensor<32x32xbf16>
  return %add, %zeros : tensor<32x32xbf16>, tensor<32x32xbf16>
}

// --- Test 5: All creation ops - nothing to hoist ---

// CHECK-LABEL: func.func private @all_creation_ops
// CHECK-NOT: call @cpu_hoisted
// CHECK: ttir.zeros
// CHECK: return
func.func private @all_creation_ops() -> tensor<32x32xbf16> attributes {tt.function_type = "const_eval"} {
  %zeros = "ttir.zeros"() <{shape = array<i32: 32, 32>}> : () -> tensor<32x32xbf16>
  return %zeros : tensor<32x32xbf16>
}

// --- Test 6: Creation op with transparent chain (reshape) ---

// CHECK-LABEL: func.func private @creation_with_transparent_chain
// CHECK: ttir.to_layout
// CHECK: call @cpu_hoisted_const_eval_{{.*}}
// CHECK: ttir.zeros
// CHECK: ttir.reshape
// CHECK: return
func.func private @creation_with_transparent_chain(
    %arg0: tensor<32x32xbf16>
) -> (tensor<32x32xbf16>, tensor<1x1024xbf16>) attributes {tt.function_type = "const_eval"} {
  %add = "ttir.add"(%arg0, %arg0) : (tensor<32x32xbf16>, tensor<32x32xbf16>) -> tensor<32x32xbf16>
  %zeros = "ttir.zeros"() <{shape = array<i32: 32, 32>}> : () -> tensor<32x32xbf16>
  %reshaped = "ttir.reshape"(%zeros) <{shape = [1 : i32, 1024 : i32]}> : (tensor<32x32xbf16>) -> tensor<1x1024xbf16>
  return %add, %reshaped : tensor<32x32xbf16>, tensor<1x1024xbf16>
}

// --- Test 7: Non-const-eval function is not affected ---

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

// --- Test 8: f32 types need no conversion ---

// CHECK-LABEL: func.func private @const_eval_f32
// CHECK: call @cpu_hoisted_const_eval_{{.*}}
// CHECK-NOT: ttir.to_layout
// CHECK: return
func.func private @const_eval_f32(
    %arg0: tensor<32x32xf32>, %arg1: tensor<32x32xf32>
) -> tensor<32x32xf32> attributes {tt.function_type = "const_eval"} {
  %0 = "ttir.add"(%arg0, %arg1) : (tensor<32x32xf32>, tensor<32x32xf32>) -> tensor<32x32xf32>
  return %0 : tensor<32x32xf32>
}

// --- Test 9: Multiple outputs ---

// CHECK-LABEL: func.func private @const_eval_multiple_outputs
// CHECK: call @cpu_hoisted_const_eval_{{.*}}
// CHECK: return
func.func private @const_eval_multiple_outputs(
    %arg0: tensor<32x32xbf16>, %arg1: tensor<32x32xbf16>
) -> (tensor<32x32xbf16>, tensor<32x32xbf16>) attributes {tt.function_type = "const_eval"} {
  %add = "ttir.add"(%arg0, %arg1) : (tensor<32x32xbf16>, tensor<32x32xbf16>) -> tensor<32x32xbf16>
  %sub = "ttir.subtract"(%arg0, %arg1) : (tensor<32x32xbf16>, tensor<32x32xbf16>) -> tensor<32x32xbf16>
  return %add, %sub : tensor<32x32xbf16>, tensor<32x32xbf16>
}

// Verify hoisted function declarations and definitions.
// CHECK: func.func private @cpu_hoisted_const_eval_{{.*}}
// CHECK: ttcore.cpu_module {
// CHECK: builtin.module {
// CHECK: func.func @cpu_hoisted_const_eval_{{.*}}
