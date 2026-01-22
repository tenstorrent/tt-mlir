// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// RUN: ttmlir-opt --ttir-to-emitpy-pipeline="enable-cpu-hoisted-const-eval=true" -o %t.mlir %s
// RUN: ttmlir-translate --mlir-to-python -o %t.py %t.mlir
// RUN: FileCheck %s --input-file=%t.py

// Basic const-eval: operation on parameters/constants should be hoisted.
// CHECK-LABEL: def forward(
// CHECK: execute_forward_consteval
// CHECK: "forward_const_eval_0"
func.func @forward(%arg0: tensor<32x32xf32> {ttcore.argument_type = #ttcore.argument_type<input>},
                   %arg1: tensor<32x32xf32> {ttcore.argument_type = #ttcore.argument_type<parameter>},
                   %arg2: tensor<32x32xf32> {ttcore.argument_type = #ttcore.argument_type<parameter>},
                   %arg3: tensor<32x32xf32> {ttcore.argument_type = #ttcore.argument_type<constant>}) -> tensor<32x32xf32> {
  %0 = "ttir.add"(%arg0, %arg1) : (tensor<32x32xf32>, tensor<32x32xf32>) -> tensor<32x32xf32>
  %1 = "ttir.subtract"(%arg2, %arg3) : (tensor<32x32xf32>, tensor<32x32xf32>) -> tensor<32x32xf32>
  %2 = "ttir.multiply"(%0, %1) : (tensor<32x32xf32>, tensor<32x32xf32>) -> tensor<32x32xf32>
  return %2 : tensor<32x32xf32>
}

// Test split const-eval: multiple independent const-eval subgraphs.
// CHECK-LABEL: def forward_split(
// CHECK: execute_forward_split_consteval
// CHECK: "forward_split_const_eval_0"
// CHECK: "forward_split_const_eval_1"
func.func @forward_split(%arg0: tensor<32x32xf32> {ttcore.argument_type = #ttcore.argument_type<input>},
                         %arg1: tensor<32x32xf32> {ttcore.argument_type = #ttcore.argument_type<parameter>},
                         %arg2: tensor<32x32xf32> {ttcore.argument_type = #ttcore.argument_type<parameter>},
                         %arg3: tensor<32x32xf32> {ttcore.argument_type = #ttcore.argument_type<constant>}) -> tensor<32x32xf32> {
  %0 = "ttir.add"(%arg0, %arg1) : (tensor<32x32xf32>, tensor<32x32xf32>) -> tensor<32x32xf32>
  %1 = "ttir.add"(%arg1, %arg2) : (tensor<32x32xf32>, tensor<32x32xf32>) -> tensor<32x32xf32>
  %2 = "ttir.add"(%arg2, %arg3) : (tensor<32x32xf32>, tensor<32x32xf32>) -> tensor<32x32xf32>
  %3 = "ttir.subtract"(%0, %1) : (tensor<32x32xf32>, tensor<32x32xf32>) -> tensor<32x32xf32>
  %4 = "ttir.multiply"(%2, %3) : (tensor<32x32xf32>, tensor<32x32xf32>) -> tensor<32x32xf32>
  return %4 : tensor<32x32xf32>
}

// Test merged const-eval: connected const-eval ops should be merged.
// CHECK-LABEL: def forward_merge(
// CHECK: execute_forward_merge_consteval
// CHECK: "forward_merge_const_eval_0"
func.func @forward_merge(%arg0: tensor<32x32xf32> {ttcore.argument_type = #ttcore.argument_type<input>},
                         %arg1: tensor<32x32xf32> {ttcore.argument_type = #ttcore.argument_type<parameter>},
                         %arg2: tensor<32x32xf32> {ttcore.argument_type = #ttcore.argument_type<parameter>},
                         %arg3: tensor<32x32xf32> {ttcore.argument_type = #ttcore.argument_type<constant>}) -> tensor<32x32xf32> {
  %0 = "ttir.add"(%arg0, %arg1) : (tensor<32x32xf32>, tensor<32x32xf32>) -> tensor<32x32xf32>
  %1 = "ttir.add"(%arg1, %arg2) : (tensor<32x32xf32>, tensor<32x32xf32>) -> tensor<32x32xf32>
  %2 = "ttir.add"(%arg2, %arg3) : (tensor<32x32xf32>, tensor<32x32xf32>) -> tensor<32x32xf32>
  %3 = "ttir.subtract"(%1, %2) : (tensor<32x32xf32>, tensor<32x32xf32>) -> tensor<32x32xf32>
  %4 = "ttir.multiply"(%0, %3) : (tensor<32x32xf32>, tensor<32x32xf32>) -> tensor<32x32xf32>
  return %4 : tensor<32x32xf32>
}

// Test const-eval with creation ops (zeros).
// CHECK-LABEL: def forward_zeros(
// CHECK: execute_forward_zeros_consteval
// CHECK: "forward_zeros_const_eval_0"
func.func @forward_zeros(%arg0: tensor<32x32xf32> {ttcore.argument_type = #ttcore.argument_type<input>},
                         %arg1: tensor<32x32xf32> {ttcore.argument_type = #ttcore.argument_type<parameter>}) -> tensor<32x32xf32> {
  %0 = "ttir.zeros"() <{shape = array<i32:32, 32>}> : () -> tensor<32x32xf32>
  %1 = "ttir.add"(%arg0, %0) : (tensor<32x32xf32>, tensor<32x32xf32>) -> tensor<32x32xf32>
  %2 = "ttir.add"(%arg1, %0) : (tensor<32x32xf32>, tensor<32x32xf32>) -> tensor<32x32xf32>
  %3 = "ttir.multiply"(%1, %2) : (tensor<32x32xf32>, tensor<32x32xf32>) -> tensor<32x32xf32>
  return %3 : tensor<32x32xf32>
}

// Test all-constant function.
// CHECK-LABEL: def forward_all_const(
// CHECK: execute_forward_all_const_consteval
// CHECK: "forward_all_const_const_eval_0"
func.func @forward_all_const(%arg0: tensor<32x32xf32> {ttcore.argument_type = #ttcore.argument_type<constant>},
                             %arg1: tensor<32x32xf32> {ttcore.argument_type = #ttcore.argument_type<constant>}) -> tensor<32x32xf32> {
  %0 = "ttir.add"(%arg0, %arg1) : (tensor<32x32xf32>, tensor<32x32xf32>) -> tensor<32x32xf32>
  return %0 : tensor<32x32xf32>
}

// Verify that CPU-hoisted functions are generated with golden_function calls.
// CHECK-LABEL: def hoisted_forward_const_eval_0(
// CHECK: golden_function
// CHECK-LABEL: def hoisted_forward_split_const_eval_0(
// CHECK: golden_function
// CHECK-LABEL: def hoisted_forward_split_const_eval_1(
// CHECK: golden_function
// CHECK-LABEL: def hoisted_forward_merge_const_eval_0(
// CHECK: golden_function
// CHECK-LABEL: def hoisted_forward_zeros_const_eval_0(
// CHECK: golden_function
// CHECK-LABEL: def hoisted_forward_all_const_const_eval_0(
// CHECK: golden_function

// CHECK-LABEL: def forward_const_eval_0(
// CHECK: hoisted_forward_const_eval_0(
// CHECK-LABEL: def forward_split_const_eval_0(
// CHECK: hoisted_forward_split_const_eval_0(
// CHECK-LABEL: def forward_split_const_eval_1(
// CHECK: hoisted_forward_split_const_eval_1(
// CHECK-LABEL: def forward_merge_const_eval_0(
// CHECK: hoisted_forward_merge_const_eval_0(
// CHECK-LABEL: def forward_zeros_const_eval_0(
// CHECK: hoisted_forward_zeros_const_eval_0(
// CHECK-LABEL: def forward_all_const_const_eval_0(
// CHECK: hoisted_forward_all_const_const_eval_0(

// CHECK-LABEL: def execute_forward_consteval
// CHECK-LABEL: def execute_forward_split_consteval
// CHECK-LABEL: def execute_forward_merge_consteval
// CHECK-LABEL: def execute_forward_zeros_consteval
// CHECK-LABEL: def execute_forward_all_const_consteval
