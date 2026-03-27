// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// RUN: ttmlir-opt --ttir-to-emitpy-pipeline -o %t.mlir %s
// RUN: ttmlir-translate --mlir-to-python -o %t.py %t.mlir
// RUN: FileCheck %s --input-file=%t.py

// Verify that the EmitPyFormExpressions pass produces compact Python output:
// 1. Dict GET: string constant is inlined into subscript: dict["key"]
// 2. List-into-call: util_create_list is inlined into callee: f([x])
// 3. Dict SET: string key and list creation are inlined: dict["key"] = [v[0]]

// CHECK-LABEL: def forward(
// CHECK:   _cached_forward["forward_const_eval_0"]
// CHECK-NOT: const{{.*}} = "forward_const_eval_0"

// CHECK-LABEL: def consteval_forward(ce_cache, input_1):
// CHECK:   if not ce_cache:
// CHECK:     forward_const_eval_0_0 = forward_const_eval_0([
// CHECK:     ce_cache["forward_const_eval_0"] = [forward_const_eval_0_0[0]]
// CHECK:   return ce_cache

module {
  func.func @forward(%arg0: tensor<32x32xbf16> {ttcore.argument_type = #ttcore.argument_type<input>},
                     %arg1: tensor<32x32xbf16> {ttcore.argument_type = #ttcore.argument_type<parameter>},
                     %arg2: tensor<32x32xbf16> {ttcore.argument_type = #ttcore.argument_type<constant>}) -> tensor<32x32xbf16> {
    %0 = "ttir.add"(%arg0, %arg1) : (tensor<32x32xbf16>, tensor<32x32xbf16>) -> tensor<32x32xbf16>
    %1 = "ttir.add"(%arg1, %arg2) : (tensor<32x32xbf16>, tensor<32x32xbf16>) -> tensor<32x32xbf16>
    %2 = "ttir.add"(%0, %1) : (tensor<32x32xbf16>, tensor<32x32xbf16>) -> tensor<32x32xbf16>
    return %2 : tensor<32x32xbf16>
  }
}
