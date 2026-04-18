// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// RUN: ttmlir-opt --ttir-to-emitpy-pipeline -o %t.mlir %s
// RUN: ttmlir-translate --mlir-to-python -o %t.py %t.mlir
// RUN: FileCheck %s --input-file=%t.py

// Verify that the EmitPyFormExpressions pass produces compact Python output
// by inlining single-use PyExpressionInterface ops into ExpressionOps:
//
// 1. Dict GET: string constant is inlined into subscript — dict["key"]
// 2. List wrapping: util_create_list is inlined — return [x]
// 3. Dict SET: string key is inlined into subscript assignment — dict["key"] = val

// forward() — dict GET for weight access and const-eval cache lookup.
//
// CHECK-LABEL: def forward(
// CHECK:   weights["weight_0"]
// CHECK:   ce_cache_forward["forward_const_eval_0"]
// CHECK-NOT: const{{.*}} = "forward_const_eval_0"
// CHECK:   return [{{.*}}]

// consteval_forward() — dict GET for weight keys, dict SET for cache entry.
//
// CHECK-LABEL: def consteval_forward(ce_cache, weights
// CHECK:   if not ce_cache:
// CHECK:     forward_const_eval_0([weights["weight_0"], weights["weight_1"]])
// CHECK:     ce_cache["forward_const_eval_0"] =
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
