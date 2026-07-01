// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// A pure data-movement const-eval subgraph (e.g. a weight transpose) is still
// CPU-hoisted, but preserving its native dtype: no bf16->f32->bf16 typecast
// round-trip, since a permutation is exact regardless of element type. A
// subgraph that performs arithmetic is hoisted on the f32 path as before.

// RUN: ttmlir-opt --ttir-to-ttnn-backend-pipeline="enable-cpu-hoisted-const-eval=true" -o %t %s
// RUN: FileCheck %s --input-file=%t

// Both subgraphs are hoisted (two call sites across the module).
// RUN: FileCheck %s --input-file=%t --check-prefix=HOISTCOUNT

// Escape hatch: data-movement-const-eval-f32=true restores the legacy f32 path
// for the pure transpose too (its const-eval gains the typecast round-trip).
// RUN: ttmlir-opt --ttir-to-ttnn-backend-pipeline="enable-cpu-hoisted-const-eval=true data-movement-const-eval-f32=true" -o %t.legacy %s
// RUN: FileCheck %s --input-file=%t.legacy --check-prefix=LEGACY

module {
  // CHECK: ttcore.device_module {
  // CHECK: builtin.module

  // Pure transpose: hoisted to the CPU module in its native bf16 dtype - the
  // hoisted call takes/returns bf16 and there is no f32 typecast round-trip.
  // CHECK-LABEL: func.func private @forward_transpose_only_const_eval_0
  // CHECK-NOT: "ttnn.typecast"
  // CHECK: call @cpu_hoisted_const_eval_{{.*}} : (tensor<32x64xbf16{{.*}}>) -> tensor<64x32xbf16

  // LEGACY-LABEL: func.func private @forward_transpose_only_const_eval_0
  // LEGACY: "ttnn.typecast"{{.*}} -> tensor<32x64xf32
  // LEGACY: call @cpu_hoisted_const_eval_{{.*}} : (tensor<32x64xf32{{.*}}>) -> tensor<64x32xf32

  // CHECK-LABEL: func.func @forward_transpose_only
  func.func @forward_transpose_only(%arg0: tensor<32x64xbf16> {ttcore.argument_type = #ttcore.argument_type<input>},
                                    %arg1: tensor<32x64xbf16> {ttcore.argument_type = #ttcore.argument_type<parameter>}) -> tensor<32x32xbf16> {
    // The transpose of the weight %arg1 is const-eval'able but pure data movement.
    %0 = "ttir.permute"(%arg1) <{permutation = array<i64: 1, 0>}> : (tensor<32x64xbf16>) -> tensor<64x32xbf16>
    // %arg0 @ transpose(%arg1) : (32x64) @ (64x32) -> 32x32
    %1 = "ttir.matmul"(%arg0, %0) : (tensor<32x64xbf16>, tensor<64x32xbf16>) -> tensor<32x32xbf16>
    return %1 : tensor<32x32xbf16>
  }

  // Arithmetic const-eval: hoisted on the f32 path (typecasts), as before.
  // CHECK-LABEL: func.func private @forward_with_arith_const_eval_0
  // CHECK: "ttnn.typecast"
  // CHECK: call @cpu_hoisted_const_eval_

  // CHECK-LABEL: func.func @forward_with_arith
  func.func @forward_with_arith(%arg0: tensor<32x32xbf16> {ttcore.argument_type = #ttcore.argument_type<input>},
                                %arg1: tensor<32x32xbf16> {ttcore.argument_type = #ttcore.argument_type<parameter>},
                                %arg2: tensor<32x32xbf16> {ttcore.argument_type = #ttcore.argument_type<parameter>}) -> tensor<32x32xbf16> {
    %0 = "ttir.add"(%arg0, %arg1) : (tensor<32x32xbf16>, tensor<32x32xbf16>) -> tensor<32x32xbf16>
    %1 = "ttir.subtract"(%arg1, %arg2) : (tensor<32x32xbf16>, tensor<32x32xbf16>) -> tensor<32x32xbf16>
    %2 = "ttir.multiply"(%0, %1) : (tensor<32x32xbf16>, tensor<32x32xbf16>) -> tensor<32x32xbf16>
    return %2 : tensor<32x32xbf16>
  }

  // CHECK: ttcore.cpu_module {

  // Both subgraphs are hoisted: the bf16 transpose and the f32 arithmetic.
  // HOISTCOUNT-COUNT-2: ttir.cpu_hoist_call
  // HOISTCOUNT-NOT: ttir.cpu_hoist_call
}
