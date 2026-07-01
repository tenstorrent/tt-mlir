// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// RUN: ttmlir-opt --ttir-to-ttnn-backend-pipeline="system-desc-path=%system_desc_path% enable-cpu-hoisted-const-eval=true" -o %t.mlir %s
// RUN: FileCheck %s --input-file=%t.mlir
// RUN: ttmlir-translate --ttnn-to-flatbuffer -o %t.ttnn %t.mlir

module {
  // CHECK: ttcore.device_module {

  // The transpose const-eval is hoisted to the CPU module preserving its native
  // bf16 dtype: the hoisted call takes/returns bf16 with no f32 typecast
  // round-trip. This produces a flatbuffer runnable on silicon for PCC.
  // CHECK-LABEL: func.func private @forward_const_eval_0
  // CHECK-NOT: "ttnn.typecast"
  // CHECK: call @cpu_hoisted_const_eval_{{.*}} : (tensor<32x64xbf16{{.*}}>) -> tensor<64x32xbf16

  // CHECK-LABEL: func.func @forward
  func.func @forward(%arg0: tensor<32x64xbf16> {ttcore.argument_type = #ttcore.argument_type<input>},
                     %arg1: tensor<32x64xbf16> {ttcore.argument_type = #ttcore.argument_type<parameter>}) -> tensor<32x32xbf16> {
    %0 = "ttir.permute"(%arg1) <{permutation = array<i64: 1, 0>}> : (tensor<32x64xbf16>) -> tensor<64x32xbf16>
    %1 = "ttir.matmul"(%arg0, %0) : (tensor<32x64xbf16>, tensor<64x32xbf16>) -> tensor<32x32xbf16>
    return %1 : tensor<32x32xbf16>
  }
}
