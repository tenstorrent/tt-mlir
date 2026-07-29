// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// RUN: ttmlir-opt --ttir-to-ttnn-common-pipeline="enable-cpu-hoisted-const-eval=false system-desc-path=%system_desc_path%" -o %t.mlir %s
//
// RUN: ttmlir-opt --ttnn-common-to-runtime-pipeline -o %t_rt.mlir %t.mlir
// RUN: ttmlir-translate --ttnn-to-flatbuffer -o %basename_t.ttnn %t_rt.mlir
//
// RUN: ttmlir-opt --ttnn-common-to-emitc-pipeline -o %t2.mlir %t.mlir
// RUN: ttmlir-translate --mlir-to-cpp -o %t.cpp %t2.mlir
// RUN: FileCheck %s --input-file=%t.cpp

// The DiT adaLN gated-residual epilogue (matmul + multiply + add) fuses at the
// TTNN level and lowers to a single
// ttnn::experimental::dit_minimal_matmul_addcmul_fused call.
func.func @dit_matmul_addcmul(%a: tensor<32x128xbf16>, %b: tensor<128x256xbf16>,
                              %gate: tensor<32x256xbf16>, %res: tensor<32x256xbf16>)
    -> tensor<32x256xbf16> {
  // CHECK: ttnn::experimental::dit_minimal_matmul_addcmul_fused
  %0 = "ttir.matmul"(%a, %b) : (tensor<32x128xbf16>, tensor<128x256xbf16>) -> tensor<32x256xbf16>
  %1 = "ttir.multiply"(%0, %gate) : (tensor<32x256xbf16>, tensor<32x256xbf16>) -> tensor<32x256xbf16>
  %2 = "ttir.add"(%1, %res) : (tensor<32x256xbf16>, tensor<32x256xbf16>) -> tensor<32x256xbf16>
  return %2 : tensor<32x256xbf16>
}
