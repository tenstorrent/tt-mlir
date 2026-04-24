// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// REQUIRES: stablehlo
// RUN: ttmlir-opt --stablehlo-to-ttir-pipeline --ttir-to-ttnn-backend-pipeline="enable-cpu-hoisted-const-eval=false system-desc-path=%system_desc_path%" -o %t.mlir %s
// RUN: ttmlir-translate --ttnn-to-flatbuffer -o %basename_t.ttnn %t.mlir
// RUN: ttmlir-opt --ttnn-to-emitc-device-pipeline -o %t2.mlir %t.mlir
// RUN: ttmlir-translate --mlir-to-cpp -o %basename_t.cpp %t2.mlir

func.func @dynamic_update_slice(%operand: tensor<1x32x64xbf16>, %update: tensor<1x1x64xbf16>, %start0: tensor<i64>, %start1: tensor<i64>, %start2: tensor<i64>) -> tensor<1x32x64xbf16> {
  %result = "stablehlo.dynamic_update_slice"(%operand, %update, %start0, %start1, %start2) : (tensor<1x32x64xbf16>, tensor<1x1x64xbf16>, tensor<i64>, tensor<i64>, tensor<i64>) -> tensor<1x32x64xbf16>
  return %result : tensor<1x32x64xbf16>
}
