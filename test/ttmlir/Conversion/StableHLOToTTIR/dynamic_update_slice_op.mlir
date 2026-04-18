// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// REQUIRES: stablehlo
// RUN: ttmlir-opt --stablehlo-to-ttir-pipeline -o %t %s
// RUN: FileCheck %s --input-file=%t

module @jit_dynamic_update_slice attributes {} {
  func.func public @test_dynamic_update_slice_3d(%operand: tensor<1x1370x1280xbf16>, %update: tensor<1x1x1280xbf16>, %start0: tensor<i64>, %start1: tensor<i64>, %start2: tensor<i64>) -> tensor<1x1370x1280xbf16> {
    %result = "stablehlo.dynamic_update_slice"(%operand, %update, %start0, %start1, %start2) : (tensor<1x1370x1280xbf16>, tensor<1x1x1280xbf16>, tensor<i64>, tensor<i64>, tensor<i64>) -> tensor<1x1370x1280xbf16>
    return %result : tensor<1x1370x1280xbf16>
  }
}

// CHECK-LABEL: func.func public @test_dynamic_update_slice_3d
// CHECK: ttir.reshape
// CHECK: ttir.concat
// CHECK: ttir.constant
// CHECK: ttir.constant
// CHECK: ttir.clamp_tensor
// CHECK: ttir.constant
// CHECK: ttir.add
// CHECK: ttir.slice_write
