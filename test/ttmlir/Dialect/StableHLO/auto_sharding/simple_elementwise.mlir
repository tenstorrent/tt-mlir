// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// REQUIRES: stablehlo
// RUN: ttmlir-opt --stablehlo-pipeline="mesh-shape=1,2 automatic-arg-analysis=true enable-auto-sharding=true" -o %t.mlir %s
// RUN: FileCheck %s --input-file=%t.mlir

func.func @main(%arg0: tensor<2048x1024xf32>) -> tensor<2048x1024xf32> {
  %0 = stablehlo.abs %arg0 : tensor<2048x1024xf32>
  return %0 : tensor<2048x1024xf32>
}

// Elementwise ops don't introduce CCL overhead for any sharding, so
// auto-sharding should pick a sharded config for memory benefit.
// CHECK: sdy.manual_computation(%arg0)
// CHECK-SAME: in_shardings=[<@mesh, [{"batch"}, {}]>]
// CHECK-SAME: out_shardings=[<@mesh, [{"batch"}, {}]>]
// CHECK: stablehlo.abs %arg1 : tensor<1024x1024xf32>
// CHECK: sdy.return %{{.*}} : tensor<1024x1024xf32>
