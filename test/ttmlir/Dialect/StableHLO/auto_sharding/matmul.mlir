// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// REQUIRES: stablehlo
// RUN: ttmlir-opt --stablehlo-pipeline="mesh-shape=1,2 automatic-arg-analysis=true enable-auto-sharding=true" -o %t.mlir %s
// RUN: FileCheck %s --input-file=%t.mlir

func.func @main(%arg0: tensor<2048x512xf32>, %arg1: tensor<512x512xf32>) -> tensor<2048x512xf32> {
  %0 = stablehlo.dot_general %arg0, %arg1, contracting_dims = [1] x [0] : (tensor<2048x512xf32>, tensor<512x512xf32>) -> tensor<2048x512xf32>
  return %0 : tensor<2048x512xf32>
}

// Auto-sharding should shard arg0 on the non-contracting (M) dimension
// to avoid introducing CCL overhead from contracting-dim sharding.
// arg1 should remain replicated since its non-contracting dim (N=512)
// has a smaller memory footprint than arg0's M dimension.
// CHECK: sdy.manual_computation(%arg0, %arg1)
// CHECK-SAME: in_shardings=[<@mesh, [{"batch"}, {}]>, <@mesh, [{}, {}]>]
// CHECK-SAME: out_shardings=[<@mesh, [{"batch"}, {}]>]
// CHECK: stablehlo.dot_general %arg2, %arg3, contracting_dims = [1] x [0] : (tensor<1024x512xf32>, tensor<512x512xf32>) -> tensor<1024x512xf32>
