// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// REQUIRES: stablehlo
// RUN: ttmlir-opt --auto-sharding %s | FileCheck %s

sdy.mesh @mesh = <["x"=1]>

func.func @main(%arg0: tensor<32x32xf32>) -> tensor<32x32xf32> {
  %0 = stablehlo.abs %arg0 : tensor<32x32xf32>
  return %0 : tensor<32x32xf32>
}

// With a size-1 mesh axis there are no shardable axes.
// Auto-sharding should skip and leave the module unchanged.
// CHECK: func.func @main(%arg0: tensor<32x32xf32>
// CHECK-NOT: sdy.sharding
// CHECK: stablehlo.abs %arg0 : tensor<32x32xf32>
