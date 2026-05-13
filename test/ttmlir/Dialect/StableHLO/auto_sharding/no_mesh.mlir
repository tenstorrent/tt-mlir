// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// REQUIRES: stablehlo
// RUN: ttmlir-opt --auto-sharding %s | FileCheck %s

func.func @main(%arg0: tensor<32x32xf32>) -> tensor<32x32xf32> {
  %0 = stablehlo.abs %arg0 : tensor<32x32xf32>
  return %0 : tensor<32x32xf32>
}

// Without a mesh, auto-sharding should skip and leave the module unchanged.
// CHECK: func.func @main(%arg0: tensor<32x32xf32>
// CHECK-NOT: sdy.sharding
// CHECK: stablehlo.abs %arg0 : tensor<32x32xf32>
