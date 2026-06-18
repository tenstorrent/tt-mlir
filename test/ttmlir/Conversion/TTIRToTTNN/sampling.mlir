// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// RUN: ttmlir-opt --ttir-to-ttnn-backend-pipeline %s | FileCheck %s

// ttnn.sampling operates on rank-4 [1, 1, batch, candidates] inputs and
// produces a rank-4 [1, 1, 1, batch] result (matching the ttnn::sampling
// kernel). TTIR's rank-2 in / rank-1 out view is bridged here by explicit
// reshape ops inserted in TTIRToTTNN — required so EmitPy / EmitC codegen
// paths see the kernel-true shape (issue #8836).

// CHECK-LABEL: func.func @sampling
// CHECK: "ttnn.reshape"
// CHECK-SAME: shape = [1 : i32, 1 : i32, 32 : i32, 64 : i32]
// CHECK: "ttnn.reshape"
// CHECK-SAME: shape = [1 : i32, 1 : i32, 32 : i32, 64 : i32]
// CHECK: "ttnn.sampling"
// CHECK-SAME: (tensor<1x1x32x64xbf16{{.*}}>, tensor<1x1x32x64x{{[us]?i32}}{{.*}}>, tensor<32xui32{{.*}}>, tensor<32xbf16{{.*}}>, tensor<32xbf16{{.*}}>)
// CHECK-SAME: -> tensor<1x1x1x32x{{[us]?i32}}
// CHECK: "ttnn.reshape"
// CHECK-SAME: shape = [32 : i32]
module {
  func.func @sampling(
      %arg0: tensor<32x64xbf16>,
      %arg1: tensor<32x64xi32>,
      %arg2: tensor<32xui32>,
      %arg3: tensor<32xbf16>,
      %arg4: tensor<32xbf16>) -> tensor<32xi32> {
    %0 = "ttir.sampling"(%arg0, %arg1, %arg2, %arg3, %arg4)
        : (tensor<32x64xbf16>, tensor<32x64xi32>, tensor<32xui32>, tensor<32xbf16>, tensor<32xbf16>) -> tensor<32xi32>
    return %0 : tensor<32xi32>
  }
}
