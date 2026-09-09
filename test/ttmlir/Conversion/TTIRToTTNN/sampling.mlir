// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// RUN: ttmlir-opt --ttir-to-ttnn-backend-pipeline %s | FileCheck %s

// End-to-end TTIR -> TTNN check for ttir.sampling. TTIRToTTNN itself is a 1:1
// lowering; the rank-2 -> rank-4 / rank-1 -> rank-4 reshape ops required by
// the ttnn::sampling kernel are inserted later by SamplingOpRank2RewritePattern
// (a workaround-pass decomposition added for issue #8836).
// tt-metal issue: https://github.com/tenstorrent/tt-metal/issues/47522

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
