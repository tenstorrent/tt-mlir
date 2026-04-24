// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// RUN: ttmlir-opt %s | FileCheck %s

// Verify basic sampling operation (batch=1, 32 candidates).
module {
  func.func @sampling_basic(
      %arg0: tensor<1x32xbf16>,
      %arg1: tensor<1x32xi32>,
      %arg2: tensor<1xui32>,
      %arg3: tensor<1xbf16>,
      %arg4: tensor<1xbf16>) -> tensor<1xi32> {
    // CHECK: "ttir.sampling"
    // CHECK-SAME: (tensor<1x32xbf16>, tensor<1x32xi32>, tensor<1xui32>, tensor<1xbf16>, tensor<1xbf16>) -> tensor<1xi32>
    %0 = "ttir.sampling"(%arg0, %arg1, %arg2, %arg3, %arg4)
        : (tensor<1x32xbf16>, tensor<1x32xi32>, tensor<1xui32>, tensor<1xbf16>, tensor<1xbf16>) -> tensor<1xi32>
    return %0 : tensor<1xi32>
  }
}

// -----

// Verify sampling with optional seed attribute.
module {
  func.func @sampling_with_seed(
      %arg0: tensor<4x128xbf16>,
      %arg1: tensor<4x128xi32>,
      %arg2: tensor<4xui32>,
      %arg3: tensor<4xbf16>,
      %arg4: tensor<4xbf16>) -> tensor<4xi32> {
    // CHECK: "ttir.sampling"
    // CHECK-SAME: seed = 42 : ui32
    %0 = "ttir.sampling"(%arg0, %arg1, %arg2, %arg3, %arg4) <{seed = 42 : ui32}>
        : (tensor<4x128xbf16>, tensor<4x128xi32>, tensor<4xui32>, tensor<4xbf16>, tensor<4xbf16>) -> tensor<4xi32>
    return %0 : tensor<4xi32>
  }
}
