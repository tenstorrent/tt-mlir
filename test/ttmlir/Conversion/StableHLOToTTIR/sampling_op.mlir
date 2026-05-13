// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// REQUIRES: stablehlo
// RUN: ttmlir-opt --stablehlo-to-ttir-pipeline -o %t %s
// RUN: FileCheck %s --input-file=%t

module attributes {} {

  // Test: stablehlo custom_call "tt.sampling" → ttir.sampling, with seed.
  // CHECK-LABEL: func.func @sampling_with_seed
  func.func @sampling_with_seed(
      %arg0: tensor<1x32xbf16>,
      %arg1: tensor<1x32xi32>,
      %arg2: tensor<1xui32>,
      %arg3: tensor<1xbf16>,
      %arg4: tensor<1xbf16>) -> tensor<1xi32> {
    // CHECK: "ttir.sampling"(%arg0, %arg1, %arg2, %arg3, %arg4)
    // CHECK-SAME: seed = 42 : ui32
    // CHECK-SAME: -> tensor<1xi32>
    %0 = stablehlo.custom_call @tt.sampling(%arg0, %arg1, %arg2, %arg3, %arg4) {
        api_version = 0 : i32,
        mhlo.frontend_attributes = {seed = "42"}
    } : (tensor<1x32xbf16>, tensor<1x32xi32>, tensor<1xui32>, tensor<1xbf16>, tensor<1xbf16>) -> tensor<1xi32>
    return %0 : tensor<1xi32>
  }

  // Test: stablehlo custom_call "tt.sampling" → ttir.sampling, without seed.
  // CHECK-LABEL: func.func @sampling_no_seed
  func.func @sampling_no_seed(
      %arg0: tensor<4x64xbf16>,
      %arg1: tensor<4x64xi32>,
      %arg2: tensor<4xui32>,
      %arg3: tensor<4xbf16>,
      %arg4: tensor<4xbf16>) -> tensor<4xi32> {
    // CHECK: "ttir.sampling"(%arg0, %arg1, %arg2, %arg3, %arg4)
    // CHECK-NOT: seed
    // CHECK-SAME: -> tensor<4xi32>
    %0 = stablehlo.custom_call @tt.sampling(%arg0, %arg1, %arg2, %arg3, %arg4) {
        api_version = 0 : i32
    } : (tensor<4x64xbf16>, tensor<4x64xi32>, tensor<4xui32>, tensor<4xbf16>, tensor<4xbf16>) -> tensor<4xi32>
    return %0 : tensor<4xi32>
  }

}
