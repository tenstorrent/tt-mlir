// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// REQUIRES: stablehlo
// RUN: ttmlir-opt --stablehlo-to-ttir-pipeline -split-input-file -o %t %s
// RUN: FileCheck %s --input-file=%t

// Exercises CustomCallLayerNormConversionPattern: stablehlo.custom_call
// @tenstorrent.layer_norm with tt.has_custom_sharding -> ttir.layer_norm.
// This is the post-flatten path after FlattenOrConvertCompositesPass, for
// leftover unreplicated layer_norm (no gather/scatter sandwich for fusion).

// Input-only: Wan final norm_out after all_gather of the hidden dim.
module @layer_norm_custom_call_input_only attributes {mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} {
  // CHECK-LABEL: func.func @main
  func.func @main(%arg0: tensor<1x4096x5120xf32>) -> tensor<1x4096x5120xf32> {
    // CHECK: "ttir.layer_norm"(%arg0)
    // CHECK-SAME: normalized_shape = array<i64: 5120>
    // CHECK-NOT: stablehlo.custom_call @tenstorrent.layer_norm
    %0 = stablehlo.custom_call @tenstorrent.layer_norm(%arg0) {
      tt.composite_attributes = {
        epsilon = 9.99999974E-6 : f32,
        normalized_shape = dense<5120> : tensor<1xi64>
      },
      tt.has_custom_sharding
    } : (tensor<1x4096x5120xf32>) -> tensor<1x4096x5120xf32>
    return %0 : tensor<1x4096x5120xf32>
  }
}

// -----

// Weight + bias operands.
module @layer_norm_custom_call_weights attributes {mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} {
  // CHECK-LABEL: func.func @main
  func.func @main(%arg0: tensor<1x1024x768xbf16>, %arg1: tensor<768xbf16>, %arg2: tensor<768xbf16>) -> tensor<1x1024x768xbf16> {
    // CHECK: "ttir.layer_norm"(%arg0, %arg1, %arg2)
    // CHECK-SAME: normalized_shape = array<i64: 768>
    // CHECK-NOT: stablehlo.custom_call @tenstorrent.layer_norm
    %0 = stablehlo.custom_call @tenstorrent.layer_norm(%arg0, %arg1, %arg2) {
      tt.composite_attributes = {
        epsilon = 9.99999974E-6 : f32,
        normalized_shape = dense<768> : tensor<1xi64>
      },
      tt.has_custom_sharding
    } : (tensor<1x1024x768xbf16>, tensor<768xbf16>, tensor<768xbf16>) -> tensor<1x1024x768xbf16>
    return %0 : tensor<1x1024x768xbf16>
  }
}
