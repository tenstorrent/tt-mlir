// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// RUN: not ttmlir-opt --split-input-file %s 2>&1 | FileCheck %s

// input_tensor must be 4D
module attributes {} {
  func.func @dispatch_input_rank(%input: tensor<128x2880xbf16>, %indices: tensor<1x1x128x4xi64>, %mapping: tensor<1x1x32x8xi64>) -> (tensor<1x2x128x2880xbf16>, tensor<1x2x128x4xi64>) {
    %0, %1 = "ttir.all_to_all_dispatch"(%input, %indices, %mapping) <{cluster_axis = 0 : i64, num_devices = 2 : i64}> : (tensor<128x2880xbf16>, tensor<1x1x128x4xi64>, tensor<1x1x32x8xi64>) -> (tensor<1x2x128x2880xbf16>, tensor<1x2x128x4xi64>)
    return %0, %1 : tensor<1x2x128x2880xbf16>, tensor<1x2x128x4xi64>
  }
}
// CHECK: error: 'ttir.all_to_all_dispatch' op input_tensor must be a 4D tensor

// -----

// expert_indices must be 4D
module attributes {} {
  func.func @dispatch_indices_rank(%input: tensor<1x1x128x2880xbf16>, %indices: tensor<128x4xi64>, %mapping: tensor<1x1x32x8xi64>) -> (tensor<1x2x128x2880xbf16>, tensor<1x2x128x4xi64>) {
    %0, %1 = "ttir.all_to_all_dispatch"(%input, %indices, %mapping) <{cluster_axis = 0 : i64, num_devices = 2 : i64}> : (tensor<1x1x128x2880xbf16>, tensor<128x4xi64>, tensor<1x1x32x8xi64>) -> (tensor<1x2x128x2880xbf16>, tensor<1x2x128x4xi64>)
    return %0, %1 : tensor<1x2x128x2880xbf16>, tensor<1x2x128x4xi64>
  }
}
// CHECK: error: 'ttir.all_to_all_dispatch' op expert_indices must be a 4D tensor

// -----

// num_devices must be positive
module attributes {} {
  func.func @dispatch_num_devices(%input: tensor<1x1x128x2880xbf16>, %indices: tensor<1x1x128x4xi64>, %mapping: tensor<1x1x32x8xi64>) -> (tensor<1x2x128x2880xbf16>, tensor<1x2x128x4xi64>) {
    %0, %1 = "ttir.all_to_all_dispatch"(%input, %indices, %mapping) <{cluster_axis = 0 : i64, num_devices = 0 : i64}> : (tensor<1x1x128x2880xbf16>, tensor<1x1x128x4xi64>, tensor<1x1x32x8xi64>) -> (tensor<1x2x128x2880xbf16>, tensor<1x2x128x4xi64>)
    return %0, %1 : tensor<1x2x128x2880xbf16>, tensor<1x2x128x4xi64>
  }
}
// CHECK: error: 'ttir.all_to_all_dispatch' op num_devices must be positive

// -----

// cluster_axis must be 0 or 1
module attributes {} {
  func.func @dispatch_cluster_axis(%input: tensor<1x1x128x2880xbf16>, %indices: tensor<1x1x128x4xi64>, %mapping: tensor<1x1x32x8xi64>) -> (tensor<1x2x128x2880xbf16>, tensor<1x2x128x4xi64>) {
    %0, %1 = "ttir.all_to_all_dispatch"(%input, %indices, %mapping) <{cluster_axis = 5 : i64, num_devices = 2 : i64}> : (tensor<1x1x128x2880xbf16>, tensor<1x1x128x4xi64>, tensor<1x1x32x8xi64>) -> (tensor<1x2x128x2880xbf16>, tensor<1x2x128x4xi64>)
    return %0, %1 : tensor<1x2x128x2880xbf16>, tensor<1x2x128x4xi64>
  }
}
// CHECK: error: 'ttir.all_to_all_dispatch' op cluster_axis must be 0 or 1
