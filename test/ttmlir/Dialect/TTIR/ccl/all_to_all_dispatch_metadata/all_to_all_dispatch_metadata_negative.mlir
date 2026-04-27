// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// RUN: not ttmlir-opt --split-input-file %s 2>&1 | FileCheck %s

// input_tensor must be 4D
module attributes {} {
  func.func @dispatch_metadata_input_rank(%input: tensor<128x2880xbf16>, %indices: tensor<1x1x128x4xi64>, %scores: tensor<1x1x128x4xbf16>, %mapping: tensor<1x1x32x8xi64>) -> (tensor<1x256x2880xbf16>, tensor<1x256x4xi64>, tensor<1x256x4xbf16>) {
    %0:3 = "ttir.composite"(%input, %indices, %scores, %mapping) <{name = "tt.all_to_all_dispatch_metadata", composite_attributes = {cluster_axis = 0 : i64, num_devices = 2 : i64}}> : (tensor<128x2880xbf16>, tensor<1x1x128x4xi64>, tensor<1x1x128x4xbf16>, tensor<1x1x32x8xi64>) -> (tensor<1x256x2880xbf16>, tensor<1x256x4xi64>, tensor<1x256x4xbf16>)
    return %0#0, %0#1, %0#2 : tensor<1x256x2880xbf16>, tensor<1x256x4xi64>, tensor<1x256x4xbf16>
  }
}
// CHECK: error: 'ttir.composite' op input_tensor must be a 4D tensor

// -----

// expert_scores must be 4D
module attributes {} {
  func.func @dispatch_metadata_scores_rank(%input: tensor<1x1x128x2880xbf16>, %indices: tensor<1x1x128x4xi64>, %scores: tensor<128x4xbf16>, %mapping: tensor<1x1x32x8xi64>) -> (tensor<1x256x2880xbf16>, tensor<1x256x4xi64>, tensor<1x256x4xbf16>) {
    %0:3 = "ttir.composite"(%input, %indices, %scores, %mapping) <{name = "tt.all_to_all_dispatch_metadata", composite_attributes = {cluster_axis = 0 : i64, num_devices = 2 : i64}}> : (tensor<1x1x128x2880xbf16>, tensor<1x1x128x4xi64>, tensor<128x4xbf16>, tensor<1x1x32x8xi64>) -> (tensor<1x256x2880xbf16>, tensor<1x256x4xi64>, tensor<1x256x4xbf16>)
    return %0#0, %0#1, %0#2 : tensor<1x256x2880xbf16>, tensor<1x256x4xi64>, tensor<1x256x4xbf16>
  }
}
// CHECK: error: 'ttir.composite' op expert_scores must be a 4D tensor

// -----

// num_devices must be positive
module attributes {} {
  func.func @dispatch_metadata_num_devices(%input: tensor<1x1x128x2880xbf16>, %indices: tensor<1x1x128x4xi64>, %scores: tensor<1x1x128x4xbf16>, %mapping: tensor<1x1x32x8xi64>) -> (tensor<1x256x2880xbf16>, tensor<1x256x4xi64>, tensor<1x256x4xbf16>) {
    %0:3 = "ttir.composite"(%input, %indices, %scores, %mapping) <{name = "tt.all_to_all_dispatch_metadata", composite_attributes = {cluster_axis = 0 : i64, num_devices = 0 : i64}}> : (tensor<1x1x128x2880xbf16>, tensor<1x1x128x4xi64>, tensor<1x1x128x4xbf16>, tensor<1x1x32x8xi64>) -> (tensor<1x256x2880xbf16>, tensor<1x256x4xi64>, tensor<1x256x4xbf16>)
    return %0#0, %0#1, %0#2 : tensor<1x256x2880xbf16>, tensor<1x256x4xi64>, tensor<1x256x4xbf16>
  }
}
// CHECK: error: 'ttir.composite' op 'num_devices' must be positive

// -----

// cluster_axis must be 0 or 1
module attributes {} {
  func.func @dispatch_metadata_cluster_axis(%input: tensor<1x1x128x2880xbf16>, %indices: tensor<1x1x128x4xi64>, %scores: tensor<1x1x128x4xbf16>, %mapping: tensor<1x1x32x8xi64>) -> (tensor<1x256x2880xbf16>, tensor<1x256x4xi64>, tensor<1x256x4xbf16>) {
    %0:3 = "ttir.composite"(%input, %indices, %scores, %mapping) <{name = "tt.all_to_all_dispatch_metadata", composite_attributes = {cluster_axis = 5 : i64, num_devices = 2 : i64}}> : (tensor<1x1x128x2880xbf16>, tensor<1x1x128x4xi64>, tensor<1x1x128x4xbf16>, tensor<1x1x32x8xi64>) -> (tensor<1x256x2880xbf16>, tensor<1x256x4xi64>, tensor<1x256x4xbf16>)
    return %0#0, %0#1, %0#2 : tensor<1x256x2880xbf16>, tensor<1x256x4xi64>, tensor<1x256x4xbf16>
  }
}
// CHECK: error: 'ttir.composite' op 'cluster_axis' must be 0 or 1
