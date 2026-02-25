// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// RUN: ttmlir-opt --split-input-file \
// RUN:   --ttcore-register-device="mesh-shape=2,2 mesh-topology=ring,linear" \
// RUN:   --ttnn-configure-ccl-ops \
// RUN:   -o %t %s
// RUN: FileCheck %s --input-file=%t


// cluster_axis=1: topologyIdx = 2-1-1 = 0 -> meshTopology[0] = ring
module attributes {} {
  // CHECK-LABEL: all_gather_cluster_axis1
  func.func @all_gather_cluster_axis1(%arg0: tensor<1x1x32x32xbf16>) -> tensor<1x1x32x64xbf16> {
    %0 = "ttnn.all_gather"(%arg0) <{all_gather_dim = 3 : si32, cluster_axis = 1 : ui32}> : (tensor<1x1x32x32xbf16>) -> tensor<1x1x32x64xbf16>
    // CHECK: "ttnn.all_gather"
    // CHECK-SAME: topology = #ttcore.topology<ring>
    return %0 : tensor<1x1x32x64xbf16>
  }
}

// cluster_axis=0: topologyIdx = 2-1-0 = 1 -> meshTopology[1] = linear
module attributes {} {
  // CHECK-LABEL: all_gather_cluster_axis0
  func.func @all_gather_cluster_axis0(%arg0: tensor<1x1x32x32xbf16>) -> tensor<1x1x32x64xbf16> {
    %0 = "ttnn.all_gather"(%arg0) <{all_gather_dim = 3 : si32, cluster_axis = 0 : ui32}> : (tensor<1x1x32x32xbf16>) -> tensor<1x1x32x64xbf16>
    // CHECK: "ttnn.all_gather"
    // CHECK-SAME: topology = #ttcore.topology<linear>
    return %0 : tensor<1x1x32x64xbf16>
  }
}
