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

// Explicit CCL configuration must not be overwritten by the device default.
module attributes {} {
  // CHECK-LABEL: all_gather_explicit_topology
  func.func @all_gather_explicit_topology(%arg0: tensor<1x1x32x32xbf16>) -> tensor<1x1x32x64xbf16> {
    %0 = "ttnn.all_gather"(%arg0) <{all_gather_dim = 3 : si32, cluster_axis = 1 : ui32, num_links = 2 : ui32, topology = #ttcore.topology<linear>}> : (tensor<1x1x32x32xbf16>) -> tensor<1x1x32x64xbf16>
    // CHECK: "ttnn.all_gather"
    // CHECK-SAME: num_links = 2 : ui32
    // CHECK-SAME: topology = #ttcore.topology<linear>
    return %0 : tensor<1x1x32x64xbf16>
  }
}

// Fused ring-joint topology is overwritten from fabric even when already set.
// cluster_axis=0 -> topologyIdx=1 -> linear. Leaving Ring here is the hang.
module attributes {} {
  // CHECK-LABEL: ring_joint_overwrites_stale_ring
  func.func @ring_joint_overwrites_stale_ring(
      %q: tensor<1x8x128x64xbf16>,
      %k: tensor<1x8x128x64xbf16>,
      %v: tensor<1x8x128x64xbf16>,
      %buf_k: tensor<1x8x256x64xbf16>,
      %buf_v: tensor<1x8x256x64xbf16>,
      %ping: !ttnn.global_semaphore,
      %pong: !ttnn.global_semaphore)
      -> tensor<1x8x128x64xbf16> {
    %0, %1, %2 = "ttnn.ring_joint_scaled_dot_product_attention"(%q, %k, %v, %buf_k, %buf_v, %ping, %pong) <{
      operandSegmentSizes = array<i32: 1, 1, 1, 0, 0, 0, 1, 1, 2>,
      joint_strategy = "rear",
      logical_n = 256 : i64,
      dim = 2 : si32,
      cluster_axis = 0 : ui32,
      program_config = #ttnn.sdpa_program_config<
        compute_with_storage_grid_size = #ttnn.core_coord<7, 8>,
        q_chunk_size = 128,
        k_chunk_size = 256>,
      topology = #ttcore.topology<ring>
    }> : (tensor<1x8x128x64xbf16>, tensor<1x8x128x64xbf16>, tensor<1x8x128x64xbf16>, tensor<1x8x256x64xbf16>, tensor<1x8x256x64xbf16>, !ttnn.global_semaphore, !ttnn.global_semaphore) -> (tensor<1x8x128x64xbf16>, tensor<1x8x0x64xbf16>, tensor<1x8x256x1xbf16>)
    // CHECK: "ttnn.ring_joint_scaled_dot_product_attention"
    // CHECK-SAME: topology = #ttcore.topology<linear>
    return %0 : tensor<1x8x128x64xbf16>
  }
}
