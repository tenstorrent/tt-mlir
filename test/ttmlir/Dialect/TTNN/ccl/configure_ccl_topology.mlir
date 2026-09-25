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

// Resolve-composites emits the fused DiT rmsnorm with no topology. Without a
// fabric-derived attr the runtime fell back to Ring and metal looked for a wrap
// link that a linear axis does not have.
// cluster_axis=0 -> topologyIdx=1 -> linear.
module attributes {} {
  // CHECK-LABEL: dit_fused_rmsnorm_defaults_from_fabric
  func.func @dit_fused_rmsnorm_defaults_from_fabric(
      %input: tensor<1x1x128x64xbf16>,
      %sem: !ttnn.global_semaphore,
      %device: !ttnn.device)
      -> tensor<1x1x128x64xbf16> {
    %0 = "ttnn.dit_fused_distributed_rmsnorm"(%input, %sem, %device) <{
      operandSegmentSizes = array<i32: 1, 0, 0, 0, 0, 0, 0, 1, 1>,
      cluster_axis = 0 : ui32,
      epsilon = 9.99999997E-7 : f32,
      num_heads_per_device = 1 : ui32,
      per_head_norm = false,
      num_links = 1 : ui32
    }> : (tensor<1x1x128x64xbf16>, !ttnn.global_semaphore, !ttnn.device) -> tensor<1x1x128x64xbf16>
    // CHECK: "ttnn.dit_fused_distributed_rmsnorm"
    // CHECK-SAME: topology = #ttcore.topology<linear>
    return %0 : tensor<1x1x128x64xbf16>
  }
}
