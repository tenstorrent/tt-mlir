// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// Ring on the sequence-parallel axis (cluster_axis 0 -> meshTopology[1]).
// ring_joint SDPA runs there and its kernels reset persistent semaphores on
// device, so the adaLN affine must stay unfused to keep the schedule that
// trace replay survives.
// RUN: ttmlir-opt --ttcore-register-device="mesh-shape=8,4 mesh-topology=linear,ring" \
// RUN:   --ttir-fusing -o %t.sp %s
// RUN: FileCheck %s --input-file=%t.sp --check-prefix=RING-SP

// Ring on the tensor-parallel axis only. cluster_axis 0 is Linear, so the
// sequence-parallel path is unaffected and the affine still folds in.
// RUN: ttmlir-opt --ttcore-register-device="mesh-shape=8,4 mesh-topology=ring,linear" \
// RUN:   --ttir-fusing -o %t.tp %s
// RUN: FileCheck %s --input-file=%t.tp --check-prefix=RING-TP

module {
  func.func @distributed_layer_norm_affine_adaln(%arg0: tensor<1x4096x1280xf32>, %scale: tensor<1x1x1280xf32>, %shift: tensor<1x1x1280xf32>) -> tensor<1x4096x1280xf32> {
    // RING-SP: "ttir.distributed_layer_norm"
    // RING-SP-SAME: operandSegmentSizes = array<i32: 1, 0, 0, 0>
    // RING-SP: "ttir.multiply"

    // RING-TP: "ttir.distributed_layer_norm"
    // RING-TP-SAME: (tensor<1x4096x1280xf32>, tensor<1280xf32>, tensor<1280xf32>) -> tensor<1x4096x1280xf32>
    // RING-TP-NOT: "ttir.multiply"
    %one = "ttir.constant"() <{value = dense<1.000000e+00> : tensor<1x1x1280xf32>}> : () -> tensor<1x1x1280xf32>
    %0 = "ttir.distributed_layer_norm"(%arg0) <{cluster_axis = 1 : ui32, epsilon = 9.99999974E-6 : f32, operandSegmentSizes = array<i32: 1, 0, 0, 0>}> : (tensor<1x4096x1280xf32>) -> tensor<1x4096x1280xf32>
    %1 = "ttir.add"(%one, %scale) : (tensor<1x1x1280xf32>, tensor<1x1x1280xf32>) -> tensor<1x1x1280xf32>
    %2 = "ttir.broadcast"(%1) <{broadcast_dimensions = array<i64: 1, 4096, 1>}> : (tensor<1x1x1280xf32>) -> tensor<1x4096x1280xf32>
    %3 = "ttir.multiply"(%0, %2) : (tensor<1x4096x1280xf32>, tensor<1x4096x1280xf32>) -> tensor<1x4096x1280xf32>
    %4 = "ttir.broadcast"(%shift) <{broadcast_dimensions = array<i64: 1, 4096, 1>}> : (tensor<1x1x1280xf32>) -> tensor<1x4096x1280xf32>
    %5 = "ttir.add"(%3, %4) : (tensor<1x4096x1280xf32>, tensor<1x4096x1280xf32>) -> tensor<1x4096x1280xf32>
    return %5 : tensor<1x4096x1280xf32>
  }
}
