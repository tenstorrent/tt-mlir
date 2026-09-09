// RUN: ttmlir-opt --split-input-file --ttir-to-ttnn-backend-pipeline="mesh-shape=1,1" %s | FileCheck %s
// Serialization round-trip: the prelude ttnn.allocate_moe_compute_semaphore +
// ttnn.empty and the wired ttnn.moe_compute must serialize to a flatbuffer
// (covers TTNNToFlatbuffer + the op-output-ref switches in runtime.cpp). Runs
// in CI with no device (moe_compute is OpModelExempt).
// RUN: ttmlir-opt --ttir-to-ttnn-backend-pipeline="mesh-shape=1,1" %s | ttmlir-translate --ttnn-to-flatbuffer -o %t.ttnn
// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// ttir.moe_compute takes raw per-expert weights; TTIRToTTNN inserts
// ttnn.prepare_moe_compute_w0_w1_weights + ttnn.prepare_moe_compute_w2_weights
// and a default mux_core_range_set ((1,1)-(3,3)). The combine output buffer
// (ttnn.empty) and semaphore (ttnn.allocate_moe_compute_semaphore) are bound in
// the prelude by MoeComputeOp's DistributedOpInterface hooks.
module attributes {} {
  func.func @moe_compute_inserts_weight_prep(
      %inp: tensor<1x32x384xbf16>,
      %idx: tensor<1x32x2xui16>,
      %scores: tensor<1x32x2xbf16>,
      %map: tensor<1x2xui16>,
      %w0: tensor<1x2x384x384xbf16>,
      %w1: tensor<1x2x384x384xbf16>,
      %w2: tensor<1x2x384x384xbf16>)
    -> tensor<2x32x384xbf16> {
    %0 = "ttir.moe_compute"(%inp, %idx, %scores, %map, %w0, %w1, %w2)
      <{operandSegmentSizes = array<i32: 1, 1, 1, 1, 1, 1, 1, 0, 0, 0>,
        layer_id = 0 : ui32,
        output_height_shard_dim = 4 : ui32,
        intermediate_size = 384 : ui32,
        cluster_axis = 1 : ui32}>
      : (tensor<1x32x384xbf16>, tensor<1x32x2xui16>, tensor<1x32x2xbf16>,
         tensor<1x2xui16>, tensor<1x2x384x384xbf16>, tensor<1x2x384x384xbf16>,
         tensor<1x2x384x384xbf16>)
      -> tensor<2x32x384xbf16>
    return %0 : tensor<2x32x384xbf16>
  }
}
// CHECK-LABEL: @moe_compute_inserts_weight_prep
// The semaphore + output buffer are allocated in the prelude (after get_device).
// CHECK: %[[DEV:[0-9]+]] = "ttnn.get_device"
// CHECK: %[[SEM:[0-9]+]] = "ttnn.allocate_moe_compute_semaphore"(%[[DEV]])
// CHECK-SAME: hidden_size = 384 : ui32
// CHECK-SAME: mux_core_range_set = #ttnn.core_range_set
// CHECK-SAME: output_height_shard_dim = 4 : ui32
// CHECK-SAME: -> !ttnn.global_semaphore
// CHECK: %[[OUT:[0-9]+]] = "ttnn.empty"(%[[DEV]])
// CHECK: "ttnn.prepare_moe_compute_w0_w1_weights"
// CHECK: "ttnn.prepare_moe_compute_w2_weights"
// moe_compute consumes the prelude-allocated output buffer + semaphore.
// CHECK: "ttnn.moe_compute"(%{{.*}}, %[[OUT]], %[[SEM]], %[[DEV]])
// CHECK-SAME: cluster_axis = 1 : ui32
// CHECK-SAME: mux_core_range_set = #ttnn.core_range_set
// CHECK-SAME: operandSegmentSizes = array<i32: 1, 1, 1, 1, 1, 1, 1, 1, 1>
