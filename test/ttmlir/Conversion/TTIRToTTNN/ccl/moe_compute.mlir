// RUN: ttmlir-opt --split-input-file --ttir-to-ttnn-backend-pipeline="mesh-shape=1,1" %s | FileCheck %s
// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// ttir.moe_compute takes raw per-expert weights; device weight prepacking is a
// TTNN concern, so TTIRToTTNN inserts ttnn.prepare_moe_compute_w0_w1_weights +
// ttnn.prepare_moe_compute_w2_weights feeding ttnn.moe_compute. Only the
// compute_only path is supported.
module attributes {} {
  func.func @moe_compute_inserts_weight_prep(
      %inp: tensor<1x32x384xbf16>,
      %idx: tensor<1x32x2xui16>,
      %scores: tensor<1x32x2xbf16>,
      %map: tensor<1x2xui16>,
      %w0: tensor<1x2x384x384xbf16>,
      %w1: tensor<1x2x384x384xbf16>,
      %w2: tensor<1x2x384x384xbf16>)
    -> tensor<110x2x32x384xbf16> {
    %0:6 = "ttir.moe_compute"(%inp, %idx, %scores, %map, %w0, %w1, %w2)
      <{operandSegmentSizes = array<i32: 1, 1, 1, 1, 1, 1, 1, 0, 0, 0>,
        layer_id = 0 : ui32,
        output_height_shard_dim = 4 : ui32,
        intermediate_size = 384 : ui32,
        compute_only = true}>
      : (tensor<1x32x384xbf16>, tensor<1x32x2xui16>, tensor<1x32x2xbf16>,
         tensor<1x2xui16>, tensor<1x2x384x384xbf16>, tensor<1x2x384x384xbf16>,
         tensor<1x2x384x384xbf16>)
      -> (tensor<110x4xui32>, tensor<1x256xui32>, tensor<2x132xui32>,
          tensor<110x2x32x384xbf16>, tensor<110x2x32x384xbf16>,
          tensor<110x2x32x384xbf16>)
    return %0#4 : tensor<110x2x32x384xbf16>
  }
}
// CHECK-LABEL: @moe_compute_inserts_weight_prep
// CHECK: "ttnn.prepare_moe_compute_w0_w1_weights"
// CHECK: "ttnn.prepare_moe_compute_w2_weights"
// CHECK: "ttnn.moe_compute"
