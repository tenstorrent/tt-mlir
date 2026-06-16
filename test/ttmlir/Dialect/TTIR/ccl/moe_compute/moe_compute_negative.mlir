// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// RUN: not ttmlir-opt --split-input-file %s 2>&1 | FileCheck %s

// moe_compute takes raw per-expert weights w0/w1/w2 (L, E, K, N); device weight
// prepacking is a TTNN concern inserted during TTIRToTTNN. The op always runs
// the full A2A combine path (cluster_axis is required) and exposes a single
// result, the combine output.

// intermediate_size must be a positive multiple of 32
module attributes {} {
  func.func @moe_compute_bad_intermediate(
      %inp: tensor<1x1x128x256xbf16>,
      %idx: tensor<1x1x128x4xi64>,
      %scores: tensor<1x1x128x4xbf16>,
      %map: tensor<1x1x16x4xi64>,
      %w0: tensor<1x16x256x128xbf16>,
      %w1: tensor<1x16x256x128xbf16>,
      %w2: tensor<1x16x128x256xbf16>)
    -> tensor<4x128x256xbf16> {
    %0 = "ttir.moe_compute"(%inp, %idx, %scores, %map, %w0, %w1, %w2)
      <{operandSegmentSizes = array<i32: 1, 1, 1, 1, 1, 1, 1, 0, 0, 0>,
        layer_id = 0 : ui32,
        output_height_shard_dim = 32 : ui32,
        intermediate_size = 100 : ui32,
        cluster_axis = 0 : ui32}>
      : (tensor<1x1x128x256xbf16>, tensor<1x1x128x4xi64>, tensor<1x1x128x4xbf16>,
         tensor<1x1x16x4xi64>, tensor<1x16x256x128xbf16>,
         tensor<1x16x256x128xbf16>, tensor<1x16x128x256xbf16>)
      -> tensor<4x128x256xbf16>
    return %0 : tensor<4x128x256xbf16>
  }
}
// CHECK: error: 'ttir.moe_compute' op intermediate_size must be a positive multiple of 32
