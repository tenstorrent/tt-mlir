// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// RUN: not ttmlir-opt --split-input-file %s 2>&1 | FileCheck %s

// moe_compute takes raw per-expert weights w0/w1/w2 (L, E, K, N); device
// weight prepacking is a TTNN concern inserted during TTIRToTTNN. Only the
// compute_only path is supported, so the full-path-only input (cluster_axis)
// must be unset.

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
    -> (tensor<1x1x16x4xi64>, tensor<1x1x128x4xbf16>, tensor<1x1x128x4xi64>,
        tensor<1x1x128x256xbf16>, tensor<1x1x128x256xbf16>,
        tensor<1x1x128x256xbf16>) {
    %0:6 = "ttir.moe_compute"(%inp, %idx, %scores, %map, %w0, %w1, %w2)
      <{operandSegmentSizes = array<i32: 1, 1, 1, 1, 1, 1, 1, 0, 0, 0>,
        layer_id = 0 : ui32,
        output_height_shard_dim = 32 : ui32,
        intermediate_size = 100 : ui32}>
      : (tensor<1x1x128x256xbf16>, tensor<1x1x128x4xi64>, tensor<1x1x128x4xbf16>,
         tensor<1x1x16x4xi64>, tensor<1x16x256x128xbf16>,
         tensor<1x16x256x128xbf16>, tensor<1x16x128x256xbf16>)
      -> (tensor<1x1x16x4xi64>, tensor<1x1x128x4xbf16>, tensor<1x1x128x4xi64>,
          tensor<1x1x128x256xbf16>, tensor<1x1x128x256xbf16>,
          tensor<1x1x128x256xbf16>)
    return %0#0, %0#1, %0#2, %0#3, %0#4, %0#5
      : tensor<1x1x16x4xi64>, tensor<1x1x128x4xbf16>, tensor<1x1x128x4xi64>,
        tensor<1x1x128x256xbf16>, tensor<1x1x128x256xbf16>, tensor<1x1x128x256xbf16>
  }
}
// CHECK: error: 'ttir.moe_compute' op intermediate_size must be a positive multiple of 32

// -----

// only the compute_only path is supported
module attributes {} {
  func.func @moe_compute_full_path_rejected(
      %inp: tensor<1x1x128x256xbf16>,
      %idx: tensor<1x1x128x4xi64>,
      %scores: tensor<1x1x128x4xbf16>,
      %map: tensor<1x1x16x4xi64>,
      %w0: tensor<1x16x256x128xbf16>,
      %w1: tensor<1x16x256x128xbf16>,
      %w2: tensor<1x16x128x256xbf16>)
    -> (tensor<1x1x16x4xi64>, tensor<1x1x128x4xbf16>, tensor<1x1x128x4xi64>,
        tensor<1x1x128x256xbf16>, tensor<1x1x128x256xbf16>,
        tensor<1x1x128x256xbf16>) {
    %0:6 = "ttir.moe_compute"(%inp, %idx, %scores, %map, %w0, %w1, %w2)
      <{operandSegmentSizes = array<i32: 1, 1, 1, 1, 1, 1, 1, 0, 0, 0>,
        layer_id = 0 : ui32,
        output_height_shard_dim = 32 : ui32,
        intermediate_size = 256 : ui32,
        compute_only = false}>
      : (tensor<1x1x128x256xbf16>, tensor<1x1x128x4xi64>, tensor<1x1x128x4xbf16>,
         tensor<1x1x16x4xi64>, tensor<1x16x256x128xbf16>,
         tensor<1x16x256x128xbf16>, tensor<1x16x128x256xbf16>)
      -> (tensor<1x1x16x4xi64>, tensor<1x1x128x4xbf16>, tensor<1x1x128x4xi64>,
          tensor<1x1x128x256xbf16>, tensor<1x1x128x256xbf16>,
          tensor<1x1x128x256xbf16>)
    return %0#0, %0#1, %0#2, %0#3, %0#4, %0#5
      : tensor<1x1x16x4xi64>, tensor<1x1x128x4xbf16>, tensor<1x1x128x4xi64>,
        tensor<1x1x128x256xbf16>, tensor<1x1x128x256xbf16>, tensor<1x1x128x256xbf16>
  }
}
// CHECK: error: 'ttir.moe_compute' op only the compute_only path is supported

// -----

// compute_only moe_compute must not set cluster_axis
module attributes {} {
  func.func @moe_compute_cluster_axis_rejected(
      %inp: tensor<1x1x128x256xbf16>,
      %idx: tensor<1x1x128x4xi64>,
      %scores: tensor<1x1x128x4xbf16>,
      %map: tensor<1x1x16x4xi64>,
      %w0: tensor<1x16x256x128xbf16>,
      %w1: tensor<1x16x256x128xbf16>,
      %w2: tensor<1x16x128x256xbf16>)
    -> (tensor<1x1x16x4xi64>, tensor<1x1x128x4xbf16>, tensor<1x1x128x4xi64>,
        tensor<1x1x128x256xbf16>, tensor<1x1x128x256xbf16>,
        tensor<1x1x128x256xbf16>) {
    %0:6 = "ttir.moe_compute"(%inp, %idx, %scores, %map, %w0, %w1, %w2)
      <{operandSegmentSizes = array<i32: 1, 1, 1, 1, 1, 1, 1, 0, 0, 0>,
        layer_id = 0 : ui32,
        output_height_shard_dim = 32 : ui32,
        intermediate_size = 256 : ui32,
        cluster_axis = 0 : ui32}>
      : (tensor<1x1x128x256xbf16>, tensor<1x1x128x4xi64>, tensor<1x1x128x4xbf16>,
         tensor<1x1x16x4xi64>, tensor<1x16x256x128xbf16>,
         tensor<1x16x256x128xbf16>, tensor<1x16x128x256xbf16>)
      -> (tensor<1x1x16x4xi64>, tensor<1x1x128x4xbf16>, tensor<1x1x128x4xi64>,
          tensor<1x1x128x256xbf16>, tensor<1x1x128x256xbf16>,
          tensor<1x1x128x256xbf16>)
    return %0#0, %0#1, %0#2, %0#3, %0#4, %0#5
      : tensor<1x1x16x4xi64>, tensor<1x1x128x4xbf16>, tensor<1x1x128x4xi64>,
        tensor<1x1x128x256xbf16>, tensor<1x1x128x256xbf16>, tensor<1x1x128x256xbf16>
  }
}
// CHECK: error: 'ttir.moe_compute' op compute_only moe_compute must not set cluster_axis

// -----

// hidden_size must be at least one tile per matmul-ring core (bh_ring_size*32,
// default 12*32 = 384); hidden=256 here leaves W2 output tiles uncomputed.
module attributes {} {
  func.func @moe_compute_hidden_too_small(
      %inp: tensor<1x1x128x256xbf16>,
      %idx: tensor<1x1x128x4xi64>,
      %scores: tensor<1x1x128x4xbf16>,
      %map: tensor<1x1x16x4xi64>,
      %w0: tensor<1x16x256x384xbf16>,
      %w1: tensor<1x16x256x384xbf16>,
      %w2: tensor<1x16x384x256xbf16>)
    -> (tensor<1x1x16x4xi64>, tensor<1x1x128x4xbf16>, tensor<1x1x128x4xi64>,
        tensor<1x1x128x256xbf16>, tensor<1x1x128x256xbf16>,
        tensor<1x1x128x256xbf16>) {
    %0:6 = "ttir.moe_compute"(%inp, %idx, %scores, %map, %w0, %w1, %w2)
      <{operandSegmentSizes = array<i32: 1, 1, 1, 1, 1, 1, 1, 0, 0, 0>,
        layer_id = 0 : ui32,
        output_height_shard_dim = 32 : ui32,
        intermediate_size = 384 : ui32}>
      : (tensor<1x1x128x256xbf16>, tensor<1x1x128x4xi64>, tensor<1x1x128x4xbf16>,
         tensor<1x1x16x4xi64>, tensor<1x16x256x384xbf16>,
         tensor<1x16x256x384xbf16>, tensor<1x16x384x256xbf16>)
      -> (tensor<1x1x16x4xi64>, tensor<1x1x128x4xbf16>, tensor<1x1x128x4xi64>,
          tensor<1x1x128x256xbf16>, tensor<1x1x128x256xbf16>,
          tensor<1x1x128x256xbf16>)
    return %0#0, %0#1, %0#2, %0#3, %0#4, %0#5
      : tensor<1x1x16x4xi64>, tensor<1x1x128x4xbf16>, tensor<1x1x128x4xi64>,
        tensor<1x1x128x256xbf16>, tensor<1x1x128x256xbf16>, tensor<1x1x128x256xbf16>
  }
}
// CHECK: error: 'ttir.moe_compute' op hidden_size (256) must be at least bh_ring_size*32 = 384
