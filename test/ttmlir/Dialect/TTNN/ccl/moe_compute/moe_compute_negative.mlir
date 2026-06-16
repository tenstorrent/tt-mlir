// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// RUN: not ttmlir-opt --split-input-file %s 2>&1 | FileCheck %s

// intermediate_size must be a positive multiple of 32
module attributes {} {
  func.func @moe_compute_bad_intermediate(
      %inp: tensor<1x1x128x256xbf16>,
      %idx: tensor<1x1x128x4xi64>,
      %scores: tensor<1x1x128x4xbf16>,
      %map: tensor<1x1x16x4xi64>,
      %w01: tensor<8x1x16x32x256x128xbf16>,
      %w2: tensor<8x1x16x32x256x128xbf16>,
      %device: !ttnn.device)
    -> (tensor<1x1x16x4xi64>, tensor<1x1x128x4xbf16>, tensor<1x1x128x4xi64>,
        tensor<1x1x128x256xbf16>, tensor<1x1x128x256xbf16>,
        tensor<1x1x128x256xbf16>) {
    %0:6 = "ttnn.moe_compute"(%inp, %idx, %scores, %map, %w01, %w2, %device)
      <{operandSegmentSizes = array<i32: 1, 1, 1, 1, 1, 1, 0, 0, 1>,
        layer_id = 0 : ui32,
        output_height_shard_dim = 32 : ui32,
        intermediate_size = 100 : ui32,
        has_bias = false,
        cluster_axis = 0 : ui32}>
      : (tensor<1x1x128x256xbf16>, tensor<1x1x128x4xi64>, tensor<1x1x128x4xbf16>,
         tensor<1x1x16x4xi64>, tensor<8x1x16x32x256x128xbf16>,
         tensor<8x1x16x32x256x128xbf16>, !ttnn.device)
      -> (tensor<1x1x16x4xi64>, tensor<1x1x128x4xbf16>, tensor<1x1x128x4xi64>,
          tensor<1x1x128x256xbf16>, tensor<1x1x128x256xbf16>,
          tensor<1x1x128x256xbf16>)
    return %0#0, %0#1, %0#2, %0#3, %0#4, %0#5
      : tensor<1x1x16x4xi64>, tensor<1x1x128x4xbf16>, tensor<1x1x128x4xi64>,
        tensor<1x1x128x256xbf16>, tensor<1x1x128x256xbf16>, tensor<1x1x128x256xbf16>
  }
}
// CHECK: error: 'ttnn.moe_compute' op intermediate_size must be a positive multiple of 32

// -----

// prepare_moe_compute_w0_w1_weights: bias_0/bias_1 must both be present or absent
module attributes {} {
  func.func @prep_w0_w1_unpaired_bias(
      %w0: tensor<1x16x256x128xbf16>,
      %w1: tensor<1x16x256x128xbf16>,
      %b0: tensor<1x16x128xbf16>,
      %device: !ttnn.device)
    -> tensor<8x1x16x32x256x128xbf16> {
    %0 = "ttnn.prepare_moe_compute_w0_w1_weights"(%w0, %w1, %b0, %device)
      <{operandSegmentSizes = array<i32: 1, 1, 1, 0, 1>,
        hidden_size = 128 : ui32,
        intermediate_size = 128 : ui32}>
      : (tensor<1x16x256x128xbf16>, tensor<1x16x256x128xbf16>,
         tensor<1x16x128xbf16>, !ttnn.device)
      -> tensor<8x1x16x32x256x128xbf16>
    return %0 : tensor<8x1x16x32x256x128xbf16>
  }
}
// CHECK: error: 'ttnn.prepare_moe_compute_w0_w1_weights' op bias_0 and bias_1 must be both present or both absent

// -----

// only the compute_only path is supported
module attributes {} {
  func.func @moe_compute_full_path_rejected(
      %inp: tensor<1x1x128x256xbf16>,
      %idx: tensor<1x1x128x4xi64>,
      %scores: tensor<1x1x128x4xbf16>,
      %map: tensor<1x1x16x4xi64>,
      %w01: tensor<8x1x16x32x256x128xbf16>,
      %w2: tensor<8x1x16x32x256x128xbf16>,
      %device: !ttnn.device)
    -> (tensor<1x1x16x4xi64>, tensor<1x1x128x4xbf16>, tensor<1x1x128x4xi64>,
        tensor<1x1x128x256xbf16>, tensor<1x1x128x256xbf16>,
        tensor<1x1x128x256xbf16>) {
    %0:6 = "ttnn.moe_compute"(%inp, %idx, %scores, %map, %w01, %w2, %device)
      <{operandSegmentSizes = array<i32: 1, 1, 1, 1, 1, 1, 0, 0, 1>,
        layer_id = 0 : ui32,
        output_height_shard_dim = 32 : ui32,
        intermediate_size = 256 : ui32,
        has_bias = false,
        cluster_axis = 0 : ui32,
        compute_only = false}>
      : (tensor<1x1x128x256xbf16>, tensor<1x1x128x4xi64>, tensor<1x1x128x4xbf16>,
         tensor<1x1x16x4xi64>, tensor<8x1x16x32x256x128xbf16>,
         tensor<8x1x16x32x256x128xbf16>, !ttnn.device)
      -> (tensor<1x1x16x4xi64>, tensor<1x1x128x4xbf16>, tensor<1x1x128x4xi64>,
          tensor<1x1x128x256xbf16>, tensor<1x1x128x256xbf16>,
          tensor<1x1x128x256xbf16>)
    return %0#0, %0#1, %0#2, %0#3, %0#4, %0#5
      : tensor<1x1x16x4xi64>, tensor<1x1x128x4xbf16>, tensor<1x1x128x4xi64>,
        tensor<1x1x128x256xbf16>, tensor<1x1x128x256xbf16>, tensor<1x1x128x256xbf16>
  }
}
// CHECK: error: 'ttnn.moe_compute' op only the compute_only path is supported

// -----

// hidden_size must be at least one tile per matmul-ring core (bh_ring_size*32,
// default 12*32 = 384); hidden=256 here leaves W2 output tiles uncomputed.
module attributes {} {
  func.func @moe_compute_hidden_too_small(
      %inp: tensor<1x1x128x256xbf16>,
      %idx: tensor<1x1x128x4xi64>,
      %scores: tensor<1x1x128x4xbf16>,
      %map: tensor<1x1x16x4xi64>,
      %w01: tensor<8x1x16x32x256x128xbf16>,
      %w2: tensor<8x1x16x32x256x128xbf16>,
      %device: !ttnn.device)
    -> (tensor<1x1x16x4xi64>, tensor<1x1x128x4xbf16>, tensor<1x1x128x4xi64>,
        tensor<1x1x128x256xbf16>, tensor<1x1x128x256xbf16>,
        tensor<1x1x128x256xbf16>) {
    %0:6 = "ttnn.moe_compute"(%inp, %idx, %scores, %map, %w01, %w2, %device)
      <{operandSegmentSizes = array<i32: 1, 1, 1, 1, 1, 1, 0, 0, 1>,
        layer_id = 0 : ui32,
        output_height_shard_dim = 32 : ui32,
        intermediate_size = 384 : ui32,
        has_bias = false}>
      : (tensor<1x1x128x256xbf16>, tensor<1x1x128x4xi64>, tensor<1x1x128x4xbf16>,
         tensor<1x1x16x4xi64>, tensor<8x1x16x32x256x128xbf16>,
         tensor<8x1x16x32x256x128xbf16>, !ttnn.device)
      -> (tensor<1x1x16x4xi64>, tensor<1x1x128x4xbf16>, tensor<1x1x128x4xi64>,
          tensor<1x1x128x256xbf16>, tensor<1x1x128x256xbf16>,
          tensor<1x1x128x256xbf16>)
    return %0#0, %0#1, %0#2, %0#3, %0#4, %0#5
      : tensor<1x1x16x4xi64>, tensor<1x1x128x4xbf16>, tensor<1x1x128x4xi64>,
        tensor<1x1x128x256xbf16>, tensor<1x1x128x256xbf16>, tensor<1x1x128x256xbf16>
  }
}
// CHECK: error: 'ttnn.moe_compute' op hidden_size (256) must be at least bh_ring_size*32 = 384
