// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// RUN: not ttmlir-opt --split-input-file %s 2>&1 | FileCheck %s

// w0_w1_tensor must be rank 6
module attributes {} {
  func.func @moe_gpt_w0w1_rank(
      %input: tensor<128x2880xbf16>,
      %indices: tensor<128x4xui16>,
      %scores: tensor<128x4xbf16>,
      %mapping: tensor<1x16xui16>,
      %w0w1: tensor<12x4x2880x128xbf16>,
      %w2: tensor<12x1x4x2x2880x128xbf16>
  ) -> (tensor<1x16xui32>, tensor<1x512xui32>, tensor<4x129xui32>, tensor<12x2x32x2880xbf16>, tensor<12x2x32x2880xbf16>) {
    %0, %1, %2, %3, %4 = "ttir.moe_gpt"(%input, %indices, %scores, %mapping, %w0w1, %w2)
        <{output_height_shard_dim = 4 : ui32, output_width_shard_dim = 3 : ui32, hidden_size = 2880 : ui32}>
        : (tensor<128x2880xbf16>, tensor<128x4xui16>, tensor<128x4xbf16>,
           tensor<1x16xui16>, tensor<12x4x2880x128xbf16>,
           tensor<12x1x4x2x2880x128xbf16>)
        -> (tensor<1x16xui32>, tensor<1x512xui32>, tensor<4x129xui32>,
            tensor<12x2x32x2880xbf16>, tensor<12x2x32x2880xbf16>)
    return %0, %1, %2, %3, %4
        : tensor<1x16xui32>, tensor<1x512xui32>, tensor<4x129xui32>,
          tensor<12x2x32x2880xbf16>, tensor<12x2x32x2880xbf16>
  }
}
// CHECK: error: 'ttir.moe_gpt' op w0_w1_tensor must be a rank 6 tensor

// -----

// w2_tensor must be rank 6
module attributes {} {
  func.func @moe_gpt_w2_rank(
      %input: tensor<128x2880xbf16>,
      %indices: tensor<128x4xui16>,
      %scores: tensor<128x4xbf16>,
      %mapping: tensor<1x16xui16>,
      %w0w1: tensor<12x1x4x4x2880x128xbf16>,
      %w2: tensor<12x4x2880x128xbf16>
  ) -> (tensor<1x16xui32>, tensor<1x512xui32>, tensor<4x129xui32>, tensor<12x2x32x2880xbf16>, tensor<12x2x32x2880xbf16>) {
    %0, %1, %2, %3, %4 = "ttir.moe_gpt"(%input, %indices, %scores, %mapping, %w0w1, %w2)
        <{output_height_shard_dim = 4 : ui32, output_width_shard_dim = 3 : ui32, hidden_size = 2880 : ui32}>
        : (tensor<128x2880xbf16>, tensor<128x4xui16>, tensor<128x4xbf16>,
           tensor<1x16xui16>, tensor<12x1x4x4x2880x128xbf16>,
           tensor<12x4x2880x128xbf16>)
        -> (tensor<1x16xui32>, tensor<1x512xui32>, tensor<4x129xui32>,
            tensor<12x2x32x2880xbf16>, tensor<12x2x32x2880xbf16>)
    return %0, %1, %2, %3, %4
        : tensor<1x16xui32>, tensor<1x512xui32>, tensor<4x129xui32>,
          tensor<12x2x32x2880xbf16>, tensor<12x2x32x2880xbf16>
  }
}
// CHECK: error: 'ttir.moe_gpt' op w2_tensor must be a rank 6 tensor

// -----

// w0_w1_tensor dim[5] must be 128
module attributes {} {
  func.func @moe_gpt_w0w1_dim5(
      %input: tensor<128x2880xbf16>,
      %indices: tensor<128x4xui16>,
      %scores: tensor<128x4xbf16>,
      %mapping: tensor<1x16xui16>,
      %w0w1: tensor<12x1x4x4x2880x64xbf16>,
      %w2: tensor<12x1x4x2x2880x128xbf16>
  ) -> (tensor<1x16xui32>, tensor<1x512xui32>, tensor<4x129xui32>, tensor<12x2x32x2880xbf16>, tensor<12x2x32x2880xbf16>) {
    %0, %1, %2, %3, %4 = "ttir.moe_gpt"(%input, %indices, %scores, %mapping, %w0w1, %w2)
        <{output_height_shard_dim = 4 : ui32, output_width_shard_dim = 3 : ui32, hidden_size = 2880 : ui32}>
        : (tensor<128x2880xbf16>, tensor<128x4xui16>, tensor<128x4xbf16>,
           tensor<1x16xui16>, tensor<12x1x4x4x2880x64xbf16>,
           tensor<12x1x4x2x2880x128xbf16>)
        -> (tensor<1x16xui32>, tensor<1x512xui32>, tensor<4x129xui32>,
            tensor<12x2x32x2880xbf16>, tensor<12x2x32x2880xbf16>)
    return %0, %1, %2, %3, %4
        : tensor<1x16xui32>, tensor<1x512xui32>, tensor<4x129xui32>,
          tensor<12x2x32x2880xbf16>, tensor<12x2x32x2880xbf16>
  }
}
// CHECK: error: 'ttir.moe_gpt' op w0_w1_tensor dim[5] must be 128 (4*TILE_SIZE)
