// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// RUN: ttmlir-opt --split-input-file --verify-diagnostics %s

// Happy path: input_batch == batch_idx_tensor.shape[0] > 1. Should verify.
module {
  func.func @paged_fill_cache_batched_ok(
      %cache: tensor<128x12x32x256xbf16>,
      %input: tensor<4x12x65x256xbf16>,
      %page_table: tensor<8x16xi32>,
      %batch_idx: tensor<4xi32>) -> tensor<128x12x32x256xbf16> {
    %0 = "ttir.paged_fill_cache"(%cache, %input, %page_table, %batch_idx)
        : (tensor<128x12x32x256xbf16>, tensor<4x12x65x256xbf16>,
           tensor<8x16xi32>, tensor<4xi32>)
        -> tensor<128x12x32x256xbf16>
    return %0 : tensor<128x12x32x256xbf16>
  }
}

// -----

// batch_idx_tensor.shape[0] != input batch.
module {
  func.func @paged_fill_cache_batch_idx_size_mismatch(
      %cache: tensor<128x12x32x256xbf16>,
      %input: tensor<4x12x65x256xbf16>,
      %page_table: tensor<8x16xi32>,
      %batch_idx: tensor<2xi32>) -> tensor<128x12x32x256xbf16> {
    // expected-error @+1 {{'ttir.paged_fill_cache' op Batch index tensor must have dim 0 equal to input batch (4), got 2}}
    %0 = "ttir.paged_fill_cache"(%cache, %input, %page_table, %batch_idx)
        : (tensor<128x12x32x256xbf16>, tensor<4x12x65x256xbf16>,
           tensor<8x16xi32>, tensor<2xi32>)
        -> tensor<128x12x32x256xbf16>
    return %0 : tensor<128x12x32x256xbf16>
  }
}

// -----

// input_batch > 1 without a batch_idx_tensor must be rejected: the lowering
// hard-codes batch_idx=0 and can only address one page-table row.
module {
  func.func @paged_fill_cache_multi_batch_without_idx_tensor(
      %cache: tensor<128x12x32x256xbf16>,
      %input: tensor<4x12x65x256xbf16>,
      %page_table: tensor<8x16xi32>) -> tensor<128x12x32x256xbf16> {
    // expected-error @+1 {{'ttir.paged_fill_cache' op Input batch must be statically 1 when no batch_idx_tensor is provided, got 4}}
    %0 = "ttir.paged_fill_cache"(%cache, %input, %page_table)
        : (tensor<128x12x32x256xbf16>, tensor<4x12x65x256xbf16>,
           tensor<8x16xi32>)
        -> tensor<128x12x32x256xbf16>
    return %0 : tensor<128x12x32x256xbf16>
  }
}

// -----

// Dynamic input batch without a batch_idx_tensor must also be rejected:
// the dim could resolve to >1 after shape inference, and the lowering
// would still hard-code batch_idx=0.
module {
  func.func @paged_fill_cache_dynamic_batch_without_idx_tensor(
      %cache: tensor<128x12x32x256xbf16>,
      %input: tensor<?x12x65x256xbf16>,
      %page_table: tensor<8x16xi32>) -> tensor<128x12x32x256xbf16> {
    // expected-error @+1 {{'ttir.paged_fill_cache' op Input batch must be statically 1 when no batch_idx_tensor is provided, got dynamic}}
    %0 = "ttir.paged_fill_cache"(%cache, %input, %page_table)
        : (tensor<128x12x32x256xbf16>, tensor<?x12x65x256xbf16>,
           tensor<8x16xi32>)
        -> tensor<128x12x32x256xbf16>
    return %0 : tensor<128x12x32x256xbf16>
  }
}

// -----

// batch_idx_tensor must be 1-D.
module {
  func.func @paged_fill_cache_batch_idx_wrong_rank(
      %cache: tensor<128x12x32x256xbf16>,
      %input: tensor<1x12x65x256xbf16>,
      %page_table: tensor<8x16xi32>,
      %batch_idx: tensor<1x1xi32>) -> tensor<128x12x32x256xbf16> {
    // expected-error @+1 {{'ttir.paged_fill_cache' op Batch index tensor must be a 1D tensor}}
    %0 = "ttir.paged_fill_cache"(%cache, %input, %page_table, %batch_idx)
        : (tensor<128x12x32x256xbf16>, tensor<1x12x65x256xbf16>,
           tensor<8x16xi32>, tensor<1x1xi32>)
        -> tensor<128x12x32x256xbf16>
    return %0 : tensor<128x12x32x256xbf16>
  }
}

// -----

// batch_idx_tensor must be an integer type.
module {
  func.func @paged_fill_cache_batch_idx_wrong_dtype(
      %cache: tensor<128x12x32x256xbf16>,
      %input: tensor<1x12x65x256xbf16>,
      %page_table: tensor<8x16xi32>,
      %batch_idx: tensor<1xbf16>) -> tensor<128x12x32x256xbf16> {
    // expected-error @+1 {{'ttir.paged_fill_cache' op Batch index tensor must be an integer type}}
    %0 = "ttir.paged_fill_cache"(%cache, %input, %page_table, %batch_idx)
        : (tensor<128x12x32x256xbf16>, tensor<1x12x65x256xbf16>,
           tensor<8x16xi32>, tensor<1xbf16>)
        -> tensor<128x12x32x256xbf16>
    return %0 : tensor<128x12x32x256xbf16>
  }
}
