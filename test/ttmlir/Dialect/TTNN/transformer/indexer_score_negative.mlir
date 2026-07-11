// RUN: not ttmlir-opt --split-input-file %s 2>&1 | FileCheck %s

// Negative tests for the ttnn.indexer_score verifier.

// Key must have a single head (dim 1 must be 1).
module {
  func.func @indexer_score_key_multi_head(%q: tensor<1x8x32x128xbf16>, %k: tensor<1x2x32x128xbf16>, %w: tensor<1x8x32x1xbf16>) -> tensor<1x1x32x32xbf16> {
    // CHECK: error: 'ttnn.indexer_score' op Key must have a single head (dim 1 must be 1)
    %0 = "ttnn.indexer_score"(%q, %k, %w) <{chunk_start_idx = 0 : ui32}> : (tensor<1x8x32x128xbf16>, tensor<1x2x32x128xbf16>, tensor<1x8x32x1xbf16>) -> tensor<1x1x32x32xbf16>
    return %0 : tensor<1x1x32x32xbf16>
  }
}

// -----

// Weights shape must be [batch, num_heads, query_seq_len, 1].
module {
  func.func @indexer_score_bad_weights(%q: tensor<1x8x32x128xbf16>, %k: tensor<1x1x32x128xbf16>, %w: tensor<1x8x32x4xbf16>) -> tensor<1x1x32x32xbf16> {
    // CHECK: error: 'ttnn.indexer_score' op Weights shape must be [batch, num_heads, query_seq_len, 1]
    %0 = "ttnn.indexer_score"(%q, %k, %w) <{chunk_start_idx = 0 : ui32}> : (tensor<1x8x32x128xbf16>, tensor<1x1x32x128xbf16>, tensor<1x8x32x4xbf16>) -> tensor<1x1x32x32xbf16>
    return %0 : tensor<1x1x32x32xbf16>
  }
}

// -----

// Result shape must be [batch, 1, query_seq_len, key_seq_len].
module {
  func.func @indexer_score_bad_result(%q: tensor<1x8x32x128xbf16>, %k: tensor<1x1x32x128xbf16>, %w: tensor<1x8x32x1xbf16>) -> tensor<1x1x32x64xbf16> {
    // CHECK: error: 'ttnn.indexer_score' op Result shape must be [batch, 1, query_seq_len, key_seq_len]
    %0 = "ttnn.indexer_score"(%q, %k, %w) <{chunk_start_idx = 0 : ui32}> : (tensor<1x8x32x128xbf16>, tensor<1x1x32x128xbf16>, tensor<1x8x32x1xbf16>) -> tensor<1x1x32x64xbf16>
    return %0 : tensor<1x1x32x64xbf16>
  }
}

// -----

// Query and key must have the same element type.
module {
  func.func @indexer_score_dtype_mismatch(%q: tensor<1x8x32x128xbf16>, %k: tensor<1x1x32x128xf32>, %w: tensor<1x8x32x1xbf16>) -> tensor<1x1x32x32xbf16> {
    // CHECK: error: 'ttnn.indexer_score' op Query and key must have the same element type
    %0 = "ttnn.indexer_score"(%q, %k, %w) <{chunk_start_idx = 0 : ui32}> : (tensor<1x8x32x128xbf16>, tensor<1x1x32x128xf32>, tensor<1x8x32x1xbf16>) -> tensor<1x1x32x32xbf16>
    return %0 : tensor<1x1x32x32xbf16>
  }
}

// -----

// Batch size (dim 0) must be 1: the op is Blackhole-only and single-batch, so a
// batch > 1 must be decomposed instead of promoted to the typed op.
module {
  func.func @indexer_score_batch_gt_one(%q: tensor<2x8x32x128xbf16>, %k: tensor<2x1x32x128xbf16>, %w: tensor<2x8x32x1xbf16>) -> tensor<2x1x32x32xbf16> {
    // CHECK: error: 'ttnn.indexer_score' op Query batch size (dim 0) must be 1, got 2
    %0 = "ttnn.indexer_score"(%q, %k, %w) <{chunk_start_idx = 0 : ui32}> : (tensor<2x8x32x128xbf16>, tensor<2x1x32x128xbf16>, tensor<2x8x32x1xbf16>) -> tensor<2x1x32x32xbf16>
    return %0 : tensor<2x1x32x32xbf16>
  }
}

// -----

// Query must be a 4D tensor.
module {
  func.func @indexer_score_query_rank(%q: tensor<8x32x128xbf16>, %k: tensor<1x1x32x128xbf16>, %w: tensor<1x8x32x1xbf16>) -> tensor<1x1x32x32xbf16> {
    // CHECK: error: 'ttnn.indexer_score' op Query must be a 4D tensor [B, Hi, Sq, D]
    %0 = "ttnn.indexer_score"(%q, %k, %w) <{chunk_start_idx = 0 : ui32}> : (tensor<8x32x128xbf16>, tensor<1x1x32x128xbf16>, tensor<1x8x32x1xbf16>) -> tensor<1x1x32x32xbf16>
    return %0 : tensor<1x1x32x32xbf16>
  }
}

// -----

// Key must be a 4D tensor.
module {
  func.func @indexer_score_key_rank(%q: tensor<1x8x32x128xbf16>, %k: tensor<1x32x128xbf16>, %w: tensor<1x8x32x1xbf16>) -> tensor<1x1x32x32xbf16> {
    // CHECK: error: 'ttnn.indexer_score' op Key must be a 4D tensor [B, 1, T, D]
    %0 = "ttnn.indexer_score"(%q, %k, %w) <{chunk_start_idx = 0 : ui32}> : (tensor<1x8x32x128xbf16>, tensor<1x32x128xbf16>, tensor<1x8x32x1xbf16>) -> tensor<1x1x32x32xbf16>
    return %0 : tensor<1x1x32x32xbf16>
  }
}

// -----

// Key batch size must match query batch size.
module {
  func.func @indexer_score_key_batch(%q: tensor<1x8x32x128xbf16>, %k: tensor<2x1x32x128xbf16>, %w: tensor<1x8x32x1xbf16>) -> tensor<1x1x32x32xbf16> {
    // CHECK: error: 'ttnn.indexer_score' op Key batch size must match query batch size
    %0 = "ttnn.indexer_score"(%q, %k, %w) <{chunk_start_idx = 0 : ui32}> : (tensor<1x8x32x128xbf16>, tensor<2x1x32x128xbf16>, tensor<1x8x32x1xbf16>) -> tensor<1x1x32x32xbf16>
    return %0 : tensor<1x1x32x32xbf16>
  }
}

// -----

// Key head dim must match query head dim.
module {
  func.func @indexer_score_key_head_dim(%q: tensor<1x8x32x128xbf16>, %k: tensor<1x1x32x64xbf16>, %w: tensor<1x8x32x1xbf16>) -> tensor<1x1x32x32xbf16> {
    // CHECK: error: 'ttnn.indexer_score' op Key head dim must match query head dim
    %0 = "ttnn.indexer_score"(%q, %k, %w) <{chunk_start_idx = 0 : ui32}> : (tensor<1x8x32x128xbf16>, tensor<1x1x32x64xbf16>, tensor<1x8x32x1xbf16>) -> tensor<1x1x32x32xbf16>
    return %0 : tensor<1x1x32x32xbf16>
  }
}
