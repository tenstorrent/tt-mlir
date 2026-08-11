// RUN: not ttmlir-opt --split-input-file %s 2>&1 | FileCheck %s

// Negative tests for the ttnn.sparse_sdpa verifier.

// Batch size (dim 0) must be 1: the metal kernel is Blackhole-only and
// single-batch, so a batch > 1 must be decomposed instead of promoted.
module {
  func.func @sparse_sdpa_batch_gt_one(%q: tensor<2x32x32x64xbf16>, %kv: tensor<2x1x64x64xbf16>, %idx: tensor<2x1x32x32xui32>) -> tensor<2x32x32x32xbf16> {
    // CHECK: error: 'ttnn.sparse_sdpa' op Query batch size (dim 0) must be 1, got 2
    %0 = "ttnn.sparse_sdpa"(%q, %kv, %idx) <{v_dim = 32 : ui32, k_chunk_size = 32 : ui32}> : (tensor<2x32x32x64xbf16>, tensor<2x1x64x64xbf16>, tensor<2x1x32x32xui32>) -> tensor<2x32x32x32xbf16>
    return %0 : tensor<2x32x32x32xbf16>
  }
}

// -----

// Query head count must be a multiple of the 32-row tile.
module {
  func.func @sparse_sdpa_bad_head_count(%q: tensor<1x16x32x64xbf16>, %kv: tensor<1x1x64x64xbf16>, %idx: tensor<1x1x32x32xui32>) -> tensor<1x16x32x32xbf16> {
    // CHECK: error: 'ttnn.sparse_sdpa' op Query head count (dim 1) must be a positive multiple of 32, got 16
    %0 = "ttnn.sparse_sdpa"(%q, %kv, %idx) <{v_dim = 32 : ui32, k_chunk_size = 32 : ui32}> : (tensor<1x16x32x64xbf16>, tensor<1x1x64x64xbf16>, tensor<1x1x32x32xui32>) -> tensor<1x16x32x32xbf16>
    return %0 : tensor<1x16x32x32xbf16>
  }
}

// -----

// Query must be a 4D tensor.
module {
  func.func @sparse_sdpa_query_rank(%q: tensor<32x32x64xbf16>, %kv: tensor<1x1x64x64xbf16>, %idx: tensor<1x1x32x32xui32>) -> tensor<1x32x32x32xbf16> {
    // CHECK: error: 'ttnn.sparse_sdpa' op Query must be a 4D tensor [1, H, S, K_DIM]
    %0 = "ttnn.sparse_sdpa"(%q, %kv, %idx) <{v_dim = 32 : ui32, k_chunk_size = 32 : ui32}> : (tensor<32x32x64xbf16>, tensor<1x1x64x64xbf16>, tensor<1x1x32x32xui32>) -> tensor<1x32x32x32xbf16>
    return %0 : tensor<1x32x32x32xbf16>
  }
}

// -----

// Kv must have a single latent head (dim 1 must be 1).
module {
  func.func @sparse_sdpa_kv_multi_head(%q: tensor<1x32x32x64xbf16>, %kv: tensor<1x2x64x64xbf16>, %idx: tensor<1x1x32x32xui32>) -> tensor<1x32x32x32xbf16> {
    // CHECK: error: 'ttnn.sparse_sdpa' op Kv must have a single head (dim 1 must be 1)
    %0 = "ttnn.sparse_sdpa"(%q, %kv, %idx) <{v_dim = 32 : ui32, k_chunk_size = 32 : ui32}> : (tensor<1x32x32x64xbf16>, tensor<1x2x64x64xbf16>, tensor<1x1x32x32xui32>) -> tensor<1x32x32x32xbf16>
    return %0 : tensor<1x32x32x32xbf16>
  }
}

// -----

// Kv head dim must match the query head dim.
module {
  func.func @sparse_sdpa_kv_head_dim(%q: tensor<1x32x32x64xbf16>, %kv: tensor<1x1x64x32xbf16>, %idx: tensor<1x1x32x32xui32>) -> tensor<1x32x32x32xbf16> {
    // CHECK: error: 'ttnn.sparse_sdpa' op Kv head dim must match query head dim
    %0 = "ttnn.sparse_sdpa"(%q, %kv, %idx) <{v_dim = 32 : ui32, k_chunk_size = 32 : ui32}> : (tensor<1x32x32x64xbf16>, tensor<1x1x64x32xbf16>, tensor<1x1x32x32xui32>) -> tensor<1x32x32x32xbf16>
    return %0 : tensor<1x32x32x32xbf16>
  }
}

// -----

// Indices shape must be [batch, 1, query_seq_len, top_k].
module {
  func.func @sparse_sdpa_bad_indices(%q: tensor<1x32x32x64xbf16>, %kv: tensor<1x1x64x64xbf16>, %idx: tensor<1x1x64x32xui32>) -> tensor<1x32x32x32xbf16> {
    // CHECK: error: 'ttnn.sparse_sdpa' op Indices shape must be [batch, 1, query_seq_len, top_k]
    %0 = "ttnn.sparse_sdpa"(%q, %kv, %idx) <{v_dim = 32 : ui32, k_chunk_size = 32 : ui32}> : (tensor<1x32x32x64xbf16>, tensor<1x1x64x64xbf16>, tensor<1x1x64x32xui32>) -> tensor<1x32x32x32xbf16>
    return %0 : tensor<1x32x32x32xbf16>
  }
}

// -----

// Indices must be an integer tensor (the metal kernel requires uint32).
module {
  func.func @sparse_sdpa_float_indices(%q: tensor<1x32x32x64xbf16>, %kv: tensor<1x1x64x64xbf16>, %idx: tensor<1x1x32x32xf32>) -> tensor<1x32x32x32xbf16> {
    // CHECK: error: 'ttnn.sparse_sdpa' op Indices must have an integer element type
    %0 = "ttnn.sparse_sdpa"(%q, %kv, %idx) <{v_dim = 32 : ui32, k_chunk_size = 32 : ui32}> : (tensor<1x32x32x64xbf16>, tensor<1x1x64x64xbf16>, tensor<1x1x32x32xf32>) -> tensor<1x32x32x32xbf16>
    return %0 : tensor<1x32x32x32xbf16>
  }
}

// -----

// v_dim must not exceed the latent width K_DIM.
module {
  func.func @sparse_sdpa_v_dim_too_large(%q: tensor<1x32x32x64xbf16>, %kv: tensor<1x1x64x64xbf16>, %idx: tensor<1x1x32x32xui32>) -> tensor<1x32x32x128xbf16> {
    // CHECK: error: 'ttnn.sparse_sdpa' op v_dim must be in (0, K_DIM=64], got 128
    %0 = "ttnn.sparse_sdpa"(%q, %kv, %idx) <{v_dim = 128 : ui32, k_chunk_size = 32 : ui32}> : (tensor<1x32x32x64xbf16>, tensor<1x1x64x64xbf16>, tensor<1x1x32x32xui32>) -> tensor<1x32x32x128xbf16>
    return %0 : tensor<1x32x32x128xbf16>
  }
}

// -----

// v_dim must be a multiple of the 32-column tile.
module {
  func.func @sparse_sdpa_v_dim_unaligned(%q: tensor<1x32x32x64xbf16>, %kv: tensor<1x1x64x64xbf16>, %idx: tensor<1x1x32x32xui32>) -> tensor<1x32x32x48xbf16> {
    // CHECK: error: 'ttnn.sparse_sdpa' op v_dim must be a multiple of 32, got 48
    %0 = "ttnn.sparse_sdpa"(%q, %kv, %idx) <{v_dim = 48 : ui32, k_chunk_size = 32 : ui32}> : (tensor<1x32x32x64xbf16>, tensor<1x1x64x64xbf16>, tensor<1x1x32x32xui32>) -> tensor<1x32x32x48xbf16>
    return %0 : tensor<1x32x32x48xbf16>
  }
}

// -----

// k_chunk_size must be a multiple of the 32-column tile.
module {
  func.func @sparse_sdpa_bad_k_chunk_size(%q: tensor<1x32x32x64xbf16>, %kv: tensor<1x1x64x64xbf16>, %idx: tensor<1x1x32x32xui32>) -> tensor<1x32x32x32xbf16> {
    // CHECK: error: 'ttnn.sparse_sdpa' op k_chunk_size must be a positive multiple of 32, got 16
    %0 = "ttnn.sparse_sdpa"(%q, %kv, %idx) <{v_dim = 32 : ui32, k_chunk_size = 16 : ui32}> : (tensor<1x32x32x64xbf16>, tensor<1x1x64x64xbf16>, tensor<1x1x32x32xui32>) -> tensor<1x32x32x32xbf16>
    return %0 : tensor<1x32x32x32xbf16>
  }
}

// -----

// k_chunk_size must divide TOPK.
module {
  func.func @sparse_sdpa_k_chunk_not_dividing_topk(%q: tensor<1x32x32x64xbf16>, %kv: tensor<1x1x64x64xbf16>, %idx: tensor<1x1x32x32xui32>) -> tensor<1x32x32x32xbf16> {
    // CHECK: error: 'ttnn.sparse_sdpa' op k_chunk_size (64) must divide top_k (32)
    %0 = "ttnn.sparse_sdpa"(%q, %kv, %idx) <{v_dim = 32 : ui32, k_chunk_size = 64 : ui32}> : (tensor<1x32x32x64xbf16>, tensor<1x1x64x64xbf16>, tensor<1x1x32x32xui32>) -> tensor<1x32x32x32xbf16>
    return %0 : tensor<1x32x32x32xbf16>
  }
}

// -----

// Result shape must be [batch, num_heads, query_seq_len, v_dim].
module {
  func.func @sparse_sdpa_bad_result(%q: tensor<1x32x32x64xbf16>, %kv: tensor<1x1x64x64xbf16>, %idx: tensor<1x1x32x32xui32>) -> tensor<1x32x64x32xbf16> {
    // CHECK: error: 'ttnn.sparse_sdpa' op Result shape must be [batch, num_heads, query_seq_len, v_dim]
    %0 = "ttnn.sparse_sdpa"(%q, %kv, %idx) <{v_dim = 32 : ui32, k_chunk_size = 32 : ui32}> : (tensor<1x32x32x64xbf16>, tensor<1x1x64x64xbf16>, tensor<1x1x32x32xui32>) -> tensor<1x32x64x32xbf16>
    return %0 : tensor<1x32x64x32xbf16>
  }
}

// -----

// Query and kv must have the same element type.
module {
  func.func @sparse_sdpa_dtype_mismatch(%q: tensor<1x32x32x64xbf16>, %kv: tensor<1x1x64x64xf32>, %idx: tensor<1x1x32x32xui32>) -> tensor<1x32x32x32xbf16> {
    // CHECK: error: 'ttnn.sparse_sdpa' op Query and kv must have the same element type
    %0 = "ttnn.sparse_sdpa"(%q, %kv, %idx) <{v_dim = 32 : ui32, k_chunk_size = 32 : ui32}> : (tensor<1x32x32x64xbf16>, tensor<1x1x64x64xf32>, tensor<1x1x32x32xui32>) -> tensor<1x32x32x32xbf16>
    return %0 : tensor<1x32x32x32xbf16>
  }
}

// -----

// scale must be positive.
module {
  func.func @sparse_sdpa_bad_scale(%q: tensor<1x32x32x64xbf16>, %kv: tensor<1x1x64x64xbf16>, %idx: tensor<1x1x32x32xui32>) -> tensor<1x32x32x32xbf16> {
    // CHECK: error: 'ttnn.sparse_sdpa' op scale must be greater than 0
    %0 = "ttnn.sparse_sdpa"(%q, %kv, %idx) <{v_dim = 32 : ui32, k_chunk_size = 32 : ui32, scale = 0.000000e+00 : f32}> : (tensor<1x32x32x64xbf16>, tensor<1x1x64x64xbf16>, tensor<1x1x32x32xui32>) -> tensor<1x32x32x32xbf16>
    return %0 : tensor<1x32x32x32xbf16>
  }
}
