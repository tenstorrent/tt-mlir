// RUN: not ttmlir-opt --split-input-file %s 2>&1 | FileCheck %s

// Negative tests for the ttir.flash_mla_prefill verifier.

// Verify that the verifier fails when query is not 4D.
module {
  func.func @flash_mla_prefill_query_rank(%query: tensor<16x32x128xbf16>, %key: tensor<1x1x32x128xbf16>) -> tensor<1x16x32x64xbf16> {
    // CHECK: error: 'ttir.flash_mla_prefill' op Query must be a 4D tensor
    %0 = "ttir.flash_mla_prefill"(%query, %key) <{operandSegmentSizes = array<i32: 1, 1, 0, 0>, head_dim_v = 64 : ui32, is_causal = true}> : (tensor<16x32x128xbf16>, tensor<1x1x32x128xbf16>) -> tensor<1x16x32x64xbf16>
    return %0 : tensor<1x16x32x64xbf16>
  }
}

// -----

// Verify that the verifier fails when key is not 4D.
module {
  func.func @flash_mla_prefill_key_rank(%query: tensor<1x16x32x128xbf16>, %key: tensor<1x32x128xbf16>) -> tensor<1x16x32x64xbf16> {
    // CHECK: error: 'ttir.flash_mla_prefill' op Key must be a 4D tensor
    %0 = "ttir.flash_mla_prefill"(%query, %key) <{operandSegmentSizes = array<i32: 1, 1, 0, 0>, head_dim_v = 64 : ui32, is_causal = true}> : (tensor<1x16x32x128xbf16>, tensor<1x32x128xbf16>) -> tensor<1x16x32x64xbf16>
    return %0 : tensor<1x16x32x64xbf16>
  }
}

// -----

// Verify that the verifier fails when result is not 4D.
module {
  func.func @flash_mla_prefill_result_rank(%query: tensor<1x16x32x128xbf16>, %key: tensor<1x1x32x128xbf16>) -> tensor<16x32x64xbf16> {
    // CHECK: error: 'ttir.flash_mla_prefill' op Output must be a 4D tensor
    %0 = "ttir.flash_mla_prefill"(%query, %key) <{operandSegmentSizes = array<i32: 1, 1, 0, 0>, head_dim_v = 64 : ui32, is_causal = true}> : (tensor<1x16x32x128xbf16>, tensor<1x1x32x128xbf16>) -> tensor<16x32x64xbf16>
    return %0 : tensor<16x32x64xbf16>
  }
}

// -----

// Verify that the verifier fails when key batch size differs from query batch.
module {
  func.func @flash_mla_prefill_key_batch(%query: tensor<1x16x32x128xbf16>, %key: tensor<2x1x32x128xbf16>) -> tensor<1x16x32x64xbf16> {
    // CHECK: error: 'ttir.flash_mla_prefill' op Key batch size must match query batch size
    %0 = "ttir.flash_mla_prefill"(%query, %key) <{operandSegmentSizes = array<i32: 1, 1, 0, 0>, head_dim_v = 64 : ui32, is_causal = true}> : (tensor<1x16x32x128xbf16>, tensor<2x1x32x128xbf16>) -> tensor<1x16x32x64xbf16>
    return %0 : tensor<1x16x32x64xbf16>
  }
}

// -----

// Verify that the verifier fails when key head size differs from query head size.
module {
  func.func @flash_mla_prefill_key_head_size(%query: tensor<1x16x32x128xbf16>, %key: tensor<1x1x32x64xbf16>) -> tensor<1x16x32x64xbf16> {
    // CHECK: error: 'ttir.flash_mla_prefill' op Key head size must match query head size
    %0 = "ttir.flash_mla_prefill"(%query, %key) <{operandSegmentSizes = array<i32: 1, 1, 0, 0>, head_dim_v = 64 : ui32, is_causal = true}> : (tensor<1x16x32x128xbf16>, tensor<1x1x32x64xbf16>) -> tensor<1x16x32x64xbf16>
    return %0 : tensor<1x16x32x64xbf16>
  }
}

// -----

// Verify that the verifier fails when query and key sequence lengths differ.
module {
  func.func @flash_mla_prefill_seq_len(%query: tensor<1x16x32x128xbf16>, %key: tensor<1x1x16x128xbf16>) -> tensor<1x16x32x64xbf16> {
    // CHECK: error: 'ttir.flash_mla_prefill' op Query and key must have the same sequence length for flash MLA prefill
    %0 = "ttir.flash_mla_prefill"(%query, %key) <{operandSegmentSizes = array<i32: 1, 1, 0, 0>, head_dim_v = 64 : ui32, is_causal = true}> : (tensor<1x16x32x128xbf16>, tensor<1x1x16x128xbf16>) -> tensor<1x16x32x64xbf16>
    return %0 : tensor<1x16x32x64xbf16>
  }
}

// -----

// Verify that the verifier fails when query num heads is not divisible by key num heads.
module {
  func.func @flash_mla_prefill_heads_divisible(%query: tensor<1x15x32x128xbf16>, %key: tensor<1x4x32x128xbf16>) -> tensor<1x15x32x64xbf16> {
    // CHECK: error: 'ttir.flash_mla_prefill' op Query num heads must be divisible by key num heads (GQA/MQA/MLA)
    %0 = "ttir.flash_mla_prefill"(%query, %key) <{operandSegmentSizes = array<i32: 1, 1, 0, 0>, head_dim_v = 64 : ui32, is_causal = true}> : (tensor<1x15x32x128xbf16>, tensor<1x4x32x128xbf16>) -> tensor<1x15x32x64xbf16>
    return %0 : tensor<1x15x32x64xbf16>
  }
}

// -----

// Verify that the verifier fails when value is not 4D.
module {
  func.func @flash_mla_prefill_value_rank(%query: tensor<1x16x32x128xbf16>, %key: tensor<1x1x32x128xbf16>, %value: tensor<1x32x64xbf16>) -> tensor<1x16x32x64xbf16> {
    // CHECK: error: 'ttir.flash_mla_prefill' op Value must be a 4D tensor
    %0 = "ttir.flash_mla_prefill"(%query, %key, %value) <{operandSegmentSizes = array<i32: 1, 1, 1, 0>, head_dim_v = 64 : ui32, is_causal = true}> : (tensor<1x16x32x128xbf16>, tensor<1x1x32x128xbf16>, tensor<1x32x64xbf16>) -> tensor<1x16x32x64xbf16>
    return %0 : tensor<1x16x32x64xbf16>
  }
}

// -----

// Verify that the verifier fails when value batch size differs from query batch.
module {
  func.func @flash_mla_prefill_value_batch(%query: tensor<1x16x32x128xbf16>, %key: tensor<1x1x32x128xbf16>, %value: tensor<2x1x32x64xbf16>) -> tensor<1x16x32x64xbf16> {
    // CHECK: error: 'ttir.flash_mla_prefill' op Value batch size must match query batch size
    %0 = "ttir.flash_mla_prefill"(%query, %key, %value) <{operandSegmentSizes = array<i32: 1, 1, 1, 0>, head_dim_v = 64 : ui32, is_causal = true}> : (tensor<1x16x32x128xbf16>, tensor<1x1x32x128xbf16>, tensor<2x1x32x64xbf16>) -> tensor<1x16x32x64xbf16>
    return %0 : tensor<1x16x32x64xbf16>
  }
}

// -----

// Verify that the verifier fails when value num heads differs from key num heads.
module {
  func.func @flash_mla_prefill_value_heads(%query: tensor<1x16x32x128xbf16>, %key: tensor<1x1x32x128xbf16>, %value: tensor<1x2x32x64xbf16>) -> tensor<1x16x32x64xbf16> {
    // CHECK: error: 'ttir.flash_mla_prefill' op Value num heads must match key num heads
    %0 = "ttir.flash_mla_prefill"(%query, %key, %value) <{operandSegmentSizes = array<i32: 1, 1, 1, 0>, head_dim_v = 64 : ui32, is_causal = true}> : (tensor<1x16x32x128xbf16>, tensor<1x1x32x128xbf16>, tensor<1x2x32x64xbf16>) -> tensor<1x16x32x64xbf16>
    return %0 : tensor<1x16x32x64xbf16>
  }
}

// -----

// Verify that the verifier fails when value sequence length differs from query/key.
module {
  func.func @flash_mla_prefill_value_seq(%query: tensor<1x16x32x128xbf16>, %key: tensor<1x1x32x128xbf16>, %value: tensor<1x1x16x64xbf16>) -> tensor<1x16x32x64xbf16> {
    // CHECK: error: 'ttir.flash_mla_prefill' op Value sequence length must match query/key sequence length
    %0 = "ttir.flash_mla_prefill"(%query, %key, %value) <{operandSegmentSizes = array<i32: 1, 1, 1, 0>, head_dim_v = 64 : ui32, is_causal = true}> : (tensor<1x16x32x128xbf16>, tensor<1x1x32x128xbf16>, tensor<1x1x16x64xbf16>) -> tensor<1x16x32x64xbf16>
    return %0 : tensor<1x16x32x64xbf16>
  }
}

// -----

// Verify that the verifier fails when value head_dim != head_dim_v.
module {
  func.func @flash_mla_prefill_value_head_dim(%query: tensor<1x16x32x128xbf16>, %key: tensor<1x1x32x128xbf16>, %value: tensor<1x1x32x32xbf16>) -> tensor<1x16x32x64xbf16> {
    // CHECK: error: 'ttir.flash_mla_prefill' op Value head size must equal head_dim_v
    %0 = "ttir.flash_mla_prefill"(%query, %key, %value) <{operandSegmentSizes = array<i32: 1, 1, 1, 0>, head_dim_v = 64 : ui32, is_causal = true}> : (tensor<1x16x32x128xbf16>, tensor<1x1x32x128xbf16>, tensor<1x1x32x32xbf16>) -> tensor<1x16x32x64xbf16>
    return %0 : tensor<1x16x32x64xbf16>
  }
}

// -----

// Verify that the verifier fails when head_dim_v exceeds key head dim and value is absent.
module {
  func.func @flash_mla_prefill_head_dim_v_too_large(%query: tensor<1x16x32x128xbf16>, %key: tensor<1x1x32x128xbf16>) -> tensor<1x16x32x256xbf16> {
    // CHECK: error: 'ttir.flash_mla_prefill' op head_dim_v cannot exceed key's head dim when value is not provided
    %0 = "ttir.flash_mla_prefill"(%query, %key) <{operandSegmentSizes = array<i32: 1, 1, 0, 0>, head_dim_v = 256 : ui32, is_causal = true}> : (tensor<1x16x32x128xbf16>, tensor<1x1x32x128xbf16>) -> tensor<1x16x32x256xbf16>
    return %0 : tensor<1x16x32x256xbf16>
  }
}

// -----

// Verify that the verifier fails when an attention mask is provided with is_causal=true.
module {
  func.func @flash_mla_prefill_mask_with_causal(%query: tensor<1x16x32x128xbf16>, %key: tensor<1x1x32x128xbf16>, %mask: tensor<1x1x32x32xbf16>) -> tensor<1x16x32x64xbf16> {
    // CHECK: error: 'ttir.flash_mla_prefill' op Attention mask is not allowed when is_causal is true
    %0 = "ttir.flash_mla_prefill"(%query, %key, %mask) <{operandSegmentSizes = array<i32: 1, 1, 0, 1>, head_dim_v = 64 : ui32, is_causal = true}> : (tensor<1x16x32x128xbf16>, tensor<1x1x32x128xbf16>, tensor<1x1x32x32xbf16>) -> tensor<1x16x32x64xbf16>
    return %0 : tensor<1x16x32x64xbf16>
  }
}

// -----

// Verify that the verifier fails when attention mask is not 4D.
module {
  func.func @flash_mla_prefill_mask_rank(%query: tensor<1x16x32x128xbf16>, %key: tensor<1x1x32x128xbf16>, %mask: tensor<1x32x32xbf16>) -> tensor<1x16x32x64xbf16> {
    // CHECK: error: 'ttir.flash_mla_prefill' op Attention mask must be a 4D tensor
    %0 = "ttir.flash_mla_prefill"(%query, %key, %mask) <{operandSegmentSizes = array<i32: 1, 1, 0, 1>, head_dim_v = 64 : ui32, is_causal = false}> : (tensor<1x16x32x128xbf16>, tensor<1x1x32x128xbf16>, tensor<1x32x32xbf16>) -> tensor<1x16x32x64xbf16>
    return %0 : tensor<1x16x32x64xbf16>
  }
}

// -----

// Verify that the verifier fails when mask batch is neither 1 nor B.
module {
  func.func @flash_mla_prefill_mask_batch(%query: tensor<1x16x32x128xbf16>, %key: tensor<1x1x32x128xbf16>, %mask: tensor<3x1x32x32xbf16>) -> tensor<1x16x32x64xbf16> {
    // CHECK: error: 'ttir.flash_mla_prefill' op Attention mask batch size must be 1 (broadcast) or match query batch size
    %0 = "ttir.flash_mla_prefill"(%query, %key, %mask) <{operandSegmentSizes = array<i32: 1, 1, 0, 1>, head_dim_v = 64 : ui32, is_causal = false}> : (tensor<1x16x32x128xbf16>, tensor<1x1x32x128xbf16>, tensor<3x1x32x32xbf16>) -> tensor<1x16x32x64xbf16>
    return %0 : tensor<1x16x32x64xbf16>
  }
}

// -----

// Verify that the verifier fails when mask dim 1 is neither 1 nor Hq.
module {
  func.func @flash_mla_prefill_mask_dim1(%query: tensor<1x16x32x128xbf16>, %key: tensor<1x1x32x128xbf16>, %mask: tensor<1x8x32x32xbf16>) -> tensor<1x16x32x64xbf16> {
    // CHECK: error: 'ttir.flash_mla_prefill' op Attention mask dim 1 must be 1 (broadcast) or match query num heads
    %0 = "ttir.flash_mla_prefill"(%query, %key, %mask) <{operandSegmentSizes = array<i32: 1, 1, 0, 1>, head_dim_v = 64 : ui32, is_causal = false}> : (tensor<1x16x32x128xbf16>, tensor<1x1x32x128xbf16>, tensor<1x8x32x32xbf16>) -> tensor<1x16x32x64xbf16>
    return %0 : tensor<1x16x32x64xbf16>
  }
}

// -----

// Verify that the verifier fails when mask dim 2 != query seq_len.
module {
  func.func @flash_mla_prefill_mask_dim2(%query: tensor<1x16x32x128xbf16>, %key: tensor<1x1x32x128xbf16>, %mask: tensor<1x1x16x32xbf16>) -> tensor<1x16x32x64xbf16> {
    // CHECK: error: 'ttir.flash_mla_prefill' op Attention mask at dim 2 must match query sequence length
    %0 = "ttir.flash_mla_prefill"(%query, %key, %mask) <{operandSegmentSizes = array<i32: 1, 1, 0, 1>, head_dim_v = 64 : ui32, is_causal = false}> : (tensor<1x16x32x128xbf16>, tensor<1x1x32x128xbf16>, tensor<1x1x16x32xbf16>) -> tensor<1x16x32x64xbf16>
    return %0 : tensor<1x16x32x64xbf16>
  }
}

// -----

// Verify that the verifier fails when mask dim 3 != key seq_len.
module {
  func.func @flash_mla_prefill_mask_dim3(%query: tensor<1x16x32x128xbf16>, %key: tensor<1x1x32x128xbf16>, %mask: tensor<1x1x32x16xbf16>) -> tensor<1x16x32x64xbf16> {
    // CHECK: error: 'ttir.flash_mla_prefill' op Attention mask at dim 3 must match key sequence length
    %0 = "ttir.flash_mla_prefill"(%query, %key, %mask) <{operandSegmentSizes = array<i32: 1, 1, 0, 1>, head_dim_v = 64 : ui32, is_causal = false}> : (tensor<1x16x32x128xbf16>, tensor<1x1x32x128xbf16>, tensor<1x1x32x16xbf16>) -> tensor<1x16x32x64xbf16>
    return %0 : tensor<1x16x32x64xbf16>
  }
}

// -----

// Verify that the verifier fails when result shape does not match [B, Hq, Sq, head_dim_v].
module {
  func.func @flash_mla_prefill_result_shape(%query: tensor<1x16x32x128xbf16>, %key: tensor<1x1x32x128xbf16>) -> tensor<1x8x32x64xbf16> {
    // CHECK: error: 'ttir.flash_mla_prefill' op Result shape must be [batch, num_query_heads, seq_len, head_dim_v]
    %0 = "ttir.flash_mla_prefill"(%query, %key) <{operandSegmentSizes = array<i32: 1, 1, 0, 0>, head_dim_v = 64 : ui32, is_causal = true}> : (tensor<1x16x32x128xbf16>, tensor<1x1x32x128xbf16>) -> tensor<1x8x32x64xbf16>
    return %0 : tensor<1x8x32x64xbf16>
  }
}

// -----

// Verify that the verifier fails when result element type does not match query element type.
module {
  func.func @flash_mla_prefill_result_dtype(%query: tensor<1x16x32x128xbf16>, %key: tensor<1x1x32x128xbf16>) -> tensor<1x16x32x64xf32> {
    // CHECK: error: 'ttir.flash_mla_prefill' op Result element type must match query element type
    %0 = "ttir.flash_mla_prefill"(%query, %key) <{operandSegmentSizes = array<i32: 1, 1, 0, 0>, head_dim_v = 64 : ui32, is_causal = true}> : (tensor<1x16x32x128xbf16>, tensor<1x1x32x128xbf16>) -> tensor<1x16x32x64xf32>
    return %0 : tensor<1x16x32x64xf32>
  }
}
