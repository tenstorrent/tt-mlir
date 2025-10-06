// RUN: not ttmlir-opt --split-input-file %s 2>&1 | FileCheck %s
// Negative tests for nlp_create_qkv_heads operation.

// Verify that the parsing fails if input_q tensor is not 4D.
module {
  func.func @nlp_create_qkv_heads_invalid_rank_input_q(%arg0: tensor<1x32x4608xbf16>) -> (tensor<1x24x32x64xbf16>, tensor<1x24x64x32xbf16>, tensor<1x24x32x64xbf16>) {
    // CHECK: error: 'ttnn.nlp_create_qkv_heads' op input_q tensor must be a 4D tensor
    %query, %key, %value = "ttnn.nlp_create_qkv_heads"(%arg0) {
      num_q_heads = 24 : ui32,
      transpose_k_heads = true
    } : (tensor<1x32x4608xbf16>) -> (tensor<1x24x32x64xbf16>, tensor<1x24x64x32xbf16>, tensor<1x24x32x64xbf16>)
    return %query, %key, %value : tensor<1x24x32x64xbf16>, tensor<1x24x64x32xbf16>, tensor<1x24x32x64xbf16>
  }
}

// -----

// Verify that the parsing fails if query tensor is not 4D.
module {
  func.func @nlp_create_qkv_heads_invalid_rank_query(%arg0: tensor<1x1x32x4608xbf16>) -> (tensor<24x32x64xbf16>, tensor<1x24x64x32xbf16>, tensor<1x24x32x64xbf16>) {
    // CHECK: error: 'ttnn.nlp_create_qkv_heads' op query tensor must be a 4D tensor
    %query, %key, %value = "ttnn.nlp_create_qkv_heads"(%arg0) {
      num_q_heads = 24 : ui32,
      transpose_k_heads = true
    } : (tensor<1x1x32x4608xbf16>) -> (tensor<24x32x64xbf16>, tensor<1x24x64x32xbf16>, tensor<1x24x32x64xbf16>)
    return %query, %key, %value : tensor<24x32x64xbf16>, tensor<1x24x64x32xbf16>, tensor<1x24x32x64xbf16>
  }
}

// -----

// Verify that the parsing fails if key tensor is not 4D.
module {
  func.func @nlp_create_qkv_heads_invalid_rank_key(%arg0: tensor<1x1x32x4608xbf16>) -> (tensor<1x24x32x64xbf16>, tensor<24x64x32xbf16>, tensor<1x24x32x64xbf16>) {
    // CHECK: error: 'ttnn.nlp_create_qkv_heads' op key tensor must be a 4D tensor
    %query, %key, %value = "ttnn.nlp_create_qkv_heads"(%arg0) {
      num_q_heads = 24 : ui32,
      transpose_k_heads = true
    } : (tensor<1x1x32x4608xbf16>) -> (tensor<1x24x32x64xbf16>, tensor<24x64x32xbf16>, tensor<1x24x32x64xbf16>)
    return %query, %key, %value : tensor<1x24x32x64xbf16>, tensor<24x64x32xbf16>, tensor<1x24x32x64xbf16>
  }
}

// -----

// Verify that the parsing fails if value tensor is not 4D.
module {
  func.func @nlp_create_qkv_heads_invalid_rank_value(%arg0: tensor<1x1x32x4608xbf16>) -> (tensor<1x24x32x64xbf16>, tensor<1x24x64x32xbf16>, tensor<24x32x64xbf16>) {
    // CHECK: error: 'ttnn.nlp_create_qkv_heads' op value tensor must be a 4D tensor
    %query, %key, %value = "ttnn.nlp_create_qkv_heads"(%arg0) {
      num_q_heads = 24 : ui32,
      transpose_k_heads = true
    } : (tensor<1x1x32x4608xbf16>) -> (tensor<1x24x32x64xbf16>, tensor<1x24x64x32xbf16>, tensor<24x32x64xbf16>)
    return %query, %key, %value : tensor<1x24x32x64xbf16>, tensor<1x24x64x32xbf16>, tensor<24x32x64xbf16>
  }
}

// -----

// Verify that the parsing fails if batch sizes don't match between input_q and query.
module {
  func.func @nlp_create_qkv_heads_batch_mismatch(%arg0: tensor<1x1x32x4608xbf16>) -> (tensor<2x24x32x64xbf16>, tensor<2x24x64x32xbf16>, tensor<2x24x32x64xbf16>) {
    // CHECK: error: 'ttnn.nlp_create_qkv_heads' op batch size must match between input_q and query
    %query, %key, %value = "ttnn.nlp_create_qkv_heads"(%arg0) {
      num_q_heads = 24 : ui32,
      transpose_k_heads = true
    } : (tensor<1x1x32x4608xbf16>) -> (tensor<2x24x32x64xbf16>, tensor<2x24x64x32xbf16>, tensor<2x24x32x64xbf16>)
    return %query, %key, %value : tensor<2x24x32x64xbf16>, tensor<2x24x64x32xbf16>, tensor<2x24x32x64xbf16>
  }
}

// -----

// Verify that the parsing fails if num_q_heads doesn't match query tensor dimension 1.
module {
  func.func @nlp_create_qkv_heads_wrong_num_heads(%arg0: tensor<1x1x32x4608xbf16>) -> (tensor<1x32x32x64xbf16>, tensor<1x32x64x32xbf16>, tensor<1x32x32x64xbf16>) {
    // CHECK: error: 'ttnn.nlp_create_qkv_heads' op query tensor dimension 1 must match num_q_heads
    %query, %key, %value = "ttnn.nlp_create_qkv_heads"(%arg0) {
      num_q_heads = 24 : ui32,
      transpose_k_heads = true
    } : (tensor<1x1x32x4608xbf16>) -> (tensor<1x32x32x64xbf16>, tensor<1x32x64x32xbf16>, tensor<1x32x32x64xbf16>)
    return %query, %key, %value : tensor<1x32x32x64xbf16>, tensor<1x32x64x32xbf16>, tensor<1x32x32x64xbf16>
  }
}

// -----

// Verify that the parsing fails if num_kv_heads doesn't match key tensor dimension 1.
module {
  func.func @nlp_create_qkv_heads_wrong_kv_heads(%arg0: tensor<1x1x32x1536xbf16>, %arg1: tensor<1x1x32x1024xbf16>) -> (tensor<1x24x32x64xbf16>, tensor<1x16x64x32xbf16>, tensor<1x16x32x64xbf16>) {
    // CHECK: error: 'ttnn.nlp_create_qkv_heads' op key tensor dimension 1 must match num_kv_heads
    %query, %key, %value = "ttnn.nlp_create_qkv_heads"(%arg0, %arg1) {
      num_q_heads = 24 : ui32,
      num_kv_heads = 8 : ui32,
      transpose_k_heads = true
    } : (tensor<1x1x32x1536xbf16>, tensor<1x1x32x1024xbf16>) -> (tensor<1x24x32x64xbf16>, tensor<1x16x64x32xbf16>, tensor<1x16x32x64xbf16>)
    return %query, %key, %value : tensor<1x24x32x64xbf16>, tensor<1x16x64x32xbf16>, tensor<1x16x32x64xbf16>
  }
}

// -----

// Verify that the parsing fails if sequence length doesn't match between input_q and query.
module {
  func.func @nlp_create_qkv_heads_seq_mismatch(%arg0: tensor<1x1x32x4608xbf16>) -> (tensor<1x24x64x64xbf16>, tensor<1x24x64x64xbf16>, tensor<1x24x64x64xbf16>) {
    // CHECK: error: 'ttnn.nlp_create_qkv_heads' op query sequence length must match input_q sequence
    %query, %key, %value = "ttnn.nlp_create_qkv_heads"(%arg0) {
      num_q_heads = 24 : ui32,
      transpose_k_heads = true
    } : (tensor<1x1x32x4608xbf16>) -> (tensor<1x24x64x64xbf16>, tensor<1x24x64x64xbf16>, tensor<1x24x64x64xbf16>)
    return %query, %key, %value : tensor<1x24x64x64xbf16>, tensor<1x24x64x64xbf16>, tensor<1x24x64x64xbf16>
  }
}

// -----

// Verify that the parsing fails if head dimensions don't match between query and value.
module {
  func.func @nlp_create_qkv_heads_head_dim_mismatch(%arg0: tensor<1x1x32x4608xbf16>) -> (tensor<1x24x32x64xbf16>, tensor<1x24x64x32xbf16>, tensor<1x24x32x128xbf16>) {
    // CHECK: error: 'ttnn.nlp_create_qkv_heads' op value head dimension must match query head dimension
    %query, %key, %value = "ttnn.nlp_create_qkv_heads"(%arg0) {
      num_q_heads = 24 : ui32,
      transpose_k_heads = true
    } : (tensor<1x1x32x4608xbf16>) -> (tensor<1x24x32x64xbf16>, tensor<1x24x64x32xbf16>, tensor<1x24x32x128xbf16>)
    return %query, %key, %value : tensor<1x24x32x64xbf16>, tensor<1x24x64x32xbf16>, tensor<1x24x32x128xbf16>
  }
}

// -----

// Verify that the parsing fails if key dimensions are wrong when transpose_k_heads=true.
module {
  func.func @nlp_create_qkv_heads_key_transpose_wrong(%arg0: tensor<1x1x32x4608xbf16>) -> (tensor<1x24x32x64xbf16>, tensor<1x24x32x64xbf16>, tensor<1x24x32x64xbf16>) {
    // CHECK: error: 'ttnn.nlp_create_qkv_heads' op key dimension 2 must be head_dim when transpose_k_heads=true
    %query, %key, %value = "ttnn.nlp_create_qkv_heads"(%arg0) {
      num_q_heads = 24 : ui32,
      transpose_k_heads = true
    } : (tensor<1x1x32x4608xbf16>) -> (tensor<1x24x32x64xbf16>, tensor<1x24x32x64xbf16>, tensor<1x24x32x64xbf16>)
    return %query, %key, %value : tensor<1x24x32x64xbf16>, tensor<1x24x32x64xbf16>, tensor<1x24x32x64xbf16>
  }
}

// -----

// Verify that the parsing fails if key dimensions are wrong when transpose_k_heads=false.
module {
  func.func @nlp_create_qkv_heads_key_no_transpose_wrong(%arg0: tensor<1x1x32x4608xbf16>) -> (tensor<1x24x32x64xbf16>, tensor<1x24x64x32xbf16>, tensor<1x24x32x64xbf16>) {
    // CHECK: error: 'ttnn.nlp_create_qkv_heads' op key sequence length must match input_q sequence when transpose_k_heads=false
    %query, %key, %value = "ttnn.nlp_create_qkv_heads"(%arg0) {
      num_q_heads = 24 : ui32,
      transpose_k_heads = false
    } : (tensor<1x1x32x4608xbf16>) -> (tensor<1x24x32x64xbf16>, tensor<1x24x64x32xbf16>, tensor<1x24x32x64xbf16>)
    return %query, %key, %value : tensor<1x24x32x64xbf16>, tensor<1x24x64x32xbf16>, tensor<1x24x32x64xbf16>
  }
}
