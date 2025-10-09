// RUN: not ttmlir-opt --split-input-file %s 2>&1 | FileCheck %s

// -----
// Test: input_q must be 4D tensor
module {
  func.func @nlp_create_qkv_heads_inputq_not_4d(%arg0: tensor<1x32x4608xbf16>) -> (tensor<1x24x32x64xbf16>, tensor<1x24x64x32xbf16>, tensor<1x24x32x64xbf16>) {
    // CHECK: error: 'ttnn.nlp_create_qkv_heads' op input_q tensor must be a 4D tensor
    %query, %key, %value = "ttnn.nlp_create_qkv_heads"(%arg0) <{
      num_q_heads = 24 : ui32,
      transpose_k_heads = true
    }> : (tensor<1x32x4608xbf16>) -> (tensor<1x24x32x64xbf16>, tensor<1x24x64x32xbf16>, tensor<1x24x32x64xbf16>)
    return %query, %key, %value : tensor<1x24x32x64xbf16>, tensor<1x24x64x32xbf16>, tensor<1x24x32x64xbf16>
  }
}

// -----
// Test: output query must be 4D tensor
module {
  func.func @nlp_create_qkv_heads_query_not_4d(%arg0: tensor<1x1x32x4608xbf16>) -> (tensor<24x32x64xbf16>, tensor<1x24x64x32xbf16>, tensor<1x24x32x64xbf16>) {
    // CHECK: error: 'ttnn.nlp_create_qkv_heads' op output query tensor must be a 4D tensor
    %query, %key, %value = "ttnn.nlp_create_qkv_heads"(%arg0) <{
      num_q_heads = 24 : ui32,
      transpose_k_heads = true
    }> : (tensor<1x1x32x4608xbf16>) -> (tensor<24x32x64xbf16>, tensor<1x24x64x32xbf16>, tensor<1x24x32x64xbf16>)
    return %query, %key, %value : tensor<24x32x64xbf16>, tensor<1x24x64x32xbf16>, tensor<1x24x32x64xbf16>
  }
}

// -----
// Test: output key must be 4D tensor
module {
  func.func @nlp_create_qkv_heads_key_not_4d(%arg0: tensor<1x1x32x4608xbf16>) -> (tensor<1x24x32x64xbf16>, tensor<24x64x32xbf16>, tensor<1x24x32x64xbf16>) {
    // CHECK: error: 'ttnn.nlp_create_qkv_heads' op output key tensor must be a 4D tensor
    %query, %key, %value = "ttnn.nlp_create_qkv_heads"(%arg0) <{
      num_q_heads = 24 : ui32,
      transpose_k_heads = true
    }> : (tensor<1x1x32x4608xbf16>) -> (tensor<1x24x32x64xbf16>, tensor<24x64x32xbf16>, tensor<1x24x32x64xbf16>)
    return %query, %key, %value : tensor<1x24x32x64xbf16>, tensor<24x64x32xbf16>, tensor<1x24x32x64xbf16>
  }
}

// -----
// Test: output value must be 4D tensor
module {
  func.func @nlp_create_qkv_heads_value_not_4d(%arg0: tensor<1x1x32x4608xbf16>) -> (tensor<1x24x32x64xbf16>, tensor<1x24x64x32xbf16>, tensor<24x32x64xbf16>) {
    // CHECK: error: 'ttnn.nlp_create_qkv_heads' op output value tensor must be a 4D tensor
    %query, %key, %value = "ttnn.nlp_create_qkv_heads"(%arg0) <{
      num_q_heads = 24 : ui32,
      transpose_k_heads = true
    }> : (tensor<1x1x32x4608xbf16>) -> (tensor<1x24x32x64xbf16>, tensor<1x24x64x32xbf16>, tensor<24x32x64xbf16>)
    return %query, %key, %value : tensor<1x24x32x64xbf16>, tensor<1x24x64x32xbf16>, tensor<24x32x64xbf16>
  }
}

// -----
// Test: input_kv must be 4D tensor
module {
  func.func @nlp_create_qkv_heads_inputkv_not_4d(%arg0: tensor<1x1x32x1536xbf16>, %arg1: tensor<1x32x1024xbf16>) -> (tensor<1x24x32x64xbf16>, tensor<1x8x64x32xbf16>, tensor<1x8x32x64xbf16>) {
    // CHECK: error: 'ttnn.nlp_create_qkv_heads' op input_kv tensor must be a 4D tensor
    %query, %key, %value = "ttnn.nlp_create_qkv_heads"(%arg0, %arg1) <{
      num_q_heads = 24 : ui32,
      num_kv_heads = 8 : ui32,
      transpose_k_heads = true
    }> : (tensor<1x1x32x1536xbf16>, tensor<1x32x1024xbf16>) -> (tensor<1x24x32x64xbf16>, tensor<1x8x64x32xbf16>, tensor<1x8x32x64xbf16>)
    return %query, %key, %value : tensor<1x24x32x64xbf16>, tensor<1x8x64x32xbf16>, tensor<1x8x32x64xbf16>
  }
}

// -----
// Test: input_q dimension 1 must be 1
module {
  func.func @nlp_create_qkv_heads_inputq_dim1_not_1(%arg0: tensor<1x2x32x4608xbf16>) -> (tensor<1x24x32x64xbf16>, tensor<1x24x64x32xbf16>, tensor<1x24x32x64xbf16>) {
    // CHECK: error: 'ttnn.nlp_create_qkv_heads' op input_q dimension 1 must be 1, got 2
    %query, %key, %value = "ttnn.nlp_create_qkv_heads"(%arg0) <{
      num_q_heads = 24 : ui32,
      transpose_k_heads = true
    }> : (tensor<1x2x32x4608xbf16>) -> (tensor<1x24x32x64xbf16>, tensor<1x24x64x32xbf16>, tensor<1x24x32x64xbf16>)
    return %query, %key, %value : tensor<1x24x32x64xbf16>, tensor<1x24x64x32xbf16>, tensor<1x24x32x64xbf16>
  }
}

// -----
// Test: input_q hidden dimension must be divisible by num_q_heads (dual input case)
module {
  func.func @nlp_create_qkv_heads_inputq_not_divisible_dual(%arg0: tensor<1x1x32x1535xbf16>, %arg1: tensor<1x1x32x1024xbf16>) -> (tensor<1x24x32x64xbf16>, tensor<1x8x64x32xbf16>, tensor<1x8x32x64xbf16>) {
    // CHECK: error: 'ttnn.nlp_create_qkv_heads' op input_q hidden dimension 3 must be divisible by num_q_heads, got input_q hidden dimension 3 = 1535 and num_q_heads=24
    %query, %key, %value = "ttnn.nlp_create_qkv_heads"(%arg0, %arg1) <{
      num_q_heads = 24 : ui32,
      num_kv_heads = 8 : ui32,
      transpose_k_heads = true
    }> : (tensor<1x1x32x1535xbf16>, tensor<1x1x32x1024xbf16>) -> (tensor<1x24x32x64xbf16>, tensor<1x8x64x32xbf16>, tensor<1x8x32x64xbf16>)
    return %query, %key, %value : tensor<1x24x32x64xbf16>, tensor<1x8x64x32xbf16>, tensor<1x8x32x64xbf16>
  }
}

// -----
// Test: input_kv dimension 1 must be 1
module {
  func.func @nlp_create_qkv_heads_inputkv_dim1_not_1(%arg0: tensor<1x1x32x1536xbf16>, %arg1: tensor<1x2x32x1024xbf16>) -> (tensor<1x24x32x64xbf16>, tensor<1x8x64x32xbf16>, tensor<1x8x32x64xbf16>) {
    // CHECK: error: 'ttnn.nlp_create_qkv_heads' op input_kv dimension 1 must be 1, got 2
    %query, %key, %value = "ttnn.nlp_create_qkv_heads"(%arg0, %arg1) <{
      num_q_heads = 24 : ui32,
      num_kv_heads = 8 : ui32,
      transpose_k_heads = true
    }> : (tensor<1x1x32x1536xbf16>, tensor<1x2x32x1024xbf16>) -> (tensor<1x24x32x64xbf16>, tensor<1x8x64x32xbf16>, tensor<1x8x32x64xbf16>)
    return %query, %key, %value : tensor<1x24x32x64xbf16>, tensor<1x8x64x32xbf16>, tensor<1x8x32x64xbf16>
  }
}

// -----
// Test: input_kv batch size must match input_q batch size
module {
  func.func @nlp_create_qkv_heads_batch_mismatch(%arg0: tensor<1x1x32x1536xbf16>, %arg1: tensor<2x1x32x1024xbf16>) -> (tensor<1x24x32x64xbf16>, tensor<1x8x64x32xbf16>, tensor<1x8x32x64xbf16>) {
    // CHECK: error: 'ttnn.nlp_create_qkv_heads' op input_kv batch size dimension 0 must match input_q batch size dimension 0, got input_kv batch = 2, input_q batch = 1
    %query, %key, %value = "ttnn.nlp_create_qkv_heads"(%arg0, %arg1) <{
      num_q_heads = 24 : ui32,
      num_kv_heads = 8 : ui32,
      transpose_k_heads = true
    }> : (tensor<1x1x32x1536xbf16>, tensor<2x1x32x1024xbf16>) -> (tensor<1x24x32x64xbf16>, tensor<1x8x64x32xbf16>, tensor<1x8x32x64xbf16>)
    return %query, %key, %value : tensor<1x24x32x64xbf16>, tensor<1x8x64x32xbf16>, tensor<1x8x32x64xbf16>
  }
}

// -----
// Test: input_kv sequence length must match input_q sequence length
module {
  func.func @nlp_create_qkv_heads_seq_mismatch(%arg0: tensor<1x1x32x1536xbf16>, %arg1: tensor<1x1x64x1024xbf16>) -> (tensor<1x24x32x64xbf16>, tensor<1x8x64x32xbf16>, tensor<1x8x32x64xbf16>) {
    // CHECK: error: 'ttnn.nlp_create_qkv_heads' op input_kv sequence length dimension 2 must match input_q sequence length dimension 2, got input_kv seq = 64, input_q seq = 32
    %query, %key, %value = "ttnn.nlp_create_qkv_heads"(%arg0, %arg1) <{
      num_q_heads = 24 : ui32,
      num_kv_heads = 8 : ui32,
      transpose_k_heads = true
    }> : (tensor<1x1x32x1536xbf16>, tensor<1x1x64x1024xbf16>) -> (tensor<1x24x32x64xbf16>, tensor<1x8x64x32xbf16>, tensor<1x8x32x64xbf16>)
    return %query, %key, %value : tensor<1x24x32x64xbf16>, tensor<1x8x64x32xbf16>, tensor<1x8x32x64xbf16>
  }
}

// -----
// Test: input_kv hidden dimension must be 2 * num_kv_heads * head_dim
module {
  func.func @nlp_create_qkv_heads_inputkv_hidden_wrong(%arg0: tensor<1x1x32x1536xbf16>, %arg1: tensor<1x1x32x1000xbf16>) -> (tensor<1x24x32x64xbf16>, tensor<1x8x64x32xbf16>, tensor<1x8x32x64xbf16>) {
    // CHECK: error: 'ttnn.nlp_create_qkv_heads' op input_kv hidden dimension must be 2 * num_kv_heads * head_dim, got 1000, expected 1024 (2 * num_kv_heads=8 * head_dim=64)
    %query, %key, %value = "ttnn.nlp_create_qkv_heads"(%arg0, %arg1) <{
      num_q_heads = 24 : ui32,
      num_kv_heads = 8 : ui32,
      transpose_k_heads = true
    }> : (tensor<1x1x32x1536xbf16>, tensor<1x1x32x1000xbf16>) -> (tensor<1x24x32x64xbf16>, tensor<1x8x64x32xbf16>, tensor<1x8x32x64xbf16>)
    return %query, %key, %value : tensor<1x24x32x64xbf16>, tensor<1x8x64x32xbf16>, tensor<1x8x32x64xbf16>
  }
}

// -----
// Test: input_q hidden dimension must be divisible by 3 * num_q_heads (single input case)
module {
  func.func @nlp_create_qkv_heads_inputq_not_divisible_single(%arg0: tensor<1x1x32x4607xbf16>) -> (tensor<1x24x32x64xbf16>, tensor<1x24x64x32xbf16>, tensor<1x24x32x64xbf16>) {
    // CHECK: error: 'ttnn.nlp_create_qkv_heads' op input_q hidden dimension 3 must be divisible by 3 * num_q_heads, got input_q hidden dimension 3 = 4607 and 3 * num_q_heads=72
    %query, %key, %value = "ttnn.nlp_create_qkv_heads"(%arg0) <{
      num_q_heads = 24 : ui32,
      transpose_k_heads = true
    }> : (tensor<1x1x32x4607xbf16>) -> (tensor<1x24x32x64xbf16>, tensor<1x24x64x32xbf16>, tensor<1x24x32x64xbf16>)
    return %query, %key, %value : tensor<1x24x32x64xbf16>, tensor<1x24x64x32xbf16>, tensor<1x24x32x64xbf16>
  }
}

// -----
// Test: output query batch must match input_q batch
module {
  func.func @nlp_create_qkv_heads_query_batch_mismatch(%arg0: tensor<1x1x32x4608xbf16>) -> (tensor<2x24x32x64xbf16>, tensor<1x24x64x32xbf16>, tensor<1x24x32x64xbf16>) {
    // CHECK: error: 'ttnn.nlp_create_qkv_heads' op input_q batch size dimension 0 must match output query batch size dimension 0, got input_q batch = 1, query batch = 2
    %query, %key, %value = "ttnn.nlp_create_qkv_heads"(%arg0) <{
      num_q_heads = 24 : ui32,
      transpose_k_heads = true
    }> : (tensor<1x1x32x4608xbf16>) -> (tensor<2x24x32x64xbf16>, tensor<1x24x64x32xbf16>, tensor<1x24x32x64xbf16>)
    return %query, %key, %value : tensor<2x24x32x64xbf16>, tensor<1x24x64x32xbf16>, tensor<1x24x32x64xbf16>
  }
}

// -----
// Test: output key batch must match input_q batch
module {
  func.func @nlp_create_qkv_heads_key_batch_mismatch(%arg0: tensor<1x1x32x4608xbf16>) -> (tensor<1x24x32x64xbf16>, tensor<2x24x64x32xbf16>, tensor<1x24x32x64xbf16>) {
    // CHECK: error: 'ttnn.nlp_create_qkv_heads' op input_q batch size dimension 0 must match output key batch size dimension 0, got input_q batch = 1, key batch = 2
    %query, %key, %value = "ttnn.nlp_create_qkv_heads"(%arg0) <{
      num_q_heads = 24 : ui32,
      transpose_k_heads = true
    }> : (tensor<1x1x32x4608xbf16>) -> (tensor<1x24x32x64xbf16>, tensor<2x24x64x32xbf16>, tensor<1x24x32x64xbf16>)
    return %query, %key, %value : tensor<1x24x32x64xbf16>, tensor<2x24x64x32xbf16>, tensor<1x24x32x64xbf16>
  }
}

// -----
// Test: output value batch must match input_q batch
module {
  func.func @nlp_create_qkv_heads_key_batch_mismatch(%arg0: tensor<1x1x32x4608xbf16>) -> (tensor<1x24x32x64xbf16>, tensor<1x24x64x32xbf16>, tensor<2x24x32x64xbf16>) {
    // CHECK: error: 'ttnn.nlp_create_qkv_heads' op input_q batch size dimension 0 must match output value batch size dimension 0, got input_q batch = 1, value batch = 2
    %query, %key, %value = "ttnn.nlp_create_qkv_heads"(%arg0) <{
      num_q_heads = 24 : ui32,
      transpose_k_heads = true
    }> : (tensor<1x1x32x4608xbf16>) -> (tensor<1x24x32x64xbf16>, tensor<1x24x64x32xbf16>, tensor<2x24x32x64xbf16>)
    return %query, %key, %value : tensor<1x24x32x64xbf16>, tensor<1x24x64x32xbf16>, tensor<2x24x32x64xbf16>
  }
}

// -----
// Test: query tensor num_heads must match num_q_heads
module {
  func.func @nlp_create_qkv_heads_query_numheads_mismatch(%arg0: tensor<1x1x32x4608xbf16>) -> (tensor<1x16x32x64xbf16>, tensor<1x24x64x32xbf16>, tensor<1x24x32x64xbf16>) {
    // CHECK: error: 'ttnn.nlp_create_qkv_heads' op query tensor dimension 1 must match num_q_heads, got 16, expected 24
    %query, %key, %value = "ttnn.nlp_create_qkv_heads"(%arg0) <{
      num_q_heads = 24 : ui32,
      transpose_k_heads = true
    }> : (tensor<1x1x32x4608xbf16>) -> (tensor<1x16x32x64xbf16>, tensor<1x24x64x32xbf16>, tensor<1x24x32x64xbf16>)
    return %query, %key, %value : tensor<1x16x32x64xbf16>, tensor<1x24x64x32xbf16>, tensor<1x24x32x64xbf16>
  }
}

// -----
// Test: key tensor num_heads must match num_kv_heads
module {
  func.func @nlp_create_qkv_heads_key_numheads_mismatch(%arg0: tensor<1x1x32x1536xbf16>, %arg1: tensor<1x1x32x1024xbf16>) -> (tensor<1x24x32x64xbf16>, tensor<1x16x64x32xbf16>, tensor<1x8x32x64xbf16>) {
    // CHECK: error: 'ttnn.nlp_create_qkv_heads' op key tensor dimension 1 must match num_kv_heads, got 16, expected 8
    %query, %key, %value = "ttnn.nlp_create_qkv_heads"(%arg0, %arg1) <{
      num_q_heads = 24 : ui32,
      num_kv_heads = 8 : ui32,
      transpose_k_heads = true
    }> : (tensor<1x1x32x1536xbf16>, tensor<1x1x32x1024xbf16>) -> (tensor<1x24x32x64xbf16>, tensor<1x16x64x32xbf16>, tensor<1x8x32x64xbf16>)
    return %query, %key, %value : tensor<1x24x32x64xbf16>, tensor<1x16x64x32xbf16>, tensor<1x8x32x64xbf16>
  }
}

// -----
// Test: value tensor num_heads must match num_kv_heads
module {
  func.func @nlp_create_qkv_heads_value_numheads_mismatch(%arg0: tensor<1x1x32x1536xbf16>, %arg1: tensor<1x1x32x1024xbf16>) -> (tensor<1x24x32x64xbf16>, tensor<1x8x64x32xbf16>, tensor<1x16x32x64xbf16>) {
    // CHECK: error: 'ttnn.nlp_create_qkv_heads' op value tensor dimension 1 must match num_kv_heads, got 16, expected 8
    %query, %key, %value = "ttnn.nlp_create_qkv_heads"(%arg0, %arg1) <{
      num_q_heads = 24 : ui32,
      num_kv_heads = 8 : ui32,
      transpose_k_heads = true
    }> : (tensor<1x1x32x1536xbf16>, tensor<1x1x32x1024xbf16>) -> (tensor<1x24x32x64xbf16>, tensor<1x8x64x32xbf16>, tensor<1x16x32x64xbf16>)
    return %query, %key, %value : tensor<1x24x32x64xbf16>, tensor<1x8x64x32xbf16>, tensor<1x16x32x64xbf16>
  }
}

// -----
// Test: query sequence length must match input_q sequence length
module {
  func.func @nlp_create_qkv_heads_query_seq_mismatch(%arg0: tensor<1x1x32x4608xbf16>) -> (tensor<1x24x64x64xbf16>, tensor<1x24x64x32xbf16>, tensor<1x24x32x64xbf16>) {
    // CHECK: error: 'ttnn.nlp_create_qkv_heads' op output query sequence length dimension 2 must match input_q sequence length dimension 2, got query seq = 64, input_q seq = 32
    %query, %key, %value = "ttnn.nlp_create_qkv_heads"(%arg0) <{
      num_q_heads = 24 : ui32,
      transpose_k_heads = true
    }> : (tensor<1x1x32x4608xbf16>) -> (tensor<1x24x64x64xbf16>, tensor<1x24x64x32xbf16>, tensor<1x24x32x64xbf16>)
    return %query, %key, %value : tensor<1x24x64x64xbf16>, tensor<1x24x64x32xbf16>, tensor<1x24x32x64xbf16>
  }
}

// -----
// Test: query head dimension must match computed head dimension
module {
  func.func @nlp_create_qkv_heads_query_headdim_mismatch(%arg0: tensor<1x1x32x4608xbf16>) -> (tensor<1x24x32x128xbf16>, tensor<1x24x64x32xbf16>, tensor<1x24x32x64xbf16>) {
    // CHECK: error: 'ttnn.nlp_create_qkv_heads' op output query head dimension 2 must match input_q head dimension, got query head_dim = 128, input_q head_dim = 64
    %query, %key, %value = "ttnn.nlp_create_qkv_heads"(%arg0) <{
      num_q_heads = 24 : ui32,
      transpose_k_heads = true
    }> : (tensor<1x1x32x4608xbf16>) -> (tensor<1x24x32x128xbf16>, tensor<1x24x64x32xbf16>, tensor<1x24x32x64xbf16>)
    return %query, %key, %value : tensor<1x24x32x128xbf16>, tensor<1x24x64x32xbf16>, tensor<1x24x32x64xbf16>
  }
}

// -----
// Test: value sequence length must match input_q sequence length
module {
  func.func @nlp_create_qkv_heads_value_seq_mismatch(%arg0: tensor<1x1x32x4608xbf16>) -> (tensor<1x24x32x64xbf16>, tensor<1x24x64x32xbf16>, tensor<1x24x64x64xbf16>) {
    // CHECK: error: 'ttnn.nlp_create_qkv_heads' op output value sequence length dimension 2 must match input_q sequence length dimension 2, got value seq = 64, input_q seq = 32
    %query, %key, %value = "ttnn.nlp_create_qkv_heads"(%arg0) <{
      num_q_heads = 24 : ui32,
      transpose_k_heads = true
    }> : (tensor<1x1x32x4608xbf16>) -> (tensor<1x24x32x64xbf16>, tensor<1x24x64x32xbf16>, tensor<1x24x64x64xbf16>)
    return %query, %key, %value : tensor<1x24x32x64xbf16>, tensor<1x24x64x32xbf16>, tensor<1x24x64x64xbf16>
  }
}

// -----
// Test: value head dimension must match computed head dimension
module {
  func.func @nlp_create_qkv_heads_value_headdim_mismatch(%arg0: tensor<1x1x32x4608xbf16>) -> (tensor<1x24x32x64xbf16>, tensor<1x24x64x32xbf16>, tensor<1x24x32x128xbf16>) {
    // CHECK: error: 'ttnn.nlp_create_qkv_heads' op output value head dimension must match input_q head dimension, got value head_dim = 128, input_q head_dim = 64
    %query, %key, %value = "ttnn.nlp_create_qkv_heads"(%arg0) <{
      num_q_heads = 24 : ui32,
      transpose_k_heads = true
    }> : (tensor<1x1x32x4608xbf16>) -> (tensor<1x24x32x64xbf16>, tensor<1x24x64x32xbf16>, tensor<1x24x32x128xbf16>)
    return %query, %key, %value : tensor<1x24x32x64xbf16>, tensor<1x24x64x32xbf16>, tensor<1x24x32x128xbf16>
  }
}

// -----
// Test: key head dimension must match when transpose_k_heads=true
module {
  func.func @nlp_create_qkv_heads_key_headdim_mismatch_transposed(%arg0: tensor<1x1x32x4608xbf16>) -> (tensor<1x24x32x64xbf16>, tensor<1x24x128x32xbf16>, tensor<1x24x32x64xbf16>) {
    // CHECK: error: 'ttnn.nlp_create_qkv_heads' op output key head dimension 2 must match input_q head dimension when transpose_k_heads=false, got key head_dim = 32, input_q head_dim = 64
    %query, %key, %value = "ttnn.nlp_create_qkv_heads"(%arg0) <{
      num_q_heads = 24 : ui32,
      transpose_k_heads = true
    }> : (tensor<1x1x32x4608xbf16>) -> (tensor<1x24x32x64xbf16>, tensor<1x24x128x32xbf16>, tensor<1x24x32x64xbf16>)
    return %query, %key, %value : tensor<1x24x32x64xbf16>, tensor<1x24x128x32xbf16>, tensor<1x24x32x64xbf16>
  }
}

// -----
// Test: key sequence length must match when transpose_k_heads=true
module {
  func.func @nlp_create_qkv_heads_key_seq_mismatch_transposed(%arg0: tensor<1x1x32x4608xbf16>) -> (tensor<1x24x32x64xbf16>, tensor<1x24x64x64xbf16>, tensor<1x24x32x64xbf16>) {
    // CHECK: error: 'ttnn.nlp_create_qkv_heads' op output key sequence length dimension 3 must match input_q sequence length dimension 2 whentranspose_k_heads=true, got key seq = 64, input_q seq = 32
    %query, %key, %value = "ttnn.nlp_create_qkv_heads"(%arg0) <{
      num_q_heads = 24 : ui32,
      transpose_k_heads = true
    }> : (tensor<1x1x32x4608xbf16>) -> (tensor<1x24x32x64xbf16>, tensor<1x24x64x64xbf16>, tensor<1x24x32x64xbf16>)
    return %query, %key, %value : tensor<1x24x32x64xbf16>, tensor<1x24x64x64xbf16>, tensor<1x24x32x64xbf16>
  }
}

// -----
// Test: key sequence length must match when transpose_k_heads=false
module {
  func.func @nlp_create_qkv_heads_key_seq_mismatch_not_transposed(%arg0: tensor<1x1x32x4608xbf16>) -> (tensor<1x24x32x64xbf16>, tensor<1x24x64x64xbf16>, tensor<1x24x32x64xbf16>) {
    // CHECK: error: 'ttnn.nlp_create_qkv_heads' op output key sequence length dimension 2 must match input_q sequence length dimension 2 whentranspose_k_heads=true, got key seq = 64, input_q seq = 32
    %query, %key, %value = "ttnn.nlp_create_qkv_heads"(%arg0) <{
      num_q_heads = 24 : ui32,
      transpose_k_heads = false
    }> : (tensor<1x1x32x4608xbf16>) -> (tensor<1x24x32x64xbf16>, tensor<1x24x64x64xbf16>, tensor<1x24x32x64xbf16>)
    return %query, %key, %value : tensor<1x24x32x64xbf16>, tensor<1x24x64x64xbf16>, tensor<1x24x32x64xbf16>
  }
}

// -----
// Test: key head dimension must match when transpose_k_heads=false
module {
  func.func @nlp_create_qkv_heads_key_headdim_mismatch_not_transposed(%arg0: tensor<1x1x32x4608xbf16>) -> (tensor<1x24x32x64xbf16>, tensor<1x24x32x128xbf16>, tensor<1x24x32x64xbf16>) {
    // CHECK: error: 'ttnn.nlp_create_qkv_heads' op output key head dimension 3 must match input_q head dimension when transpose_k_heads=false, got key head_dim = 128, input_q head_dim = 64
    %query, %key, %value = "ttnn.nlp_create_qkv_heads"(%arg0) <{
      num_q_heads = 24 : ui32,
      transpose_k_heads = false
    }> : (tensor<1x1x32x4608xbf16>) -> (tensor<1x24x32x64xbf16>, tensor<1x24x32x128xbf16>, tensor<1x24x32x64xbf16>)
    return %query, %key, %value : tensor<1x24x32x64xbf16>, tensor<1x24x32x128xbf16>, tensor<1x24x32x64xbf16>
  }
}
