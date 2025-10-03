// RUN: not ttmlir-opt --split-input-file %s 2>&1 | FileCheck %s
// Negative tests for split query key value and split heads operation.

// Verify that the parsing fails if input tensor is not 3D.
module {
  func.func @split_qkv_and_split_heads_invalid_1(%arg0: tensor<68x3072xf32>) -> (tensor<2x16x34x64xf32>, tensor<2x16x64x34xf32>, tensor<2x16x34x64xf32>) {
    %0 = ttir.empty() : tensor<2x16x34x64xf32>
    %1 = ttir.empty() :  tensor<2x16x64x34xf32>
    %2 = ttir.empty() : tensor<2x16x34x64xf32>
    // CHECK: error: 'ttir.split_query_key_value_and_split_heads' op expected rank of input tensor is 3, got rank 2
    %3, %4, %5 = "ttir.split_query_key_value_and_split_heads"(%arg0, %0, %1, %2) <{num_heads = 16 : ui32, transpose_key = false}> : (tensor<68x3072xf32>, tensor<2x16x34x64xf32>, tensor<2x16x64x34xf32>, tensor<2x16x34x64xf32>) -> (tensor<2x16x34x64xf32>, tensor<2x16x64x34xf32>, tensor<2x16x34x64xf32>)
    return %3, %4, %5 : tensor<2x16x34x64xf32>, tensor<2x16x64x34xf32>, tensor<2x16x34x64xf32>
  }
}

// Verify that the parsing fails if output query tensor is not 4D.
module {
  func.func @split_qkv_and_split_heads_invalid_2(%arg0: tensor<2x34x3072xf32>) -> (tensor<16x34x64xf32>, tensor<2x16x64x34xf32>, tensor<2x16x34x64xf32>) {
    %0 = ttir.empty() : tensor<32x34x64xf32>
    %1 = ttir.empty() :  tensor<2x16x64x34xf32>
    %2 = ttir.empty() : tensor<2x16x34x64xf32>
    // CHECK:  error: 'ttir.split_query_key_value_and_split_heads' op expected rank of query/key/value output tensor is 4, got query rank: 3, key rank: 4, value rank: 4
    %3, %4, %5 = "ttir.split_query_key_value_and_split_heads"(%arg0, %0, %1, %2) <{num_heads = 16 : ui32, transpose_key = false}> : (tensor<2x34x3072xf32>, tensor<32x34x64xf32>, tensor<2x16x64x34xf32>, tensor<2x16x34x64xf32>) -> (tensor<32x34x64xf32>, tensor<2x16x64x34xf32>, tensor<2x16x34x64xf32>)
    return %3, %4, %5 : tensor<32x34x64xf32>, tensor<2x16x64x34xf32>, tensor<2x16x34x64xf32>
  }
}

// Verify that the parsing fails if output key tensor is not 4D.
module {
  func.func @split_qkv_and_split_heads_invalid_3(%arg0: tensor<2x34x3072xf32>) -> (tensor<2x16x34x64xf32>, tensor<2x16x2176xf32>, tensor<2x16x34x64xf32>) {
    %0 = ttir.empty() : tensor<2x16x34x64xf32>
    %1 = ttir.empty() :  tensor<2x16x2176xf32>
    %2 = ttir.empty() : tensor<2x16x34x64xf32>
    // CHECK:  error: 'ttir.split_query_key_value_and_split_heads' op expected rank of query/key/value output tensor is 4, got query rank: 4, key rank: 3, value rank: 4
    %3, %4, %5 = "ttir.split_query_key_value_and_split_heads"(%arg0, %0, %1, %2) <{num_heads = 16 : ui32, transpose_key = false}> : (tensor<2x34x3072xf32>, tensor<2x16x34x64xf32>, tensor<2x16x2176xf32>, tensor<2x16x34x64xf32>) -> (tensor<2x16x34x64xf32>, tensor<2x16x2176xf32>, tensor<2x16x34x64xf32>)
    return %3, %4, %5 : tensor<2x16x34x64xf32>, tensor<2x16x2176xf32>, tensor<2x16x34x64xf32>
  }
}

// Verify that the parsing fails if output value tensor is not 4D.
module {
  func.func @split_qkv_and_split_heads_invalid_4(%arg0: tensor<2x34x3072xf32>) -> (tensor<2x16x34x64xf32>, tensor<2x16x64x34xf32>, tensor<16x34x64xf32>) {
    %0 = ttir.empty() : tensor<2x16x34x64xf32>
    %1 = ttir.empty() :  tensor<2x16x64x34xf32>
    %2 = ttir.empty() : tensor<16x34x64xf32>
    // CHECK:  error: 'ttir.split_query_key_value_and_split_heads' op expected rank of query/key/value output tensor is 4, got query rank: 4, key rank: 4, value rank: 3
    %3, %4, %5 = "ttir.split_query_key_value_and_split_heads"(%arg0, %0, %1, %2) <{num_heads = 16 : ui32, transpose_key = false}> : (tensor<2x34x3072xf32>, tensor<2x16x34x64xf32>, tensor<2x16x64x34xf32>, tensor<16x34x64xf32>) -> (tensor<2x16x34x64xf32>, tensor<2x16x64x34xf32>, tensor<16x34x64xf32>)
    return %3, %4, %5 : tensor<2x16x34x64xf32>, tensor<2x16x64x34xf32>, tensor<16x34x64xf32>
  }
}

// Verify that the parsing fails if batch sizes don't match
module {
  func.func @split_qkv_and_split_heads_invalid_batch(%arg0: tensor<2x34x3072xf32>) -> (tensor<4x16x34x64xf32>, tensor<2x16x64x34xf32>, tensor<2x16x34x64xf32>) {
    %0 = ttir.empty() : tensor<4x16x34x64xf32>
    %1 = ttir.empty() : tensor<2x16x64x34xf32>
    %2 = ttir.empty() : tensor<2x16x34x64xf32>
    // CHECK: error: 'ttir.split_query_key_value_and_split_heads' op expected query output batch dimension to be 2, got query batch size: 4, key batch size: 2, value batch size: 2
    %3, %4, %5 = "ttir.split_query_key_value_and_split_heads"(%arg0, %0, %1, %2) <{num_heads = 16 : ui32, transpose_key = false}> : (tensor<2x34x3072xf32>, tensor<4x16x34x64xf32>, tensor<2x16x64x34xf32>, tensor<2x16x34x64xf32>) -> (tensor<4x16x34x64xf32>, tensor<2x16x64x34xf32>, tensor<2x16x34x64xf32>)
    return %3, %4, %5 : tensor<4x16x34x64xf32>, tensor<2x16x64x34xf32>, tensor<2x16x34x64xf32>
  }
}

// Verify that the parsing fails if sequence lengths don't match
module {
  func.func @split_qkv_and_split_heads_invalid_seq_len(%arg0: tensor<2x34x3072xf32>) -> (tensor<2x16x34x64xf32>, tensor<2x16x64x32xf32>, tensor<2x16x34x64xf32>) {
    %0 = ttir.empty() : tensor<2x16x34x64xf32>
    %1 = ttir.empty() : tensor<2x16x64x32xf32>
    %2 = ttir.empty() : tensor<2x16x34x64xf32>
    // CHECK: error: 'ttir.split_query_key_value_and_split_heads' op expected query output sequence dimension to be 34, got query sequence size: 34, key sequence size: 32, value sequence size: 34
    %3, %4, %5 = "ttir.split_query_key_value_and_split_heads"(%arg0, %0, %1, %2) <{num_heads = 16 : ui32, transpose_key = false}> : (tensor<2x34x3072xf32>, tensor<2x16x34x64xf32>, tensor<2x16x64x32xf32>, tensor<2x16x34x64xf32>) -> (tensor<2x16x34x64xf32>, tensor<2x16x64x32xf32>, tensor<2x16x34x64xf32>)
    return %3, %4, %5 : tensor<2x16x34x64xf32>, tensor<2x16x64x32xf32>, tensor<2x16x34x64xf32>
  }
}

// Verify that the parsing fails if num_kv_heads don't match across Q, K, V
module {
  func.func @split_qkv_and_split_heads_invalid_num_kv_heads(%arg0: tensor<2x34x3072xf32>) -> (tensor<2x16x34x64xf32>, tensor<2x8x64x34xf32>, tensor<2x16x34x64xf32>) {
    %0 = ttir.empty() : tensor<2x16x34x64xf32>
    %1 = ttir.empty() : tensor<2x8x64x34xf32>
    %2 = ttir.empty() : tensor<2x16x34x64xf32>
    // CHECK: error: 'ttir.split_query_key_value_and_split_heads' op expected num_kv_heads to be the same for query, key and value, got query num_kv_heads: 16, key num_kv_heads: 8, value num_kv_heads: 16
    %3, %4, %5 = "ttir.split_query_key_value_and_split_heads"(%arg0, %0, %1, %2) <{num_heads = 16 : ui32, transpose_key = false}> : (tensor<2x34x3072xf32>, tensor<2x16x34x64xf32>, tensor<2x8x64x34xf32>, tensor<2x16x34x64xf32>) -> (tensor<2x16x34x64xf32>, tensor<2x8x64x34xf32>, tensor<2x16x34x64xf32>)
    return %3, %4, %5 : tensor<2x16x34x64xf32>, tensor<2x8x64x34xf32>, tensor<2x16x34x64xf32>
  }
}

// Verify that the parsing fails if num_heads attribute doesn't match num_kv_heads dimension
module {
  func.func @split_qkv_and_split_heads_invalid_num_heads_attr(%arg0: tensor<2x34x3072xf32>) -> (tensor<2x16x34x64xf32>, tensor<2x16x64x34xf32>, tensor<2x16x34x64xf32>) {
    %0 = ttir.empty() : tensor<2x16x34x64xf32>
    %1 = ttir.empty() : tensor<2x16x64x34xf32>
    %2 = ttir.empty() : tensor<2x16x34x64xf32>
    // CHECK: error: 'ttir.split_query_key_value_and_split_heads' op expected num_heads attribute to be equal to num_kv_heads = 16, got num_heads = 8
    %3, %4, %5 = "ttir.split_query_key_value_and_split_heads"(%arg0, %0, %1, %2) <{num_heads = 8 : ui32, transpose_key = false}> : (tensor<2x34x3072xf32>, tensor<2x16x34x64xf32>, tensor<2x16x64x34xf32>, tensor<2x16x34x64xf32>) -> (tensor<2x16x34x64xf32>, tensor<2x16x64x34xf32>, tensor<2x16x34x64xf32>)
    return %3, %4, %5 : tensor<2x16x34x64xf32>, tensor<2x16x64x34xf32>, tensor<2x16x34x64xf32>
  }
}

// Verify that the parsing fails if head_size doesn't match across Q, K, V
module {
  func.func @split_qkv_and_split_heads_invalid_head_size(%arg0: tensor<2x34x3072xf32>) -> (tensor<2x16x34x64xf32>, tensor<2x16x32x34xf32>, tensor<2x16x34x64xf32>) {
    %0 = ttir.empty() : tensor<2x16x34x64xf32>
    %1 = ttir.empty() : tensor<2x16x32x34xf32>
    %2 = ttir.empty() : tensor<2x16x34x64xf32>
    // CHECK: error: 'ttir.split_query_key_value_and_split_heads' op expected head_size to be the same for query, key and value, got query head_size: 64, key head_size: 32, value head_size: 64
    %3, %4, %5 = "ttir.split_query_key_value_and_split_heads"(%arg0, %0, %1, %2) <{num_heads = 16 : ui32, transpose_key = false}> : (tensor<2x34x3072xf32>, tensor<2x16x34x64xf32>, tensor<2x16x32x34xf32>, tensor<2x16x34x64xf32>) -> (tensor<2x16x34x64xf32>, tensor<2x16x32x34xf32>, tensor<2x16x34x64xf32>)
    return %3, %4, %5 : tensor<2x16x34x64xf32>, tensor<2x16x32x34xf32>, tensor<2x16x34x64xf32>
  }
}

// Verify that the parsing fails if hidden dimension calculation is incorrect
module {
  func.func @split_qkv_and_split_heads_invalid_hidden_dim(%arg0: tensor<2x34x2048xf32>) -> (tensor<2x16x34x64xf32>, tensor<2x16x64x34xf32>, tensor<2x16x34x64xf32>) {
    %0 = ttir.empty() : tensor<2x16x34x64xf32>
    %1 = ttir.empty() : tensor<2x16x64x34xf32>
    %2 = ttir.empty() : tensor<2x16x34x64xf32>
    // CHECK: error: 'ttir.split_query_key_value_and_split_heads' op expected input[2] to be 3 * hidden dimension ( = 3 * num_kv_heads * head_size) = 3072, got 2048
    %3, %4, %5 = "ttir.split_query_key_value_and_split_heads"(%arg0, %0, %1, %2) <{num_heads = 16 : ui32, transpose_key = false}> : (tensor<2x34x2048xf32>, tensor<2x16x34x64xf32>, tensor<2x16x64x34xf32>, tensor<2x16x34x64xf32>) -> (tensor<2x16x34x64xf32>, tensor<2x16x64x34xf32>, tensor<2x16x34x64xf32>)
    return %3, %4, %5 : tensor<2x16x34x64xf32>, tensor<2x16x64x34xf32>, tensor<2x16x34x64xf32>
  }
}
