// RUN: not ttmlir-opt --split-input-file %s 2>&1 | FileCheck %s
// Negative tests for split query key value and split heads operation.

// Verify that the verifier fails if input tensor is not 3D.
module {
  func.func @split_qkv_and_split_heads_invalid_1(%arg0: tensor<68x3072xf32>) -> (tensor<2x16x32x64xf32>, tensor<2x16x32x64xf32>, tensor<2x16x32x64xf32>) {
    %0 = ttir.empty() : tensor<2x16x32x64xf32>
    %1 = ttir.empty() :  tensor<2x16x32x64xf32>
    %2 = ttir.empty() : tensor<2x16x32x64xf32>
    // CHECK: error: 'ttir.split_query_key_value_and_split_heads' op expected rank of input tensor is 3, got rank 2
    %3, %4, %5 = "ttir.split_query_key_value_and_split_heads"(%arg0, %0, %1, %2) <{num_heads = 16 : ui32, transpose_key = false}> : (tensor<68x3072xf32>, tensor<2x16x32x64xf32>, tensor<2x16x32x64xf32>, tensor<2x16x32x64xf32>) -> (tensor<2x16x32x64xf32>, tensor<2x16x32x64xf32>, tensor<2x16x32x64xf32>)
    return %3, %4, %5 : tensor<2x16x32x64xf32>, tensor<2x16x32x64xf32>, tensor<2x16x32x64xf32>
  }
}

//-----

// Verify that the verifier fails if output query tensor is not 4D.
module {
  func.func @split_qkv_and_split_heads_invalid_2(%arg0: tensor<2x32x3072xf32>) -> (tensor<2x16x32xf32>, tensor<2x16x32x64xf32>, tensor<2x16x32x64xf32>) {
    %0 = ttir.empty() : tensor<2x16x32xf32>
    %1 = ttir.empty() : tensor<2x16x32x64xf32>
    %2 = ttir.empty() : tensor<2x16x32x64xf32>
    // CHECK:  error: 'ttir.split_query_key_value_and_split_heads' op expected rank of query/key/value output tensor is 4, got query rank: 3, key rank: 4, value rank: 4
    %3, %4, %5 = "ttir.split_query_key_value_and_split_heads"(%arg0, %0, %1, %2) <{num_heads = 16 : ui32, transpose_key = false}> : (tensor<2x32x3072xf32>, tensor<2x16x32xf32>, tensor<2x16x32x64xf32>, tensor<2x16x32x64xf32>) -> (tensor<2x16x32xf32>, tensor<2x16x32x64xf32>, tensor<2x16x32x64xf32>)
    return %3, %4, %5 : tensor<2x16x32xf32>, tensor<2x16x32x64xf32>, tensor<2x16x32x64xf32>
  }
}

//-----

// Verify that the verifier fails if output key tensor is not 4D.
module {
  func.func @split_qkv_and_split_heads_invalid_2(%arg0: tensor<2x32x3072xf32>) -> (tensor<2x16x32x64xf32>, tensor<2x16x32xf32>, tensor<2x16x32x64xf32>) {
    %0 = ttir.empty() : tensor<2x16x32x64xf32>
    %1 = ttir.empty() : tensor<2x16x32xf32>
    %2 = ttir.empty() : tensor<2x16x32x64xf32>
    // CHECK:  error: 'ttir.split_query_key_value_and_split_heads' op expected rank of query/key/value output tensor is 4, got query rank: 4, key rank: 3, value rank: 4
    %3, %4, %5 = "ttir.split_query_key_value_and_split_heads"(%arg0, %0, %1, %2) <{num_heads = 16 : ui32, transpose_key = false}> : (tensor<2x32x3072xf32>, tensor<2x16x32x64xf32>, tensor<2x16x32xf32>, tensor<2x16x32x64xf32>) -> (tensor<2x16x32x64xf32>, tensor<2x16x32xf32>, tensor<2x16x32x64xf32>)
    return %3, %4, %5 : tensor<2x16x32x64xf32>, tensor<2x16x32xf32>, tensor<2x16x32x64xf32>
  }
}

//-----

// Verify that the verifier fails if output value tensor is not 4D.
module {
  func.func @split_qkv_and_split_heads_invalid_2(%arg0: tensor<2x32x3072xf32>) -> (tensor<2x16x32x64xf32>, tensor<2x16x32xf32>, tensor<2x16x32xf32>) {
    %0 = ttir.empty() : tensor<2x16x32x64xf32>
    %1 = ttir.empty() : tensor<2x16x32x64xf32>
    %2 = ttir.empty() : tensor<2x16x32xf32>
    // CHECK:  error: 'ttir.split_query_key_value_and_split_heads' op expected rank of query/key/value output tensor is 4, got query rank: 4, key rank: 4, value rank: 3
    %3, %4, %5 = "ttir.split_query_key_value_and_split_heads"(%arg0, %0, %1, %2) <{num_heads = 16 : ui32, transpose_key = false}> : (tensor<2x32x3072xf32>, tensor<2x16x32x64xf32>, tensor<2x16x32x64xf32>, tensor<2x16x32xf32>) -> (tensor<2x16x32x64xf32>, tensor<2x16x32x64xf32>, tensor<2x16x32xf32>)
    return %3, %4, %5 : tensor<2x16x32x64xf32>, tensor<2x16x32x64xf32>, tensor<2x16x32xf32>
  }
}

//-----

// Verify that the verifier fails if output query has invalid shape.
module {
  func.func @split_qkv_invalid_query_shape(%arg0: tensor<2x32x3072xf32>) -> (tensor<4x16x32x64xf32>, tensor<2x16x32x64xf32>, tensor<2x16x32x64xf32>) {
    %0 = ttir.empty() : tensor<4x16x32x64xf32>
    %1 = ttir.empty() : tensor<2x16x32x64xf32>
    %2 = ttir.empty() : tensor<2x16x32x64xf32>
    // CHECK:  error: 'ttir.split_query_key_value_and_split_heads' op expected query output shape
    %3, %4, %5 = "ttir.split_query_key_value_and_split_heads"(%arg0, %0, %1, %2) <{num_heads = 16 : ui32, transpose_key = false}> : (tensor<2x32x3072xf32>, tensor<4x16x32x64xf32>, tensor<2x16x32x64xf32>, tensor<2x16x32x64xf32>) -> (tensor<4x16x32x64xf32>, tensor<2x16x32x64xf32>, tensor<2x16x32x64xf32>)
    return %3, %4, %5 : tensor<4x16x32x64xf32>, tensor<2x16x32x64xf32>, tensor<2x16x32x64xf32>
  }
}

//-----

// Verify that the verifier fails if output key has invalid shape.
module {
  func.func @split_qkv_invalid_key_shape(%arg0: tensor<2x32x3072xf32>) -> (tensor<2x16x32x64xf32>, tensor<4x16x32x64xf32>, tensor<2x16x32x64xf32>) {
    %0 = ttir.empty() : tensor<2x16x32x64xf32>
    %1 = ttir.empty() : tensor<4x16x32x64xf32>
    %2 = ttir.empty() : tensor<2x16x32x64xf32>
    // CHECK:  error: 'ttir.split_query_key_value_and_split_heads' op expected key output shape
    %3, %4, %5 = "ttir.split_query_key_value_and_split_heads"(%arg0, %0, %1, %2) <{num_heads = 16 : ui32, transpose_key = false}> : (tensor<2x32x3072xf32>, tensor<2x16x32x64xf32>, tensor<4x16x32x64xf32>, tensor<2x16x32x64xf32>) -> (tensor<2x16x32x64xf32>, tensor<4x16x32x64xf32>, tensor<2x16x32x64xf32>)
    return %3, %4, %5 : tensor<2x16x32x64xf32>, tensor<4x16x32x64xf32>, tensor<2x16x32x64xf32>
  }
}

//-----

// Verify that the verifier fails if output value has invalid shape.
module {
  func.func @split_qkv_invalid_value_shape(%arg0: tensor<2x32x3072xf32>) -> (tensor<2x16x32x64xf32>, tensor<2x16x32x64xf32>, tensor<4x16x32x64xf32>) {
    %0 = ttir.empty() : tensor<2x16x32x64xf32>
    %1 = ttir.empty() : tensor<2x16x32x64xf32>
    %2 = ttir.empty() : tensor<4x16x32x64xf32>
    // CHECK:  error: 'ttir.split_query_key_value_and_split_heads' op expected value output shape
    %3, %4, %5 = "ttir.split_query_key_value_and_split_heads"(%arg0, %0, %1, %2) <{num_heads = 16 : ui32, transpose_key = false}> : (tensor<2x32x3072xf32>, tensor<2x16x32x64xf32>, tensor<2x16x32x64xf32>, tensor<4x16x32x64xf32>) -> (tensor<2x16x32x64xf32>, tensor<2x16x32x64xf32>, tensor<4x16x32x64xf32>)
    return %3, %4, %5 : tensor<2x16x32x64xf32>, tensor<2x16x32x64xf32>, tensor<4x16x32x64xf32>
  }
}

//-----

// Verify that the verifier fails if output key has invalid shape because we passed transpose_key = true
module {
  func.func @split_qkv_invalid_key_shape_transpose(%arg0: tensor<2x32x3072xf32>) -> (tensor<2x16x32x64xf32>, tensor<2x16x32x64xf32>, tensor<2x16x32x64xf32>) {
    %0 = ttir.empty() : tensor<2x16x32x64xf32>
    %1 = ttir.empty() : tensor<2x16x32x64xf32>
    %2 = ttir.empty() : tensor<2x16x32x64xf32>
    // CHECK:  error: 'ttir.split_query_key_value_and_split_heads' op expected key output shape
    %3, %4, %5 = "ttir.split_query_key_value_and_split_heads"(%arg0, %0, %1, %2) <{num_heads = 16 : ui32, transpose_key = true}> : (tensor<2x32x3072xf32>, tensor<2x16x32x64xf32>, tensor<2x16x32x64xf32>, tensor<2x16x32x64xf32>) -> (tensor<2x16x32x64xf32>, tensor<2x16x32x64xf32>, tensor<2x16x32x64xf32>)
    return %3, %4, %5 : tensor<2x16x32x64xf32>, tensor<2x16x32x64xf32>, tensor<2x16x32x64xf32>
  }
}
