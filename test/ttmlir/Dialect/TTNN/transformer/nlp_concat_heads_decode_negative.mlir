// RUN: not ttmlir-opt --split-input-file %s 2>&1 | FileCheck %s
// Negative tests for nlp_concat_heads_decode operation.

// Verify that the parsing fails if input tensor is not 4D.
module {
  func.func @nlp_concat_heads_decode_invalid_rank_input(%arg0: tensor<1x32x128xbf16>) -> tensor<1x1x1x4096xbf16> {
    // CHECK: error: 'ttnn.nlp_concat_heads_decode' op input tensor must be a 4D tensor
    %0 = "ttnn.nlp_concat_heads_decode"(%arg0) {num_heads = 32 : ui32} : (tensor<1x32x128xbf16>) -> tensor<1x1x1x4096xbf16>
    return %0 : tensor<1x1x1x4096xbf16>
  }
}

// -----

// Verify that the parsing fails if output tensor is not 4D.
module {
  func.func @nlp_concat_heads_decode_invalid_rank_output(%arg0: tensor<1x1x32x128xbf16>) -> tensor<1x1x4096xbf16> {
    // CHECK: error: 'ttnn.nlp_concat_heads_decode' op output tensor must be a 4D tensor
    %0 = "ttnn.nlp_concat_heads_decode"(%arg0) {num_heads = 32 : ui32} : (tensor<1x1x32x128xbf16>) -> tensor<1x1x4096xbf16>
    return %0 : tensor<1x1x4096xbf16>
  }
}


// -----

// Verify that the parsing fails if sequence dimensions do not match.
module {
  func.func @nlp_concat_heads_decode_invalid_sequence(%arg0: tensor<2x1x32x128xbf16>) -> tensor<3x1x1x4096xbf16> {
    // CHECK: error: 'ttnn.nlp_concat_heads_decode' op input sequence dimension must match output sequence dimension
    %0 = "ttnn.nlp_concat_heads_decode"(%arg0) {num_heads = 32 : ui32} : (tensor<2x1x32x128xbf16>) -> tensor<3x1x1x4096xbf16>
    return %0 : tensor<3x1x1x4096xbf16>
  }
}

// -----

// Verify that the parsing fails if num_heads * head_size does not match output hidden dimension.
module {
  func.func @nlp_concat_heads_decode_invalid_hidden_size(%arg0: tensor<1x1x32x128xbf16>) -> tensor<1x1x1x4096xbf16> {
    // CHECK: error: 'ttnn.nlp_concat_heads_decode' op Output hidden dimension must equal num_heads * head_size, got num_heads = 32, head_size = 16, expected hidden size = 2048, actual output hidden size = 4096
    %0 = "ttnn.nlp_concat_heads_decode"(%arg0) {num_heads = 16 : ui32} : (tensor<1x1x32x128xbf16>) -> tensor<1x1x1x4096xbf16>
    return %0 : tensor<1x1x1x4096xbf16>
  }
}

// -----

// Verify that output dim 1 is 1
module {
  func.func @nlp_concat_heads_decode_invalid_dim1(%arg0: tensor<1x1x32x128xbf16>) -> tensor<1x32x1x4096xbf16> {
    // CHECK: error: 'ttnn.nlp_concat_heads_decode' op output dimension 1 must be 1, got 32
    %0 = "ttnn.nlp_concat_heads_decode"(%arg0) {num_heads = 32 : ui32} : (tensor<1x1x32x128xbf16>) -> tensor<1x32x1x4096xbf16>
    return %0 : tensor<1x32x1x4096xbf16>
  }
}

// -----

// Verify that output dim 1 is 1
module {
  func.func @nlp_concat_heads_invalid_head_size(%arg0: tensor<1x1x32x128xbf16>) -> tensor<1x1x1x4096xbf16> {
    // CHECK: error: 'ttnn.nlp_concat_heads_decode' op num_heads attribute must be less
    %0 = "ttnn.nlp_concat_heads_decode"(%arg0) {num_heads = 34 : ui32} : (tensor<1x1x32x128xbf16>) -> tensor<1x1x1x4096xbf16>
    return %0 : tensor<1x1x1x4096xbf16>
  }
}
