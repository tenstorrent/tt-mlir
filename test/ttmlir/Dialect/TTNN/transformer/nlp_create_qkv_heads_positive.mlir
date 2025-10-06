// RUN: ttmlir-opt --ttir-to-ttnn-backend-pipeline %s | FileCheck %s
module {
  // Test single input mode with fused QKV tensor
  func.func @nlp_create_qkv_heads_single_input(%arg0: tensor<1x1x32x4608xbf16>) -> (tensor<1x24x32x64xbf16>, tensor<1x24x64x32xbf16>, tensor<1x24x32x64xbf16>) {
    // CHECK: %[[QUERY:.*]], %[[KEY:.*]], %[[VALUE:.*]] = "ttnn.nlp_create_qkv_heads"(%arg0)
    %query, %key, %value = "ttnn.nlp_create_qkv_heads"(%arg0) <{
      num_q_heads = 24 : ui32,
      transpose_k_heads = true
    }> : (tensor<1x1x32x4608xbf16>) -> (tensor<1x24x32x64xbf16>, tensor<1x24x64x32xbf16>, tensor<1x24x32x64xbf16>)
    return %query, %key, %value : tensor<1x24x32x64xbf16>, tensor<1x24x64x32xbf16>, tensor<1x24x32x64xbf16>
  }

  // Test dual input mode with separate Q and KV tensors (GQA)
  func.func @nlp_create_qkv_heads_dual_input_gqa(%arg0: tensor<1x1x32x1536xbf16>, %arg1: tensor<1x1x32x1024xbf16>) -> (tensor<1x24x32x64xbf16>, tensor<1x8x64x32xbf16>, tensor<1x8x32x64xbf16>) {
    // CHECK: %[[QUERY:.*]], %[[KEY:.*]], %[[VALUE:.*]] = "ttnn.nlp_create_qkv_heads"(%arg0, %arg1)
    %query, %key, %value = "ttnn.nlp_create_qkv_heads"(%arg0, %arg1) <{
      num_q_heads = 24 : ui32,
      num_kv_heads = 8 : ui32,
      transpose_k_heads = true
    }> : (tensor<1x1x32x1536xbf16>, tensor<1x1x32x1024xbf16>) -> (tensor<1x24x32x64xbf16>, tensor<1x8x64x32xbf16>, tensor<1x8x32x64xbf16>)
    return %query, %key, %value : tensor<1x24x32x64xbf16>, tensor<1x8x64x32xbf16>, tensor<1x8x32x64xbf16>
  }

  // Test without key transpose
  func.func @nlp_create_qkv_heads_no_transpose(%arg0: tensor<1x1x32x4608xbf16>) -> (tensor<1x24x32x64xbf16>, tensor<1x24x32x64xbf16>, tensor<1x24x32x64xbf16>) {
    // CHECK: %[[QUERY:.*]], %[[KEY:.*]], %[[VALUE:.*]] = "ttnn.nlp_create_qkv_heads"(%arg0)
    %query, %key, %value = "ttnn.nlp_create_qkv_heads"(%arg0) <{
      num_q_heads = 24 : ui32,
      transpose_k_heads = false
    }> : (tensor<1x1x32x4608xbf16>) -> (tensor<1x24x32x64xbf16>, tensor<1x24x32x64xbf16>, tensor<1x24x32x64xbf16>)
    return %query, %key, %value : tensor<1x24x32x64xbf16>, tensor<1x24x32x64xbf16>, tensor<1x24x32x64xbf16>
  }

  // Test with 32 heads and 128 head dimension
  func.func @nlp_create_qkv_heads_32_heads(%arg0: tensor<1x1x32x12288xbf16>) -> (tensor<1x32x32x128xbf16>, tensor<1x32x128x32xbf16>, tensor<1x32x32x128xbf16>) {
    // CHECK: %[[QUERY:.*]], %[[KEY:.*]], %[[VALUE:.*]] = "ttnn.nlp_create_qkv_heads"(%arg0)
    %query, %key, %value = "ttnn.nlp_create_qkv_heads"(%arg0) <{
      num_q_heads = 32 : ui32,
      transpose_k_heads = true
    }> : (tensor<1x1x32x12288xbf16>) -> (tensor<1x32x32x128xbf16>, tensor<1x32x128x32xbf16>, tensor<1x32x32x128xbf16>)
    return %query, %key, %value : tensor<1x32x32x128xbf16>, tensor<1x32x128x32xbf16>, tensor<1x32x32x128xbf16>
  }

  // Test with batch size 2 and longer sequence
  func.func @nlp_create_qkv_heads_batch_2(%arg0: tensor<2x1x128x4608xbf16>) -> (tensor<2x24x128x64xbf16>, tensor<2x24x64x128xbf16>, tensor<2x24x128x64xbf16>) {
    // CHECK: %[[QUERY:.*]], %[[KEY:.*]], %[[VALUE:.*]] = "ttnn.nlp_create_qkv_heads"(%arg0)
    %query, %key, %value = "ttnn.nlp_create_qkv_heads"(%arg0) <{
      num_q_heads = 24 : ui32,
      transpose_k_heads = true
    }> : (tensor<2x1x128x4608xbf16>) -> (tensor<2x24x128x64xbf16>, tensor<2x24x64x128xbf16>, tensor<2x24x128x64xbf16>)
    return %query, %key, %value : tensor<2x24x128x64xbf16>, tensor<2x24x64x128xbf16>, tensor<2x24x128x64xbf16>
  }

  // Test with smaller head dimension (32)
  func.func @nlp_create_qkv_heads_small_head(%arg0: tensor<1x1x64x2304xbf16>) -> (tensor<1x24x64x32xbf16>, tensor<1x24x32x64xbf16>, tensor<1x24x64x32xbf16>) {
    // CHECK: %[[QUERY:.*]], %[[KEY:.*]], %[[VALUE:.*]] = "ttnn.nlp_create_qkv_heads"(%arg0)
    %query, %key, %value = "ttnn.nlp_create_qkv_heads"(%arg0) <{
      num_q_heads = 24 : ui32,
      transpose_k_heads = true
    }> : (tensor<1x1x64x2304xbf16>) -> (tensor<1x24x64x32xbf16>, tensor<1x24x32x64xbf16>, tensor<1x24x64x32xbf16>)
    return %query, %key, %value : tensor<1x24x64x32xbf16>, tensor<1x24x32x64xbf16>, tensor<1x24x64x32xbf16>
  }

  // Test dual input with equal Q and KV heads (MHA)
  func.func @nlp_create_qkv_heads_mha(%arg0: tensor<1x1x32x1536xbf16>, %arg1: tensor<1x1x32x3072xbf16>) -> (tensor<1x24x32x64xbf16>, tensor<1x24x64x32xbf16>, tensor<1x24x32x64xbf16>) {
    // CHECK: %[[QUERY:.*]], %[[KEY:.*]], %[[VALUE:.*]] = "ttnn.nlp_create_qkv_heads"(%arg0, %arg1)
    %query, %key, %value = "ttnn.nlp_create_qkv_heads"(%arg0, %arg1) <{
      num_q_heads = 24 : ui32,
      num_kv_heads = 24 : ui32,
      transpose_k_heads = true
    }> : (tensor<1x1x32x1536xbf16>, tensor<1x1x32x3072xbf16>) -> (tensor<1x24x32x64xbf16>, tensor<1x24x64x32xbf16>, tensor<1x24x32x64xbf16>)
    return %query, %key, %value : tensor<1x24x32x64xbf16>, tensor<1x24x64x32xbf16>, tensor<1x24x32x64xbf16>
  }

  // Test with 8 heads (fewer heads)
  func.func @nlp_create_qkv_heads_8_heads(%arg0: tensor<1x1x32x1536xbf16>) -> (tensor<1x8x32x64xbf16>, tensor<1x8x64x32xbf16>, tensor<1x8x32x64xbf16>) {
    // CHECK: %[[QUERY:.*]], %[[KEY:.*]], %[[VALUE:.*]] = "ttnn.nlp_create_qkv_heads"(%arg0)
    %query, %key, %value = "ttnn.nlp_create_qkv_heads"(%arg0) <{
      num_q_heads = 8 : ui32,
      transpose_k_heads = true
    }> : (tensor<1x1x32x1536xbf16>) -> (tensor<1x8x32x64xbf16>, tensor<1x8x64x32xbf16>, tensor<1x8x32x64xbf16>)
    return %query, %key, %value : tensor<1x8x32x64xbf16>, tensor<1x8x64x32xbf16>, tensor<1x8x32x64xbf16>
  }
}
