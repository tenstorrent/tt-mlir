// RUN: ttmlir-opt --ttir-to-ttnn-backend-pipeline %s | FileCheck %s
module {
  func.func @nlp_create_qkv_heads_single_basic_transposed(%arg0: tensor<1x1x32x4608xbf16>) -> (tensor<1x24x32x64xbf16>, tensor<1x24x64x32xbf16>, tensor<1x24x32x64xbf16>) {
    // CHECK: "ttnn.nlp_create_qkv_heads"(%arg0)
    %query, %key, %value = "ttnn.nlp_create_qkv_heads"(%arg0) <{
      num_q_heads = 24 : ui32,
      transpose_k_heads = true
    }> : (tensor<1x1x32x4608xbf16>) -> (tensor<1x24x32x64xbf16>, tensor<1x24x64x32xbf16>, tensor<1x24x32x64xbf16>)
    return %query, %key, %value : tensor<1x24x32x64xbf16>, tensor<1x24x64x32xbf16>, tensor<1x24x32x64xbf16>
  }

  func.func @nlp_create_qkv_heads_single_basic_not_transposed(%arg0: tensor<1x1x32x4608xbf16>) -> (tensor<1x24x32x64xbf16>, tensor<1x24x32x64xbf16>, tensor<1x24x32x64xbf16>) {
    // CHECK: "ttnn.nlp_create_qkv_heads"(%arg0)
    %query, %key, %value = "ttnn.nlp_create_qkv_heads"(%arg0) <{
      num_q_heads = 24 : ui32,
      transpose_k_heads = false
    }> : (tensor<1x1x32x4608xbf16>) -> (tensor<1x24x32x64xbf16>, tensor<1x24x32x64xbf16>, tensor<1x24x32x64xbf16>)
    return %query, %key, %value : tensor<1x24x32x64xbf16>, tensor<1x24x32x64xbf16>, tensor<1x24x32x64xbf16>
  }

  func.func @nlp_create_qkv_heads_single_batch_2(%arg0: tensor<2x1x32x4608xbf16>) -> (tensor<2x24x32x64xbf16>, tensor<2x24x64x32xbf16>, tensor<2x24x32x64xbf16>) {
    // CHECK: "ttnn.nlp_create_qkv_heads"(%arg0)
    %query, %key, %value = "ttnn.nlp_create_qkv_heads"(%arg0) <{
      num_q_heads = 24 : ui32,
      transpose_k_heads = true
    }> : (tensor<2x1x32x4608xbf16>) -> (tensor<2x24x32x64xbf16>, tensor<2x24x64x32xbf16>, tensor<2x24x32x64xbf16>)
    return %query, %key, %value : tensor<2x24x32x64xbf16>, tensor<2x24x64x32xbf16>, tensor<2x24x32x64xbf16>
  }

  func.func @nlp_create_qkv_heads_single_seq_64(%arg0: tensor<1x1x64x4608xbf16>) -> (tensor<1x24x64x64xbf16>, tensor<1x24x64x64xbf16>, tensor<1x24x64x64xbf16>) {
    // CHECK: "ttnn.nlp_create_qkv_heads"(%arg0)
    %query, %key, %value = "ttnn.nlp_create_qkv_heads"(%arg0) <{
      num_q_heads = 24 : ui32,
      transpose_k_heads = true
    }> : (tensor<1x1x64x4608xbf16>) -> (tensor<1x24x64x64xbf16>, tensor<1x24x64x64xbf16>, tensor<1x24x64x64xbf16>)
    return %query, %key, %value : tensor<1x24x64x64xbf16>, tensor<1x24x64x64xbf16>, tensor<1x24x64x64xbf16>
  }

  func.func @nlp_create_qkv_heads_single_8_heads(%arg0: tensor<1x1x32x1536xbf16>) -> (tensor<1x8x32x64xbf16>, tensor<1x8x64x32xbf16>, tensor<1x8x32x64xbf16>) {
    // CHECK: "ttnn.nlp_create_qkv_heads"(%arg0)
    %query, %key, %value = "ttnn.nlp_create_qkv_heads"(%arg0) <{
      num_q_heads = 8 : ui32,
      transpose_k_heads = true
    }> : (tensor<1x1x32x1536xbf16>) -> (tensor<1x8x32x64xbf16>, tensor<1x8x64x32xbf16>, tensor<1x8x32x64xbf16>)
    return %query, %key, %value : tensor<1x8x32x64xbf16>, tensor<1x8x64x32xbf16>, tensor<1x8x32x64xbf16>
  }

  func.func @nlp_create_qkv_heads_single_head_128(%arg0: tensor<1x1x32x12288xbf16>) -> (tensor<1x32x32x128xbf16>, tensor<1x32x128x32xbf16>, tensor<1x32x32x128xbf16>) {
    // CHECK: "ttnn.nlp_create_qkv_heads"(%arg0)
    %query, %key, %value = "ttnn.nlp_create_qkv_heads"(%arg0) <{
      num_q_heads = 32 : ui32,
      transpose_k_heads = true
    }> : (tensor<1x1x32x12288xbf16>) -> (tensor<1x32x32x128xbf16>, tensor<1x32x128x32xbf16>, tensor<1x32x32x128xbf16>)
    return %query, %key, %value : tensor<1x32x32x128xbf16>, tensor<1x32x128x32xbf16>, tensor<1x32x32x128xbf16>
  }

  func.func @nlp_create_qkv_heads_single_1_head(%arg0: tensor<1x1x32x192xbf16>) -> (tensor<1x1x32x64xbf16>, tensor<1x1x64x32xbf16>, tensor<1x1x32x64xbf16>) {
    // CHECK: "ttnn.nlp_create_qkv_heads"(%arg0)
    %query, %key, %value = "ttnn.nlp_create_qkv_heads"(%arg0) <{
      num_q_heads = 1 : ui32,
      transpose_k_heads = true
    }> : (tensor<1x1x32x192xbf16>) -> (tensor<1x1x32x64xbf16>, tensor<1x1x64x32xbf16>, tensor<1x1x32x64xbf16>)
    return %query, %key, %value : tensor<1x1x32x64xbf16>, tensor<1x1x64x32xbf16>, tensor<1x1x32x64xbf16>
  }

  func.func @nlp_create_qkv_heads_single_head_32(%arg0: tensor<1x1x32x1536xbf16>) -> (tensor<1x16x32x32xbf16>, tensor<1x16x32x32xbf16>, tensor<1x16x32x32xbf16>) {
    // CHECK: "ttnn.nlp_create_qkv_heads"(%arg0)
    %query, %key, %value = "ttnn.nlp_create_qkv_heads"(%arg0) <{
      num_q_heads = 16 : ui32,
      transpose_k_heads = true
    }> : (tensor<1x1x32x1536xbf16>) -> (tensor<1x16x32x32xbf16>, tensor<1x16x32x32xbf16>, tensor<1x16x32x32xbf16>)
    return %query, %key, %value : tensor<1x16x32x32xbf16>, tensor<1x16x32x32xbf16>, tensor<1x16x32x32xbf16>
  }

  func.func @nlp_create_qkv_heads_single_large(%arg0: tensor<4x1x128x4608xbf16>) -> (tensor<4x24x128x64xbf16>, tensor<4x24x64x128xbf16>, tensor<4x24x128x64xbf16>) {
    // CHECK: "ttnn.nlp_create_qkv_heads"(%arg0)
    %query, %key, %value = "ttnn.nlp_create_qkv_heads"(%arg0) <{
      num_q_heads = 24 : ui32,
      transpose_k_heads = true
    }> : (tensor<4x1x128x4608xbf16>) -> (tensor<4x24x128x64xbf16>, tensor<4x24x64x128xbf16>, tensor<4x24x128x64xbf16>)
    return %query, %key, %value : tensor<4x24x128x64xbf16>, tensor<4x24x64x128xbf16>, tensor<4x24x128x64xbf16>
  }

  func.func @nlp_create_qkv_heads_single_f32(%arg0: tensor<1x1x32x4608xf32>) -> (tensor<1x24x32x64xf32>, tensor<1x24x64x32xf32>, tensor<1x24x32x64xf32>) {
    // CHECK: "ttnn.nlp_create_qkv_heads"(%arg0)
    %query, %key, %value = "ttnn.nlp_create_qkv_heads"(%arg0) <{
      num_q_heads = 24 : ui32,
      transpose_k_heads = true
    }> : (tensor<1x1x32x4608xf32>) -> (tensor<1x24x32x64xf32>, tensor<1x24x64x32xf32>, tensor<1x24x32x64xf32>)
    return %query, %key, %value : tensor<1x24x32x64xf32>, tensor<1x24x64x32xf32>, tensor<1x24x32x64xf32>
  }

  func.func @nlp_create_qkv_heads_dual_basic_gqa(%arg0: tensor<1x1x32x1536xbf16>, %arg1: tensor<1x1x32x1024xbf16>) -> (tensor<1x24x32x64xbf16>, tensor<1x8x64x32xbf16>, tensor<1x8x32x64xbf16>) {
    // CHECK: "ttnn.nlp_create_qkv_heads"(%arg0, %arg1)
    %query, %key, %value = "ttnn.nlp_create_qkv_heads"(%arg0, %arg1) <{
      num_q_heads = 24 : ui32,
      num_kv_heads = 8 : ui32,
      transpose_k_heads = true
    }> : (tensor<1x1x32x1536xbf16>, tensor<1x1x32x1024xbf16>) -> (tensor<1x24x32x64xbf16>, tensor<1x8x64x32xbf16>, tensor<1x8x32x64xbf16>)
    return %query, %key, %value : tensor<1x24x32x64xbf16>, tensor<1x8x64x32xbf16>, tensor<1x8x32x64xbf16>
  }

  func.func @nlp_create_qkv_heads_dual_not_transposed(%arg0: tensor<1x1x32x1536xbf16>, %arg1: tensor<1x1x32x1024xbf16>) -> (tensor<1x24x32x64xbf16>, tensor<1x8x32x64xbf16>, tensor<1x8x32x64xbf16>) {
    // CHECK: "ttnn.nlp_create_qkv_heads"(%arg0, %arg1)
    %query, %key, %value = "ttnn.nlp_create_qkv_heads"(%arg0, %arg1) <{
      num_q_heads = 24 : ui32,
      num_kv_heads = 8 : ui32,
      transpose_k_heads = false
    }> : (tensor<1x1x32x1536xbf16>, tensor<1x1x32x1024xbf16>) -> (tensor<1x24x32x64xbf16>, tensor<1x8x32x64xbf16>, tensor<1x8x32x64xbf16>)
    return %query, %key, %value : tensor<1x24x32x64xbf16>, tensor<1x8x32x64xbf16>, tensor<1x8x32x64xbf16>
  }

  func.func @nlp_create_qkv_heads_dual_mqa(%arg0: tensor<1x1x32x1536xbf16>, %arg1: tensor<1x1x32x128xbf16>) -> (tensor<1x24x32x64xbf16>, tensor<1x1x64x32xbf16>, tensor<1x1x32x64xbf16>) {
    // CHECK: "ttnn.nlp_create_qkv_heads"(%arg0, %arg1)
    %query, %key, %value = "ttnn.nlp_create_qkv_heads"(%arg0, %arg1) <{
      num_q_heads = 24 : ui32,
      num_kv_heads = 1 : ui32,
      transpose_k_heads = true
    }> : (tensor<1x1x32x1536xbf16>, tensor<1x1x32x128xbf16>) -> (tensor<1x24x32x64xbf16>, tensor<1x1x64x32xbf16>, tensor<1x1x32x64xbf16>)
    return %query, %key, %value : tensor<1x24x32x64xbf16>, tensor<1x1x64x32xbf16>, tensor<1x1x32x64xbf16>
  }

  func.func @nlp_create_qkv_heads_dual_equal_heads(%arg0: tensor<1x1x32x1536xbf16>, %arg1: tensor<1x1x32x3072xbf16>) -> (tensor<1x24x32x64xbf16>, tensor<1x24x64x32xbf16>, tensor<1x24x32x64xbf16>) {
    // CHECK: "ttnn.nlp_create_qkv_heads"(%arg0, %arg1)
    %query, %key, %value = "ttnn.nlp_create_qkv_heads"(%arg0, %arg1) <{
      num_q_heads = 24 : ui32,
      num_kv_heads = 24 : ui32,
      transpose_k_heads = true
    }> : (tensor<1x1x32x1536xbf16>, tensor<1x1x32x3072xbf16>) -> (tensor<1x24x32x64xbf16>, tensor<1x24x64x32xbf16>, tensor<1x24x32x64xbf16>)
    return %query, %key, %value : tensor<1x24x32x64xbf16>, tensor<1x24x64x32xbf16>, tensor<1x24x32x64xbf16>
  }

  func.func @nlp_create_qkv_heads_dual_batch_4(%arg0: tensor<4x1x32x1536xbf16>, %arg1: tensor<4x1x32x1024xbf16>) -> (tensor<4x24x32x64xbf16>, tensor<4x8x64x32xbf16>, tensor<4x8x32x64xbf16>) {
    // CHECK: "ttnn.nlp_create_qkv_heads"(%arg0, %arg1)
    %query, %key, %value = "ttnn.nlp_create_qkv_heads"(%arg0, %arg1) <{
      num_q_heads = 24 : ui32,
      num_kv_heads = 8 : ui32,
      transpose_k_heads = true
    }> : (tensor<4x1x32x1536xbf16>, tensor<4x1x32x1024xbf16>) -> (tensor<4x24x32x64xbf16>, tensor<4x8x64x32xbf16>, tensor<4x8x32x64xbf16>)
    return %query, %key, %value : tensor<4x24x32x64xbf16>, tensor<4x8x64x32xbf16>, tensor<4x8x32x64xbf16>
  }

  func.func @nlp_create_qkv_heads_dual_seq_128(%arg0: tensor<1x1x128x1536xbf16>, %arg1: tensor<1x1x128x1024xbf16>) -> (tensor<1x24x128x64xbf16>, tensor<1x8x64x128xbf16>, tensor<1x8x128x64xbf16>) {
    // CHECK: "ttnn.nlp_create_qkv_heads"(%arg0, %arg1)
    %query, %key, %value = "ttnn.nlp_create_qkv_heads"(%arg0, %arg1) <{
      num_q_heads = 24 : ui32,
      num_kv_heads = 8 : ui32,
      transpose_k_heads = true
    }> : (tensor<1x1x128x1536xbf16>, tensor<1x1x128x1024xbf16>) -> (tensor<1x24x128x64xbf16>, tensor<1x8x64x128xbf16>, tensor<1x8x128x64xbf16>)
    return %query, %key, %value : tensor<1x24x128x64xbf16>, tensor<1x8x64x128xbf16>, tensor<1x8x128x64xbf16>
  }

  func.func @nlp_create_qkv_heads_dual_head_128(%arg0: tensor<1x1x32x3072xbf16>, %arg1: tensor<1x1x32x2048xbf16>) -> (tensor<1x24x32x128xbf16>, tensor<1x8x128x32xbf16>, tensor<1x8x32x128xbf16>) {
    // CHECK: "ttnn.nlp_create_qkv_heads"(%arg0, %arg1)
    %query, %key, %value = "ttnn.nlp_create_qkv_heads"(%arg0, %arg1) <{
      num_q_heads = 24 : ui32,
      num_kv_heads = 8 : ui32,
      transpose_k_heads = true
    }> : (tensor<1x1x32x3072xbf16>, tensor<1x1x32x2048xbf16>) -> (tensor<1x24x32x128xbf16>, tensor<1x8x128x32xbf16>, tensor<1x8x32x128xbf16>)
    return %query, %key, %value : tensor<1x24x32x128xbf16>, tensor<1x8x128x32xbf16>, tensor<1x8x32x128xbf16>
  }

  func.func @nlp_create_qkv_heads_dual_32_4_heads(%arg0: tensor<1x1x32x2048xbf16>, %arg1: tensor<1x1x32x512xbf16>) -> (tensor<1x32x32x64xbf16>, tensor<1x4x64x32xbf16>, tensor<1x4x32x64xbf16>) {
    // CHECK: "ttnn.nlp_create_qkv_heads"(%arg0, %arg1)
    %query, %key, %value = "ttnn.nlp_create_qkv_heads"(%arg0, %arg1) <{
      num_q_heads = 32 : ui32,
      num_kv_heads = 4 : ui32,
      transpose_k_heads = true
    }> : (tensor<1x1x32x2048xbf16>, tensor<1x1x32x512xbf16>) -> (tensor<1x32x32x64xbf16>, tensor<1x4x64x32xbf16>, tensor<1x4x32x64xbf16>)
    return %query, %key, %value : tensor<1x32x32x64xbf16>, tensor<1x4x64x32xbf16>, tensor<1x4x32x64xbf16>
  }

  func.func @nlp_create_qkv_heads_dual_16_2_heads(%arg0: tensor<1x1x32x1024xbf16>, %arg1: tensor<1x1x32x256xbf16>) -> (tensor<1x16x32x64xbf16>, tensor<1x2x64x32xbf16>, tensor<1x2x32x64xbf16>) {
    // CHECK: "ttnn.nlp_create_qkv_heads"(%arg0, %arg1)
    %query, %key, %value = "ttnn.nlp_create_qkv_heads"(%arg0, %arg1) <{
      num_q_heads = 16 : ui32,
      num_kv_heads = 2 : ui32,
      transpose_k_heads = true
    }> : (tensor<1x1x32x1024xbf16>, tensor<1x1x32x256xbf16>) -> (tensor<1x16x32x64xbf16>, tensor<1x2x64x32xbf16>, tensor<1x2x32x64xbf16>)
    return %query, %key, %value : tensor<1x16x32x64xbf16>, tensor<1x2x64x32xbf16>, tensor<1x2x32x64xbf16>
  }

  func.func @nlp_create_qkv_heads_dual_f32(%arg0: tensor<1x1x32x1536xf32>, %arg1: tensor<1x1x32x1024xf32>) -> (tensor<1x24x32x64xf32>, tensor<1x8x64x32xf32>, tensor<1x8x32x64xf32>) {
    // CHECK: "ttnn.nlp_create_qkv_heads"(%arg0, %arg1)
    %query, %key, %value = "ttnn.nlp_create_qkv_heads"(%arg0, %arg1) <{
      num_q_heads = 24 : ui32,
      num_kv_heads = 8 : ui32,
      transpose_k_heads = true
    }> : (tensor<1x1x32x1536xf32>, tensor<1x1x32x1024xf32>) -> (tensor<1x24x32x64xf32>, tensor<1x8x64x32xf32>, tensor<1x8x32x64xf32>)
    return %query, %key, %value : tensor<1x24x32x64xf32>, tensor<1x8x64x32xf32>, tensor<1x8x32x64xf32>
  }

  func.func @nlp_create_qkv_heads_min_config(%arg0: tensor<1x1x2x96xbf16>, %arg1: tensor<1x1x2x64xbf16>) -> (tensor<1x3x2x32xbf16>, tensor<1x1x32x2xbf16>, tensor<1x1x2x32xbf16>) {
    // CHECK: "ttnn.nlp_create_qkv_heads"(%arg0, %arg1)
    %query, %key, %value = "ttnn.nlp_create_qkv_heads"(%arg0, %arg1) <{
      num_q_heads = 3 : ui32,
      num_kv_heads = 1 : ui32,
      transpose_k_heads = true
    }> : (tensor<1x1x2x96xbf16>, tensor<1x1x2x64xbf16>) -> (tensor<1x3x2x32xbf16>, tensor<1x1x32x2xbf16>, tensor<1x1x2x32xbf16>)
    return %query, %key, %value : tensor<1x3x2x32xbf16>, tensor<1x1x32x2xbf16>, tensor<1x1x2x32xbf16>
  }

  func.func @nlp_create_qkv_heads_large_config(%arg0: tensor<8x1x256x2048xbf16>, %arg1: tensor<8x1x256x2048xbf16>) -> (tensor<8x32x256x64xbf16>, tensor<8x16x64x256xbf16>, tensor<8x16x256x64xbf16>) {
    // CHECK: "ttnn.nlp_create_qkv_heads"(%arg0, %arg1)
    %query, %key, %value = "ttnn.nlp_create_qkv_heads"(%arg0, %arg1) <{
      num_q_heads = 32 : ui32,
      num_kv_heads = 16 : ui32,
      transpose_k_heads = true
    }> : (tensor<8x1x256x2048xbf16>, tensor<8x1x256x2048xbf16>) -> (tensor<8x32x256x64xbf16>, tensor<8x16x64x256xbf16>, tensor<8x16x256x64xbf16>)
    return %query, %key, %value : tensor<8x32x256x64xbf16>, tensor<8x16x64x256xbf16>, tensor<8x16x256x64xbf16>
  }

  func.func @nlp_create_qkv_heads_llama_style(%arg0: tensor<1x1x32x4096xbf16>, %arg1: tensor<1x1x32x2048xbf16>) -> (tensor<1x32x32x128xbf16>, tensor<1x8x128x32xbf16>, tensor<1x8x32x128xbf16>) {
    // CHECK: "ttnn.nlp_create_qkv_heads"(%arg0, %arg1)
    %query, %key, %value = "ttnn.nlp_create_qkv_heads"(%arg0, %arg1) <{
      num_q_heads = 32 : ui32,
      num_kv_heads = 8 : ui32,
      transpose_k_heads = true
    }> : (tensor<1x1x32x4096xbf16>, tensor<1x1x32x2048xbf16>) -> (tensor<1x32x32x128xbf16>, tensor<1x8x128x32xbf16>, tensor<1x8x32x128xbf16>)
    return %query, %key, %value : tensor<1x32x32x128xbf16>, tensor<1x8x128x32xbf16>, tensor<1x8x32x128xbf16>
  }

  func.func @nlp_create_qkv_heads_gpt_style(%arg0: tensor<1x1x512x2304xbf16>) -> (tensor<1x12x512x64xbf16>, tensor<1x12x64x512xbf16>, tensor<1x12x512x64xbf16>) {
    // CHECK: "ttnn.nlp_create_qkv_heads"(%arg0)
    %query, %key, %value = "ttnn.nlp_create_qkv_heads"(%arg0) <{
      num_q_heads = 12 : ui32,
      transpose_k_heads = true
    }> : (tensor<1x1x512x2304xbf16>) -> (tensor<1x12x512x64xbf16>, tensor<1x12x64x512xbf16>, tensor<1x12x512x64xbf16>)
    return %query, %key, %value : tensor<1x12x512x64xbf16>, tensor<1x12x64x512xbf16>, tensor<1x12x512x64xbf16>
  }

  func.func @nlp_create_qkv_heads_bert_style(%arg0: tensor<1x1x512x2304xbf16>) -> (tensor<1x12x512x64xbf16>, tensor<1x12x512x64xbf16>, tensor<1x12x512x64xbf16>) {
    // CHECK: "ttnn.nlp_create_qkv_heads"(%arg0)
    %query, %key, %value = "ttnn.nlp_create_qkv_heads"(%arg0) <{
      num_q_heads = 12 : ui32,
      transpose_k_heads = false
    }> : (tensor<1x1x512x2304xbf16>) -> (tensor<1x12x512x64xbf16>, tensor<1x12x512x64xbf16>, tensor<1x12x512x64xbf16>)
    return %query, %key, %value : tensor<1x12x512x64xbf16>, tensor<1x12x512x64xbf16>, tensor<1x12x512x64xbf16>
  }
}
