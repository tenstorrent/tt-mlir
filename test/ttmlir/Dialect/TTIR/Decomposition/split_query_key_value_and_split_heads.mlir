// RUN: ttmlir-opt --ttir-to-ttir-decomposition="config=cpu-fallback" -o %t %s
// RUN: FileCheck %s --input-file=%t

// Test 1: MHA case - fused QKV tensor, no transpose_key.
module {
  func.func @split_qkv_mha_no_transpose(%arg0: tensor<2x128x768xf32>) -> (tensor<2x8x128x32xf32>, tensor<2x8x128x32xf32>, tensor<2x8x128x32xf32>) {
    // CHECK-LABEL: func.func @split_qkv_mha_no_transpose
    // CHECK-NOT: ttir.split_query_key_value_and_split_heads

    // Slice Q, K, V from fused tensor.
    // CHECK: "ttir.slice_static"
    // CHECK: "ttir.slice_static"
    // CHECK: "ttir.slice_static"

    // Reshape Q, K, V: [batch, seq, hidden] -> [batch, seq, num_heads, head_size].
    // CHECK: "ttir.reshape"
    // CHECK: "ttir.reshape"
    // CHECK: "ttir.reshape"

    // Permute Q, K, V: [batch, seq, num_heads, head_size] -> [batch, num_heads, seq, head_size].
    // CHECK: "ttir.permute"
    // CHECK: "ttir.permute"
    // CHECK: "ttir.permute"

    %0, %1, %2 = "ttir.split_query_key_value_and_split_heads"(%arg0) <{
      num_heads = 8 : ui32,
      transpose_key = false
    }> : (tensor<2x128x768xf32>) -> (tensor<2x8x128x32xf32>, tensor<2x8x128x32xf32>, tensor<2x8x128x32xf32>)
    return %0, %1, %2 : tensor<2x8x128x32xf32>, tensor<2x8x128x32xf32>, tensor<2x8x128x32xf32>
  }
}

// Test 2: MHA case - fused QKV tensor, with transpose_key.
module {
  func.func @split_qkv_mha_transpose_key(%arg0: tensor<2x128x768xf32>) -> (tensor<2x8x128x32xf32>, tensor<2x8x32x128xf32>, tensor<2x8x128x32xf32>) {
    // CHECK-LABEL: func.func @split_qkv_mha_transpose_key
    // CHECK-NOT: ttir.split_query_key_value_and_split_heads

    // With transpose_key=true, we get 4 permutes (extra one for K).
    // CHECK: "ttir.permute"
    // CHECK: "ttir.permute"
    // CHECK: "ttir.permute"
    // CHECK: "ttir.permute"

    %0, %1, %2 = "ttir.split_query_key_value_and_split_heads"(%arg0) <{
      num_heads = 8 : ui32,
      transpose_key = true
    }> : (tensor<2x128x768xf32>) -> (tensor<2x8x128x32xf32>, tensor<2x8x32x128xf32>, tensor<2x8x128x32xf32>)
    return %0, %1, %2 : tensor<2x8x128x32xf32>, tensor<2x8x32x128xf32>, tensor<2x8x128x32xf32>
  }
}

// Test 3: GQA case - separate Q and KV tensors.
module {
  func.func @split_qkv_gqa(%arg0: tensor<2x128x256xf32>, %arg1: tensor<2x128x128xf32>) -> (tensor<2x8x128x32xf32>, tensor<2x2x128x32xf32>, tensor<2x2x128x32xf32>) {
    // CHECK-LABEL: func.func @split_qkv_gqa
    // CHECK-NOT: ttir.split_query_key_value_and_split_heads

    // Q: reshape + permute (no slice needed for Q in GQA).
    // CHECK: "ttir.reshape"
    // CHECK: "ttir.permute"

    // K, V: slice + reshape + permute each.
    // CHECK: "ttir.slice_static"
    // CHECK: "ttir.slice_static"
    // CHECK: "ttir.reshape"
    // CHECK: "ttir.reshape"
    // CHECK: "ttir.permute"
    // CHECK: "ttir.permute"

    %0, %1, %2 = "ttir.split_query_key_value_and_split_heads"(%arg0, %arg1) <{
      num_heads = 8 : ui32,
      num_kv_heads = 2 : ui32,
      transpose_key = false
    }> : (tensor<2x128x256xf32>, tensor<2x128x128xf32>) -> (tensor<2x8x128x32xf32>, tensor<2x2x128x32xf32>, tensor<2x2x128x32xf32>)
    return %0, %1, %2 : tensor<2x8x128x32xf32>, tensor<2x2x128x32xf32>, tensor<2x2x128x32xf32>
  }
}
