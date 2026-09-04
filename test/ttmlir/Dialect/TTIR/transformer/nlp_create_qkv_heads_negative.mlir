// RUN: not ttmlir-opt --split-input-file %s 2>&1 | FileCheck %s

module {
  func.func @nlp_create_qkv_heads_rank3(%arg0: tensor<1x4096x1280xbf16>) -> tensor<1x10x4096x128xbf16> {
    // CHECK: error: 'ttir.nlp_create_qkv_heads' op input tensor must be a 4D tensor
    %q, %k, %v = "ttir.nlp_create_qkv_heads"(%arg0) <{num_q_heads = 10 : ui32, num_kv_heads = 0 : ui32}> : (tensor<1x4096x1280xbf16>) -> (tensor<1x10x4096x128xbf16>, tensor<1x0x4096x128xbf16>, tensor<1x0x4096x128xbf16>)
    return %q : tensor<1x10x4096x128xbf16>
  }
}

// -----

module {
  func.func @nlp_create_qkv_heads_bad_seq_dim(%arg0: tensor<1x4096x1x1280xbf16>) -> tensor<1x10x1x128xbf16> {
    // CHECK: error: 'ttir.nlp_create_qkv_heads' op input tensor dimension 1 must be 1
    %q, %k, %v = "ttir.nlp_create_qkv_heads"(%arg0) <{num_q_heads = 10 : ui32, num_kv_heads = 0 : ui32}> : (tensor<1x4096x1x1280xbf16>) -> (tensor<1x10x1x128xbf16>, tensor<1x0x1x128xbf16>, tensor<1x0x1x128xbf16>)
    return %q : tensor<1x10x1x128xbf16>
  }
}
