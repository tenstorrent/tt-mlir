// RUN: ttmlir-opt --split-input-file %s | FileCheck %s

module {
  func.func @nlp_create_qkv_heads_v_only(%arg0: tensor<1x1x4096x1280xbf16>) -> tensor<1x10x4096x128xbf16> {
    // CHECK: "ttir.nlp_create_qkv_heads"
    %q, %k, %v = "ttir.nlp_create_qkv_heads"(%arg0) <{num_q_heads = 10 : ui32, num_kv_heads = 0 : ui32}> : (tensor<1x1x4096x1280xbf16>) -> (tensor<1x10x4096x128xbf16>, tensor<1x0x4096x128xbf16>, tensor<1x0x4096x128xbf16>)
    return %q : tensor<1x10x4096x128xbf16>
  }
}
