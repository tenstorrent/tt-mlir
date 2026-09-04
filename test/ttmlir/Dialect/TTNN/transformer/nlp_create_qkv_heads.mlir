// RUN: ttmlir-opt --ttir-to-ttnn-backend-pipeline="enable-ttnn-decomposition-pass=false" %s | FileCheck %s
module {
  // CHECK-LABEL: func.func @nlp_create_qkv_heads_v_only
  // CHECK: "ttnn.nlp_create_qkv_heads"
  // CHECK-SAME: num_kv_heads = 0
  // CHECK-SAME: num_q_heads = 10
  func.func @nlp_create_qkv_heads_v_only(%arg0: tensor<1x1x4096x1280xbf16>) -> tensor<1x10x4096x128xbf16> {
    %q, %k, %v = "ttir.nlp_create_qkv_heads"(%arg0) <{num_q_heads = 10 : ui32, num_kv_heads = 0 : ui32}> : (tensor<1x1x4096x1280xbf16>) -> (tensor<1x10x4096x128xbf16>, tensor<1x0x4096x128xbf16>, tensor<1x0x4096x128xbf16>)
    return %q : tensor<1x10x4096x128xbf16>
  }

  // CHECK-LABEL: func.func @nlp_create_qkv_heads_fused
  // CHECK: "ttnn.nlp_create_qkv_heads"
  func.func @nlp_create_qkv_heads_fused(%arg0: tensor<1x1x32x3840xbf16>) -> (tensor<1x10x32x128xbf16>, tensor<1x10x32x128xbf16>, tensor<1x10x32x128xbf16>) {
    %q, %k, %v = "ttir.nlp_create_qkv_heads"(%arg0) <{num_q_heads = 10 : ui32, num_kv_heads = 10 : ui32}> : (tensor<1x1x32x3840xbf16>) -> (tensor<1x10x32x128xbf16>, tensor<1x10x32x128xbf16>, tensor<1x10x32x128xbf16>)
    return %q, %k, %v : tensor<1x10x32x128xbf16>, tensor<1x10x32x128xbf16>, tensor<1x10x32x128xbf16>
  }
}
