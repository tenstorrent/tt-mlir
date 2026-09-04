// REQUIRES: opmodel
// RUN: ttmlir-opt --ttir-to-ttnn-backend-pipeline="system-desc-path=%system_desc_path% optimization-level=1 mesh-shape=1,2" %s | FileCheck %s

// Wan 14B Graph A V heads: reshape + permute into SDPA value, not Q/K.
module {
  // CHECK-LABEL: func.func @wan_v_heads_prefill
  // CHECK: "ttnn.nlp_create_qkv_heads"
  // CHECK-SAME: num_kv_heads = 0
  // CHECK-SAME: num_q_heads = 10
  // CHECK: "ttnn.scaled_dot_product_attention"
  func.func @wan_v_heads_prefill(
      %v: tensor<1x4096x1280xbf16>,
      %q: tensor<1x10x4096x128xbf16>,
      %k: tensor<1x10x4096x128xbf16>) -> tensor<1x10x4096x128xbf16> {
    %v0 = "ttir.reshape"(%v) <{shape = [1 : i32, 4096 : i32, 10 : i32, 128 : i32]}> : (tensor<1x4096x1280xbf16>) -> tensor<1x4096x10x128xbf16>
    %v1 = "ttir.permute"(%v0) <{permutation = array<i64: 0, 2, 1, 3>}> : (tensor<1x4096x10x128xbf16>) -> tensor<1x10x4096x128xbf16>
    %out = "ttir.scaled_dot_product_attention"(%q, %k, %v1) <{is_causal = true, operandSegmentSizes = array<i32: 1, 1, 1, 0, 0>}> : (tensor<1x10x4096x128xbf16>, tensor<1x10x4096x128xbf16>, tensor<1x10x4096x128xbf16>) -> tensor<1x10x4096x128xbf16>
    return %out : tensor<1x10x4096x128xbf16>
  }

  // CHECK-LABEL: func.func @wan_v_heads_prefill_sp
  // CHECK: "ttnn.nlp_create_qkv_heads"
  // CHECK-SAME: num_kv_heads = 0
  // CHECK: "ttnn.all_gather"
  // CHECK-SAME: all_gather_dim = 2
  // CHECK: "ttnn.slice_static"
  // CHECK: "ttnn.scaled_dot_product_attention"
  func.func @wan_v_heads_prefill_sp(
      %v: tensor<32x64xbf16>,
      %q: tensor<1x2x32x32xbf16>,
      %k: tensor<1x2x60x32xbf16>) -> tensor<1x2x32x32xbf16> {
    %v0 = "ttir.reshape"(%v) <{shape = [1 : i32, 32 : i32, 2 : i32, 32 : i32]}> : (tensor<32x64xbf16>) -> tensor<1x32x2x32xbf16>
    %v1 = "ttir.all_gather"(%v0) <{all_gather_dim = 1 : si32, cluster_axis = 0 : ui32}> : (tensor<1x32x2x32xbf16>) -> tensor<1x64x2x32xbf16>
    %v2 = "ttir.slice_static"(%v1) <{begins = [0 : i32, 0 : i32, 0 : i32, 0 : i32], ends = [1 : i32, 60 : i32, 2 : i32, 32 : i32], step = [1 : i32, 1 : i32, 1 : i32, 1 : i32]}> : (tensor<1x64x2x32xbf16>) -> tensor<1x60x2x32xbf16>
    %v3 = "ttir.permute"(%v2) <{permutation = array<i64: 0, 2, 1, 3>}> : (tensor<1x60x2x32xbf16>) -> tensor<1x2x60x32xbf16>
    %out = "ttir.scaled_dot_product_attention"(%q, %k, %v3) <{is_causal = false, operandSegmentSizes = array<i32: 1, 1, 1, 0, 0>}> : (tensor<1x2x32x32xbf16>, tensor<1x2x60x32xbf16>, tensor<1x2x60x32xbf16>) -> tensor<1x2x32x32xbf16>
    return %out : tensor<1x2x32x32xbf16>
  }

  // CHECK-LABEL: func.func @no_fuse_query_heads
  // CHECK-NOT: "ttnn.nlp_create_qkv_heads"
  func.func @no_fuse_query_heads(
      %q_in: tensor<1x4096x1280xbf16>,
      %k: tensor<1x10x4096x128xbf16>,
      %v: tensor<1x10x4096x128xbf16>) -> tensor<1x10x4096x128xbf16> {
    %q0 = "ttir.reshape"(%q_in) <{shape = [1 : i32, 4096 : i32, 10 : i32, 128 : i32]}> : (tensor<1x4096x1280xbf16>) -> tensor<1x4096x10x128xbf16>
    %q1 = "ttir.permute"(%q0) <{permutation = array<i64: 0, 2, 1, 3>}> : (tensor<1x4096x10x128xbf16>) -> tensor<1x10x4096x128xbf16>
    %out = "ttir.scaled_dot_product_attention"(%q1, %k, %v) <{is_causal = true, operandSegmentSizes = array<i32: 1, 1, 1, 0, 0>}> : (tensor<1x10x4096x128xbf16>, tensor<1x10x4096x128xbf16>, tensor<1x10x4096x128xbf16>) -> tensor<1x10x4096x128xbf16>
    return %out : tensor<1x10x4096x128xbf16>
  }
}
