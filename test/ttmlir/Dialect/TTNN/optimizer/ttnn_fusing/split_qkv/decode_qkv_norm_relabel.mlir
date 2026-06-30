// REQUIRES: opmodel
// RUN: ttmlir-opt --ttir-to-ttnn-backend-pipeline="system-desc-path=%system_desc_path% optimization-level=1" %s | FileCheck %s

// Decode (seq=1) head split where a per-head q/k RMSNorm sits between the split
// and the [B,H,1,D]->[1,B,H,D] decode relabel. The relabel must sink up through
// the norm (DecodeRelabelThroughRMSNorm) so all three split outputs reach a
// relabel and NLPCreateQKVHeadsDecodeFusing fires. The norm ends up on [1,B,H,D].
module {
  // CHECK-LABEL: func.func @decode_qkv_norm_relabel
  // CHECK: "ttnn.nlp_create_qkv_heads_decode"
  // CHECK-NOT: "ttnn.split_query_key_value_and_split_heads"
  func.func @decode_qkv_norm_relabel(
      %input: tensor<16x5120xbf16>,
      %wq: tensor<1536x5120xbf16>, %qn: tensor<128xbf16>,
      %wk: tensor<128x5120xbf16>, %kn: tensor<128xbf16>,
      %wv: tensor<128x5120xbf16>)
      -> (tensor<1x16x12x128xbf16>, tensor<1x16x1x128xbf16>, tensor<1x16x1x128xbf16>) {
    // Q: matmul -> [16,12,1,128] -> rms_norm -> relabel
    %q0 = "ttir.matmul"(%input, %wq) <{transpose_a = false, transpose_b = true}> : (tensor<16x5120xbf16>, tensor<1536x5120xbf16>) -> tensor<16x1536xbf16>
    %q1 = "ttir.reshape"(%q0) <{shape = [16 : i32, 12 : i32, 1 : i32, 128 : i32]}> : (tensor<16x1536xbf16>) -> tensor<16x12x1x128xbf16>
    %qn0 = "ttir.rms_norm"(%q1, %qn) <{normalized_shape = array<i64: 128>, epsilon = 9.99999974E-6 : f32, operandSegmentSizes = array<i32: 1, 1, 0>}> : (tensor<16x12x1x128xbf16>, tensor<128xbf16>) -> tensor<16x12x1x128xbf16>
    %qd = "ttir.permute"(%qn0) <{permutation = array<i64: 2, 0, 1, 3>}> : (tensor<16x12x1x128xbf16>) -> tensor<1x16x12x128xbf16>
    // K: matmul -> [16,1,1,128] -> rms_norm -> relabel
    %k0 = "ttir.matmul"(%input, %wk) <{transpose_a = false, transpose_b = true}> : (tensor<16x5120xbf16>, tensor<128x5120xbf16>) -> tensor<16x128xbf16>
    %k1 = "ttir.reshape"(%k0) <{shape = [16 : i32, 1 : i32, 1 : i32, 128 : i32]}> : (tensor<16x128xbf16>) -> tensor<16x1x1x128xbf16>
    %kn0 = "ttir.rms_norm"(%k1, %kn) <{normalized_shape = array<i64: 128>, epsilon = 9.99999974E-6 : f32, operandSegmentSizes = array<i32: 1, 1, 0>}> : (tensor<16x1x1x128xbf16>, tensor<128xbf16>) -> tensor<16x1x1x128xbf16>
    %kd = "ttir.permute"(%kn0) <{permutation = array<i64: 2, 0, 1, 3>}> : (tensor<16x1x1x128xbf16>) -> tensor<1x16x1x128xbf16>
    // V: matmul -> [16,1,1,128] -> relabel (no norm)
    %v0 = "ttir.matmul"(%input, %wv) <{transpose_a = false, transpose_b = true}> : (tensor<16x5120xbf16>, tensor<128x5120xbf16>) -> tensor<16x128xbf16>
    %v1 = "ttir.reshape"(%v0) <{shape = [16 : i32, 1 : i32, 1 : i32, 128 : i32]}> : (tensor<16x128xbf16>) -> tensor<16x1x1x128xbf16>
    %vd = "ttir.permute"(%v1) <{permutation = array<i64: 2, 0, 1, 3>}> : (tensor<16x1x1x128xbf16>) -> tensor<1x16x1x128xbf16>
    return %qd, %kd, %vd : tensor<1x16x12x128xbf16>, tensor<1x16x1x128xbf16>, tensor<1x16x1x128xbf16>
  }
}
