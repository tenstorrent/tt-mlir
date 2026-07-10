// REQUIRES: opmodel
// RUN: ttmlir-opt --ttir-to-ttnn-backend-pipeline="system-desc-path=%system_desc_path% optimization-level=1" %s | FileCheck %s

// Decode (seq=1) GQA head split. The [B,H,1,D] -> [1,B,H,D] decode relabel is
// emitted as permute[2,0,1,3]; PermuteOp::canonicalize rewrites it to an
// equivalent reshape (only the S=1 dim moves) before NLPCreateQKVHeadsDecodeFusing
// runs. The decode QKV-heads upgrade must still fire on that reshape form.
module {
  // CHECK-LABEL: func.func @decode_qkv_reshape_relabel
  // CHECK: "ttnn.nlp_create_qkv_heads_decode"
  // CHECK-NOT: "ttnn.split_query_key_value_and_split_heads"
  func.func @decode_qkv_reshape_relabel(
      %input: tensor<16x5120xbf16>,
      %wq: tensor<1536x5120xbf16>,
      %wk: tensor<128x5120xbf16>,
      %wv: tensor<128x5120xbf16>)
      -> (tensor<1x16x12x128xbf16>, tensor<1x16x1x128xbf16>, tensor<1x16x1x128xbf16>) {
    // Q: 12 heads -> [16,12,1,128] (BHSD, S=1) then decode relabel permute.
    %q0 = "ttir.matmul"(%input, %wq) <{transpose_a = false, transpose_b = true}> : (tensor<16x5120xbf16>, tensor<1536x5120xbf16>) -> tensor<16x1536xbf16>
    %q1 = "ttir.reshape"(%q0) <{shape = [16 : i32, 12 : i32, 1 : i32, 128 : i32]}> : (tensor<16x1536xbf16>) -> tensor<16x12x1x128xbf16>
    %qd = "ttir.permute"(%q1) <{permutation = array<i64: 2, 0, 1, 3>}> : (tensor<16x12x1x128xbf16>) -> tensor<1x16x12x128xbf16>
    // K: 1 head
    %k0 = "ttir.matmul"(%input, %wk) <{transpose_a = false, transpose_b = true}> : (tensor<16x5120xbf16>, tensor<128x5120xbf16>) -> tensor<16x128xbf16>
    %k1 = "ttir.reshape"(%k0) <{shape = [16 : i32, 1 : i32, 1 : i32, 128 : i32]}> : (tensor<16x128xbf16>) -> tensor<16x1x1x128xbf16>
    %kd = "ttir.permute"(%k1) <{permutation = array<i64: 2, 0, 1, 3>}> : (tensor<16x1x1x128xbf16>) -> tensor<1x16x1x128xbf16>
    // V: 1 head
    %v0 = "ttir.matmul"(%input, %wv) <{transpose_a = false, transpose_b = true}> : (tensor<16x5120xbf16>, tensor<128x5120xbf16>) -> tensor<16x128xbf16>
    %v1 = "ttir.reshape"(%v0) <{shape = [16 : i32, 1 : i32, 1 : i32, 128 : i32]}> : (tensor<16x128xbf16>) -> tensor<16x1x1x128xbf16>
    %vd = "ttir.permute"(%v1) <{permutation = array<i64: 2, 0, 1, 3>}> : (tensor<16x1x1x128xbf16>) -> tensor<1x16x1x128xbf16>
    return %qd, %kd, %vd : tensor<1x16x12x128xbf16>, tensor<1x16x1x128xbf16>, tensor<1x16x1x128xbf16>
  }
}
