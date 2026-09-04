// RUN: ttmlir-opt --ttir-to-ttir-decomposition %s | FileCheck %s

module {
  // Rank-3 (H, S, D) operands, causal mask: collapse to 4D and reshape results
  // back to rank 3.
  // CHECK-LABEL: func.func @sdpa_bw_rank3_causal
  func.func @sdpa_bw_rank3_causal(
      %grad_output: tensor<8x64x64xbf16>, %attn_output: tensor<8x64x64xbf16>,
      %query: tensor<8x64x64xbf16>, %key: tensor<8x64x64xbf16>,
      %value: tensor<8x64x64xbf16>, %intermediates: tensor<8x64x32xf32>)
      -> (tensor<8x64x64xbf16>, tensor<8x64x64xbf16>, tensor<8x64x64xbf16>) {
    // CHECK-DAG: "ttir.reshape"(%arg0) <{shape = [1 : i32, 8 : i32, 64 : i32, 64 : i32]}>
    // CHECK-DAG: "ttir.reshape"(%arg5) <{shape = [1 : i32, 8 : i32, 64 : i32, 32 : i32]}>
    // CHECK: "ttir.sdpa_bw"
    // CHECK-SAME: -> (tensor<1x8x64x64xbf16>, tensor<1x8x64x64xbf16>, tensor<1x8x64x64xbf16>)
    // CHECK-DAG: "ttir.reshape"({{.*}}) <{shape = [8 : i32, 64 : i32, 64 : i32]}>
    %0, %1, %2 = "ttir.sdpa_bw"(%grad_output, %attn_output, %query, %key, %value, %intermediates) <{
        mask_type = #ttcore.attention_mask_type<causal>,
        dropout_probability = 0.000000e+00 : f32}>
        : (tensor<8x64x64xbf16>, tensor<8x64x64xbf16>, tensor<8x64x64xbf16>,
           tensor<8x64x64xbf16>, tensor<8x64x64xbf16>, tensor<8x64x32xf32>)
          -> (tensor<8x64x64xbf16>, tensor<8x64x64xbf16>, tensor<8x64x64xbf16>)
    return %0, %1, %2 : tensor<8x64x64xbf16>, tensor<8x64x64xbf16>, tensor<8x64x64xbf16>
  }

  // A rank-4 op is already legal and must be left alone.
  // CHECK-LABEL: func.func @sdpa_bw_rank4
  func.func @sdpa_bw_rank4(
      %grad_output: tensor<1x8x64x64xbf16>, %attn_output: tensor<1x8x64x64xbf16>,
      %query: tensor<1x8x64x64xbf16>, %key: tensor<1x8x64x64xbf16>,
      %value: tensor<1x8x64x64xbf16>, %intermediates: tensor<1x8x64x32xf32>)
      -> (tensor<1x8x64x64xbf16>, tensor<1x8x64x64xbf16>, tensor<1x8x64x64xbf16>) {
    // CHECK-NOT: "ttir.reshape"
    // CHECK: "ttir.sdpa_bw"
    %0, %1, %2 = "ttir.sdpa_bw"(%grad_output, %attn_output, %query, %key, %value, %intermediates) <{
        mask_type = #ttcore.attention_mask_type<causal>,
        dropout_probability = 0.000000e+00 : f32}>
        : (tensor<1x8x64x64xbf16>, tensor<1x8x64x64xbf16>, tensor<1x8x64x64xbf16>,
           tensor<1x8x64x64xbf16>, tensor<1x8x64x64xbf16>, tensor<1x8x64x32xf32>)
          -> (tensor<1x8x64x64xbf16>, tensor<1x8x64x64xbf16>, tensor<1x8x64x64xbf16>)
    return %0, %1, %2 : tensor<1x8x64x64xbf16>, tensor<1x8x64x64xbf16>, tensor<1x8x64x64xbf16>
  }
}
