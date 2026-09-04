// RUN: ttmlir-opt --ttir-to-ttir-decomposition %s | FileCheck %s

module {
  // Rank-3 (H, S, D) operands, causal mask, no intermediates.
  // CHECK-LABEL: func.func @sdpa_fw_rank3_causal
  func.func @sdpa_fw_rank3_causal(%query: tensor<8x64x64xbf16>, %key: tensor<8x64x64xbf16>,
                                  %value: tensor<8x64x64xbf16>) -> tensor<8x64x64xbf16> {
    // CHECK-DAG: "ttir.reshape"(%arg0) <{shape = [1 : i32, 8 : i32, 64 : i32, 64 : i32]}>
    // CHECK-DAG: "ttir.reshape"(%arg1) <{shape = [1 : i32, 8 : i32, 64 : i32, 64 : i32]}>
    // CHECK-DAG: "ttir.reshape"(%arg2) <{shape = [1 : i32, 8 : i32, 64 : i32, 64 : i32]}>
    // CHECK: "ttir.sdpa_fw"
    // CHECK-SAME: -> tensor<1x8x64x64xbf16>
    // CHECK: "ttir.reshape"({{.*}}) <{shape = [8 : i32, 64 : i32, 64 : i32]}>
    %0 = "ttir.sdpa_fw"(%query, %key, %value) <{
        mask_type = #ttcore.attention_mask_type<causal>,
        dropout_probability = 0.000000e+00 : f32,
        return_intermediates = false}>
        : (tensor<8x64x64xbf16>, tensor<8x64x64xbf16>, tensor<8x64x64xbf16>)
          -> tensor<8x64x64xbf16>
    return %0 : tensor<8x64x64xbf16>
  }

  // A rank-4 op is already legal and must be left alone.
  // CHECK-LABEL: func.func @sdpa_fw_rank4
  func.func @sdpa_fw_rank4(%query: tensor<1x8x64x64xbf16>, %key: tensor<1x8x64x64xbf16>,
                           %value: tensor<1x8x64x64xbf16>) -> tensor<1x8x64x64xbf16> {
    // CHECK-NOT: "ttir.reshape"
    // CHECK: "ttir.sdpa_fw"
    %0 = "ttir.sdpa_fw"(%query, %key, %value) <{
        mask_type = #ttcore.attention_mask_type<causal>,
        dropout_probability = 0.000000e+00 : f32,
        return_intermediates = false}>
        : (tensor<1x8x64x64xbf16>, tensor<1x8x64x64xbf16>, tensor<1x8x64x64xbf16>)
          -> tensor<1x8x64x64xbf16>
    return %0 : tensor<1x8x64x64xbf16>
  }

  // Rank-3 operands with an arbitrary mask and log-sum-exp intermediates: both
  // results are reshaped back to their original rank.
  // CHECK-LABEL: func.func @sdpa_fw_rank3_arbitrary_intermediates
  func.func @sdpa_fw_rank3_arbitrary_intermediates(
      %query: tensor<8x64x64xbf16>, %key: tensor<8x64x64xbf16>,
      %value: tensor<8x64x64xbf16>, %mask: tensor<1x1x64x64xbf16>)
      -> (tensor<8x64x64xbf16>, tensor<8x64x32xf32>) {
    // CHECK-DAG: "ttir.reshape"(%arg0) <{shape = [1 : i32, 8 : i32, 64 : i32, 64 : i32]}>
    // CHECK-DAG: "ttir.reshape"(%arg1) <{shape = [1 : i32, 8 : i32, 64 : i32, 64 : i32]}>
    // CHECK-DAG: "ttir.reshape"(%arg2) <{shape = [1 : i32, 8 : i32, 64 : i32, 64 : i32]}>
    // CHECK: "ttir.sdpa_fw"
    // CHECK-SAME: mask_type = #ttcore.attention_mask_type<arbitrary>
    // CHECK-SAME: -> (tensor<1x8x64x64xbf16>, tensor<1x8x64x32xf32>)
    // CHECK-DAG: "ttir.reshape"({{.*}}) <{shape = [8 : i32, 64 : i32, 64 : i32]}>
    // CHECK-DAG: "ttir.reshape"({{.*}}) <{shape = [8 : i32, 64 : i32, 32 : i32]}>
    %0, %1 = "ttir.sdpa_fw"(%query, %key, %value, %mask) <{
        mask_type = #ttcore.attention_mask_type<arbitrary>,
        dropout_probability = 0.000000e+00 : f32,
        return_intermediates = true}>
        : (tensor<8x64x64xbf16>, tensor<8x64x64xbf16>, tensor<8x64x64xbf16>, tensor<1x1x64x64xbf16>)
          -> (tensor<8x64x64xbf16>, tensor<8x64x32xf32>)
    return %0, %1 : tensor<8x64x64xbf16>, tensor<8x64x32xf32>
  }
}
