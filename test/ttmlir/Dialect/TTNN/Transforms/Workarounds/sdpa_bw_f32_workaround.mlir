// RUN: ttmlir-opt --ttcore-register-device --ttnn-layout --ttnn-workaround --canonicalize -o %t %s
// RUN: FileCheck %s --input-file=%t

// RUN: ttmlir-opt --ttcore-register-device --ttnn-layout --ttnn-workaround="ttnn-optimization-level=1" --canonicalize -o %t1 %s
// RUN: FileCheck %s --check-prefix=OPT1 --input-file=%t1

// Test that at opt-level 0 f32 inputs to ttml sdpa_bw are automatically
// converted to bf16 (and the bf16 grad outputs cast back to f32) by the
// workaround pass, while the f32 log-sum-exp intermediates are left untouched.

module {
  // f32 grad_output/attn_output/Q/K/V, causal, no mask.
  func.func public @sdpa_bw_f32_causal(%grad_output: tensor<1x8x64x64xf32>, %attn_output: tensor<1x8x64x64xf32>, %query: tensor<1x8x64x64xf32>, %key: tensor<1x8x64x64xf32>, %value: tensor<1x8x64x64xf32>, %intermediates: tensor<1x8x64x32xf32>) -> (tensor<1x8x64x64xf32>, tensor<1x8x64x64xf32>, tensor<1x8x64x64xf32>) {
    // CHECK-LABEL: func.func public @sdpa_bw_f32_causal
    // CHECK-DAG: "ttnn.to_tensor_spec"(%arg0)
    // CHECK-DAG: "ttnn.to_tensor_spec"(%arg1)
    // CHECK-DAG: "ttnn.to_tensor_spec"(%arg2)
    // CHECK-DAG: "ttnn.to_tensor_spec"(%arg3)
    // CHECK-DAG: "ttnn.to_tensor_spec"(%arg4)
    // CHECK: %[[GQ:.*]], %[[GK:.*]], %[[GV:.*]] = "ttnn.sdpa_bw"
    // CHECK-SAME: -> (tensor<1x8x64x64xbf16{{.*}}, tensor<1x8x64x64xbf16{{.*}}, tensor<1x8x64x64xbf16
    // CHECK-DAG: "ttnn.to_tensor_spec"(%[[GQ]])
    // CHECK-DAG: "ttnn.to_tensor_spec"(%[[GK]])
    // CHECK-DAG: "ttnn.to_tensor_spec"(%[[GV]])

    // At opt-level 1 the workaround is skipped: no bf16 casts, op stays f32.
    // OPT1-LABEL: func.func public @sdpa_bw_f32_causal
    // OPT1-NOT: ttnn.to_tensor_spec
    // OPT1: "ttnn.sdpa_bw"
    // OPT1-SAME: -> (tensor<1x8x64x64xf32{{.*}}, tensor<1x8x64x64xf32{{.*}}, tensor<1x8x64x64xf32
    %0, %1, %2 = "ttnn.sdpa_bw"(%grad_output, %attn_output, %query, %key, %value, %intermediates) <{mask_type = #ttcore.attention_mask_type<causal>, dropout_probability = 0.000000e+00 : f32}> : (tensor<1x8x64x64xf32>, tensor<1x8x64x64xf32>, tensor<1x8x64x64xf32>, tensor<1x8x64x64xf32>, tensor<1x8x64x64xf32>, tensor<1x8x64x32xf32>) -> (tensor<1x8x64x64xf32>, tensor<1x8x64x64xf32>, tensor<1x8x64x64xf32>)
    return %0, %1, %2 : tensor<1x8x64x64xf32>, tensor<1x8x64x64xf32>, tensor<1x8x64x64xf32>
  }

  // f32 inputs + arbitrary mask.
  func.func public @sdpa_bw_f32_arbitrary_mask(%grad_output: tensor<1x8x64x64xf32>, %attn_output: tensor<1x8x64x64xf32>, %query: tensor<1x8x64x64xf32>, %key: tensor<1x8x64x64xf32>, %value: tensor<1x8x64x64xf32>, %intermediates: tensor<1x8x64x32xf32>, %mask: tensor<1x1x64x64xf32>) -> (tensor<1x8x64x64xf32>, tensor<1x8x64x64xf32>, tensor<1x8x64x64xf32>) {
    // CHECK-LABEL: func.func public @sdpa_bw_f32_arbitrary_mask
    // CHECK-DAG: "ttnn.to_tensor_spec"(%arg0)
    // CHECK-DAG: "ttnn.to_tensor_spec"(%arg1)
    // CHECK-DAG: "ttnn.to_tensor_spec"(%arg2)
    // CHECK-DAG: "ttnn.to_tensor_spec"(%arg3)
    // CHECK-DAG: "ttnn.to_tensor_spec"(%arg4)
    // CHECK-DAG: "ttnn.to_tensor_spec"(%arg6)
    // CHECK: "ttnn.sdpa_bw"
    %0, %1, %2 = "ttnn.sdpa_bw"(%grad_output, %attn_output, %query, %key, %value, %intermediates, %mask) <{mask_type = #ttcore.attention_mask_type<arbitrary>, dropout_probability = 0.000000e+00 : f32}> : (tensor<1x8x64x64xf32>, tensor<1x8x64x64xf32>, tensor<1x8x64x64xf32>, tensor<1x8x64x64xf32>, tensor<1x8x64x64xf32>, tensor<1x8x64x32xf32>, tensor<1x1x64x64xf32>) -> (tensor<1x8x64x64xf32>, tensor<1x8x64x64xf32>, tensor<1x8x64x64xf32>)
    return %0, %1, %2 : tensor<1x8x64x64xf32>, tensor<1x8x64x64xf32>, tensor<1x8x64x64xf32>
  }

  // bf16 inputs should not trigger the workaround.
  func.func public @sdpa_bw_bf16_no_workaround(%grad_output: tensor<1x8x64x64xbf16>, %attn_output: tensor<1x8x64x64xbf16>, %query: tensor<1x8x64x64xbf16>, %key: tensor<1x8x64x64xbf16>, %value: tensor<1x8x64x64xbf16>, %intermediates: tensor<1x8x64x32xf32>) -> (tensor<1x8x64x64xbf16>, tensor<1x8x64x64xbf16>, tensor<1x8x64x64xbf16>) {
    // CHECK-LABEL: func.func public @sdpa_bw_bf16_no_workaround
    // CHECK-NOT: ttnn.to_tensor_spec
    // CHECK: "ttnn.sdpa_bw"
    %0, %1, %2 = "ttnn.sdpa_bw"(%grad_output, %attn_output, %query, %key, %value, %intermediates) <{mask_type = #ttcore.attention_mask_type<causal>, dropout_probability = 0.000000e+00 : f32}> : (tensor<1x8x64x64xbf16>, tensor<1x8x64x64xbf16>, tensor<1x8x64x64xbf16>, tensor<1x8x64x64xbf16>, tensor<1x8x64x64xbf16>, tensor<1x8x64x32xf32>) -> (tensor<1x8x64x64xbf16>, tensor<1x8x64x64xbf16>, tensor<1x8x64x64xbf16>)
    return %0, %1, %2 : tensor<1x8x64x64xbf16>, tensor<1x8x64x64xbf16>, tensor<1x8x64x64xbf16>
  }
}
