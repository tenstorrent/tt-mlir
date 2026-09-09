// RUN: ttmlir-opt --ttcore-register-device --ttnn-layout --ttnn-workaround --canonicalize -o %t %s
// RUN: FileCheck %s --input-file=%t

// RUN: ttmlir-opt --ttcore-register-device --ttnn-layout --ttnn-workaround="ttnn-optimization-level=1" --canonicalize -o %t1 %s
// RUN: FileCheck %s --check-prefix=OPT1 --input-file=%t1

// Test that at opt-level 0 f32 inputs to ttml sdpa_fw are automatically
// converted to bf16 (and the bf16 output cast back to f32) by the workaround
// pass, while the f32 log-sum-exp intermediates are left untouched.

module {
  // f32 Q/K/V, causal, no mask, no intermediates.
  func.func public @sdpa_fw_f32_causal(%query: tensor<1x8x64x64xf32>, %key: tensor<1x8x64x64xf32>, %value: tensor<1x8x64x64xf32>) -> tensor<1x8x64x64xf32> {
    // CHECK-LABEL: func.func public @sdpa_fw_f32_causal
    // CHECK-DAG: %[[Q_BF16:.*]] = "ttnn.to_tensor_spec"(%arg0)
    // CHECK-DAG: %[[K_BF16:.*]] = "ttnn.to_tensor_spec"(%arg1)
    // CHECK-DAG: %[[V_BF16:.*]] = "ttnn.to_tensor_spec"(%arg2)
    // CHECK: %[[OUT_BF16:.*]] = "ttnn.sdpa_fw"(%{{[0-9]+}}, %{{[0-9]+}}, %{{[0-9]+}})
    // CHECK-SAME: tensor<1x8x64x64xbf16
    // CHECK-SAME: -> tensor<1x8x64x64xbf16
    // CHECK: %{{[0-9]+}} = "ttnn.to_tensor_spec"(%[[OUT_BF16]])
    // CHECK-SAME: tensor<1x8x64x64xbf16
    // CHECK-SAME: -> tensor<1x8x64x64xf32

    // At opt-level 1 the workaround is skipped: no bf16 casts, op stays f32.
    // OPT1-LABEL: func.func public @sdpa_fw_f32_causal
    // OPT1-NOT: ttnn.to_tensor_spec
    // OPT1: "ttnn.sdpa_fw"
    // OPT1-SAME: tensor<1x8x64x64xf32
    // OPT1-SAME: -> tensor<1x8x64x64xf32
    %0 = "ttnn.sdpa_fw"(%query, %key, %value) <{mask_type = #ttcore.attention_mask_type<causal>, dropout_probability = 0.000000e+00 : f32, return_intermediates = false}> : (tensor<1x8x64x64xf32>, tensor<1x8x64x64xf32>, tensor<1x8x64x64xf32>) -> tensor<1x8x64x64xf32>
    return %0 : tensor<1x8x64x64xf32>
  }

  // f32 Q/K/V + arbitrary mask.
  func.func public @sdpa_fw_f32_arbitrary_mask(%query: tensor<1x8x64x64xf32>, %key: tensor<1x8x64x64xf32>, %value: tensor<1x8x64x64xf32>, %mask: tensor<1x1x64x64xf32>) -> tensor<1x8x64x64xf32> {
    // CHECK-LABEL: func.func public @sdpa_fw_f32_arbitrary_mask
    // CHECK-DAG: "ttnn.to_tensor_spec"(%arg0)
    // CHECK-DAG: "ttnn.to_tensor_spec"(%arg1)
    // CHECK-DAG: "ttnn.to_tensor_spec"(%arg2)
    // CHECK-DAG: "ttnn.to_tensor_spec"(%arg3)
    // CHECK: %[[OUT_BF16:.*]] = "ttnn.sdpa_fw"(%{{[0-9]+}}, %{{[0-9]+}}, %{{[0-9]+}}, %{{[0-9]+}})
    // CHECK-SAME: tensor<1x1x64x64xbf16
    // CHECK-SAME: -> tensor<1x8x64x64xbf16
    // CHECK: "ttnn.to_tensor_spec"(%[[OUT_BF16]])
    // CHECK-SAME: -> tensor<1x8x64x64xf32
    %0 = "ttnn.sdpa_fw"(%query, %key, %value, %mask) <{mask_type = #ttcore.attention_mask_type<arbitrary>, dropout_probability = 0.000000e+00 : f32, return_intermediates = false}> : (tensor<1x8x64x64xf32>, tensor<1x8x64x64xf32>, tensor<1x8x64x64xf32>, tensor<1x1x64x64xf32>) -> tensor<1x8x64x64xf32>
    return %0 : tensor<1x8x64x64xf32>
  }

  // f32 Q/K/V returning intermediates: output cast to bf16, f32 intermediates untouched.
  func.func public @sdpa_fw_f32_intermediates(%query: tensor<1x8x64x64xf32>, %key: tensor<1x8x64x64xf32>, %value: tensor<1x8x64x64xf32>) -> (tensor<1x8x64x64xf32>, tensor<1x8x64x32xf32>) {
    // CHECK-LABEL: func.func public @sdpa_fw_f32_intermediates
    // CHECK-DAG: "ttnn.to_tensor_spec"(%arg0)
    // CHECK-DAG: "ttnn.to_tensor_spec"(%arg1)
    // CHECK-DAG: "ttnn.to_tensor_spec"(%arg2)
    // CHECK: %[[OUT_BF16:.*]], %[[INTERM:.*]] = "ttnn.sdpa_fw"(%{{[0-9]+}}, %{{[0-9]+}}, %{{[0-9]+}})
    // CHECK-SAME: -> (tensor<1x8x64x64xbf16{{.*}}, tensor<1x8x64x32xf32
    // CHECK: "ttnn.to_tensor_spec"(%[[OUT_BF16]])
    // CHECK-SAME: -> tensor<1x8x64x64xf32
    %0, %1 = "ttnn.sdpa_fw"(%query, %key, %value) <{mask_type = #ttcore.attention_mask_type<causal>, dropout_probability = 0.000000e+00 : f32, return_intermediates = true}> : (tensor<1x8x64x64xf32>, tensor<1x8x64x64xf32>, tensor<1x8x64x64xf32>) -> (tensor<1x8x64x64xf32>, tensor<1x8x64x32xf32>)
    return %0, %1 : tensor<1x8x64x64xf32>, tensor<1x8x64x32xf32>
  }

  // bf16 inputs should not trigger the workaround (no extra to_tensor_spec to bf16).
  func.func public @sdpa_fw_bf16_no_workaround(%query: tensor<1x8x64x64xbf16>, %key: tensor<1x8x64x64xbf16>, %value: tensor<1x8x64x64xbf16>) -> tensor<1x8x64x64xbf16> {
    // CHECK-LABEL: func.func public @sdpa_fw_bf16_no_workaround
    // CHECK-NOT: ttnn.to_tensor_spec
    // CHECK: "ttnn.sdpa_fw"
    %0 = "ttnn.sdpa_fw"(%query, %key, %value) <{mask_type = #ttcore.attention_mask_type<causal>, dropout_probability = 0.000000e+00 : f32, return_intermediates = false}> : (tensor<1x8x64x64xbf16>, tensor<1x8x64x64xbf16>, tensor<1x8x64x64xbf16>) -> tensor<1x8x64x64xbf16>
    return %0 : tensor<1x8x64x64xbf16>
  }
}
