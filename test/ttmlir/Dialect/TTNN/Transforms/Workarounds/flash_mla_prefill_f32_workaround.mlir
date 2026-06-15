// RUN: ttmlir-opt --ttcore-register-device --ttnn-layout --ttnn-workaround --canonicalize -o %t %s
// RUN: FileCheck %s --input-file=%t

// Also run at optimization-level 1 (the level used by the optimizer pipeline
// and by composite-promotion OpModel validation). At this level the operand
// workaround only fires for ops in the allow-list, so this run guards that
// flash_mla_prefill is registered there (otherwise the Q/K/V casts would be skipped).
// RUN: ttmlir-opt --ttcore-register-device --ttnn-layout --ttnn-workaround="ttnn-optimization-level=1" --canonicalize -o %t1 %s
// RUN: FileCheck %s --input-file=%t1

// Test that f32 inputs to flash_mla_prefill are automatically converted to
// bf16 (and the bf16 output cast back to f32) by the workaround pass.

module {
  // f32 query + key only (no value, no mask, causal).
  func.func public @flash_mla_prefill_f32_qk(%query: tensor<1x16x32x128xf32>, %key: tensor<1x1x32x128xf32>) -> tensor<1x16x32x64xf32> {
    // CHECK-LABEL: func.func public @flash_mla_prefill_f32_qk
    // CHECK-DAG: %[[Q_BF16:.*]] = "ttnn.to_layout"(%arg0)
    // CHECK-DAG: %[[K_BF16:.*]] = "ttnn.to_layout"(%arg1)
    // CHECK: %[[OUT_BF16:.*]] = "ttnn.flash_mla_prefill"(%{{[0-9]+}}, %{{[0-9]+}})
    // CHECK-SAME: tensor<1x16x32x128xbf16
    // CHECK-SAME: tensor<1x1x32x128xbf16
    // CHECK-SAME: -> tensor<1x16x32x64xbf16
    // CHECK: %{{[0-9]+}} = "ttnn.to_layout"(%[[OUT_BF16]])
    // CHECK-SAME: tensor<1x16x32x64xbf16
    // CHECK-SAME: -> tensor<1x16x32x64xf32
    %0 = "ttnn.flash_mla_prefill"(%query, %key) <{operandSegmentSizes = array<i32: 1, 1, 0, 0>, head_dim_v = 64 : ui32, is_causal = true}> : (tensor<1x16x32x128xf32>, tensor<1x1x32x128xf32>) -> tensor<1x16x32x64xf32>
    return %0 : tensor<1x16x32x64xf32>
  }

  // f32 query + key + value (causal).
  func.func public @flash_mla_prefill_f32_qkv(%query: tensor<1x16x32x128xf32>, %key: tensor<1x1x32x128xf32>, %value: tensor<1x1x32x64xf32>) -> tensor<1x16x32x64xf32> {
    // CHECK-LABEL: func.func public @flash_mla_prefill_f32_qkv
    // CHECK-DAG: "ttnn.to_layout"(%arg0)
    // CHECK-DAG: "ttnn.to_layout"(%arg1)
    // CHECK-DAG: "ttnn.to_layout"(%arg2)
    // CHECK: %[[OUT_BF16:.*]] = "ttnn.flash_mla_prefill"(%{{[0-9]+}}, %{{[0-9]+}}, %{{[0-9]+}})
    // CHECK-SAME: tensor<1x16x32x128xbf16
    // CHECK-SAME: tensor<1x1x32x128xbf16
    // CHECK-SAME: tensor<1x1x32x64xbf16
    // CHECK-SAME: -> tensor<1x16x32x64xbf16
    // CHECK: "ttnn.to_layout"(%[[OUT_BF16]])
    // CHECK-SAME: -> tensor<1x16x32x64xf32
    %0 = "ttnn.flash_mla_prefill"(%query, %key, %value) <{operandSegmentSizes = array<i32: 1, 1, 1, 0>, head_dim_v = 64 : ui32, is_causal = true}> : (tensor<1x16x32x128xf32>, tensor<1x1x32x128xf32>, tensor<1x1x32x64xf32>) -> tensor<1x16x32x64xf32>
    return %0 : tensor<1x16x32x64xf32>
  }

  // f32 query + key + mask (non-causal).
  func.func public @flash_mla_prefill_f32_qk_mask(%query: tensor<1x16x32x128xf32>, %key: tensor<1x1x32x128xf32>, %mask: tensor<1x1x32x32xf32>) -> tensor<1x16x32x64xf32> {
    // CHECK-LABEL: func.func public @flash_mla_prefill_f32_qk_mask
    // CHECK-DAG: "ttnn.to_layout"(%arg0)
    // CHECK-DAG: "ttnn.to_layout"(%arg1)
    // CHECK-DAG: "ttnn.to_layout"(%arg2)
    // CHECK: %[[OUT_BF16:.*]] = "ttnn.flash_mla_prefill"(%{{[0-9]+}}, %{{[0-9]+}}, %{{[0-9]+}})
    // CHECK-SAME: tensor<1x16x32x128xbf16
    // CHECK-SAME: tensor<1x1x32x128xbf16
    // CHECK-SAME: tensor<1x1x32x32xbf16
    // CHECK-SAME: -> tensor<1x16x32x64xbf16
    // CHECK: "ttnn.to_layout"(%[[OUT_BF16]])
    // CHECK-SAME: -> tensor<1x16x32x64xf32
    %0 = "ttnn.flash_mla_prefill"(%query, %key, %mask) <{operandSegmentSizes = array<i32: 1, 1, 0, 1>, head_dim_v = 64 : ui32, is_causal = false}> : (tensor<1x16x32x128xf32>, tensor<1x1x32x128xf32>, tensor<1x1x32x32xf32>) -> tensor<1x16x32x64xf32>
    return %0 : tensor<1x16x32x64xf32>
  }

  // f32 query + key + value + mask (non-causal, all optional operands present).
  func.func public @flash_mla_prefill_f32_all(%query: tensor<2x8x64x128xf32>, %key: tensor<2x1x64x128xf32>, %value: tensor<2x1x64x96xf32>, %mask: tensor<2x1x64x64xf32>) -> tensor<2x8x64x96xf32> {
    // CHECK-LABEL: func.func public @flash_mla_prefill_f32_all
    // CHECK-DAG: "ttnn.to_layout"(%arg0)
    // CHECK-DAG: "ttnn.to_layout"(%arg1)
    // CHECK-DAG: "ttnn.to_layout"(%arg2)
    // CHECK-DAG: "ttnn.to_layout"(%arg3)
    // CHECK: %[[OUT_BF16:.*]] = "ttnn.flash_mla_prefill"(%{{[0-9]+}}, %{{[0-9]+}}, %{{[0-9]+}}, %{{[0-9]+}})
    // CHECK-SAME: -> tensor<2x8x64x96xbf16
    // CHECK: "ttnn.to_layout"(%[[OUT_BF16]])
    // CHECK-SAME: -> tensor<2x8x64x96xf32
    %0 = "ttnn.flash_mla_prefill"(%query, %key, %value, %mask) <{operandSegmentSizes = array<i32: 1, 1, 1, 1>, head_dim_v = 96 : ui32, is_causal = false, scale = 1.250000e-01 : f32}> : (tensor<2x8x64x128xf32>, tensor<2x1x64x128xf32>, tensor<2x1x64x96xf32>, tensor<2x1x64x64xf32>) -> tensor<2x8x64x96xf32>
    return %0 : tensor<2x8x64x96xf32>
  }

  // bf16 inputs should not trigger the workaround (no extra to_layout to bf16).
  func.func public @flash_mla_prefill_bf16_no_workaround(%query: tensor<1x16x32x128xbf16>, %key: tensor<1x1x32x128xbf16>) -> tensor<1x16x32x64xbf16> {
    // CHECK-LABEL: func.func public @flash_mla_prefill_bf16_no_workaround
    // CHECK-NOT: ttnn.to_layout
    // CHECK: "ttnn.flash_mla_prefill"
    %0 = "ttnn.flash_mla_prefill"(%query, %key) <{operandSegmentSizes = array<i32: 1, 1, 0, 0>, head_dim_v = 64 : ui32, is_causal = true}> : (tensor<1x16x32x128xbf16>, tensor<1x1x32x128xbf16>) -> tensor<1x16x32x64xbf16>
    return %0 : tensor<1x16x32x64xbf16>
  }
}
