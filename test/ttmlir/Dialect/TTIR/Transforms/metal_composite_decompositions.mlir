// RUN: ttmlir-opt --ttir-decompose-composites %s | FileCheck %s --check-prefix=SDPA
// RUN: ttmlir-opt --ttir-decompose-composites %s | FileCheck %s --check-prefix=RMS
// RUN: ttmlir-opt --ttir-decompose-composites %s | FileCheck %s --check-prefix=LAYERNORM
// RUN: ttmlir-opt --ttir-decompose-composites %s | FileCheck %s --check-prefix=SOFTMAX

// =============================================================================
// SDPA decomposition — is_causal=true (MHA, equal heads)
// =============================================================================

// SDPA-LABEL: func.func @sdpa_causal
// SDPA-NOT: ttir.scaled_dot_product_attention
// SDPA-NOT: ttir.softmax
// SDPA-DAG: "ttir.permute"
// SDPA-DAG: "ttir.matmul"
// SDPA: "ttir.arange"
// SDPA: "ttir.arange"
// SDPA: "ttir.ge"
// SDPA: "ttir.where"
// SDPA: "ttir.add"
// SDPA: "ttir.multiply"
// SDPA: "ttir.max"
// SDPA: "ttir.subtract"
// SDPA: "ttir.exp"
// SDPA: "ttir.sum"
// SDPA: "ttir.div"
// SDPA: "ttir.matmul"
// SDPA: return
func.func @sdpa_causal(%q: tensor<1x8x64x64xbf16>, %k: tensor<1x8x64x64xbf16>, %v: tensor<1x8x64x64xbf16>) -> tensor<1x8x64x64xbf16> {
  %0 = "ttir.scaled_dot_product_attention"(%q, %k, %v) <{is_causal = true, operandSegmentSizes = array<i32: 1, 1, 1, 0, 0>}> : (tensor<1x8x64x64xbf16>, tensor<1x8x64x64xbf16>, tensor<1x8x64x64xbf16>) -> tensor<1x8x64x64xbf16>
  return %0 : tensor<1x8x64x64xbf16>
}

// =============================================================================
// SDPA decomposition — is_causal=false (no causal mask)
// =============================================================================

// SDPA-LABEL: func.func @sdpa_non_causal
// SDPA-NOT: ttir.scaled_dot_product_attention
// SDPA-NOT: ttir.softmax
// SDPA-NOT: ttir.arange
// SDPA-NOT: ttir.ge
// SDPA-NOT: ttir.where
// SDPA: "ttir.permute"
// SDPA: "ttir.matmul"
// SDPA: "ttir.multiply"
// SDPA: "ttir.max"
// SDPA: "ttir.subtract"
// SDPA: "ttir.exp"
// SDPA: "ttir.sum"
// SDPA: "ttir.div"
// SDPA: "ttir.matmul"
// SDPA: return
func.func @sdpa_non_causal(%q: tensor<1x8x64x64xbf16>, %k: tensor<1x8x64x64xbf16>, %v: tensor<1x8x64x64xbf16>) -> tensor<1x8x64x64xbf16> {
  %0 = "ttir.scaled_dot_product_attention"(%q, %k, %v) <{is_causal = false, operandSegmentSizes = array<i32: 1, 1, 1, 0, 0>}> : (tensor<1x8x64x64xbf16>, tensor<1x8x64x64xbf16>, tensor<1x8x64x64xbf16>) -> tensor<1x8x64x64xbf16>
  return %0 : tensor<1x8x64x64xbf16>
}

// =============================================================================
// SDPA decomposition — GQA (3:1 head ratio) with is_causal=true
// =============================================================================

// SDPA-LABEL: func.func @sdpa_gqa_causal
// SDPA-NOT: ttir.scaled_dot_product_attention
// SDPA-NOT: ttir.softmax
// SDPA: "ttir.reshape"
// SDPA: "ttir.permute"
// SDPA: "ttir.matmul"
// SDPA: "ttir.reshape"
// SDPA: "ttir.arange"
// SDPA: "ttir.arange"
// SDPA: "ttir.ge"
// SDPA: "ttir.where"
// SDPA: "ttir.add"
// SDPA: "ttir.max"
// SDPA: "ttir.subtract"
// SDPA: "ttir.exp"
// SDPA: "ttir.sum"
// SDPA: "ttir.div"
// SDPA: "ttir.reshape"
// SDPA: "ttir.matmul"
// SDPA: "ttir.reshape"
// SDPA: return
func.func @sdpa_gqa_causal(%q: tensor<1x24x128x128xbf16>, %k: tensor<1x8x128x128xbf16>, %v: tensor<1x8x128x128xbf16>) -> tensor<1x24x128x128xbf16> {
  %0 = "ttir.scaled_dot_product_attention"(%q, %k, %v) <{is_causal = true, operandSegmentSizes = array<i32: 1, 1, 1, 0, 0>}> : (tensor<1x24x128x128xbf16>, tensor<1x8x128x128xbf16>, tensor<1x8x128x128xbf16>) -> tensor<1x24x128x128xbf16>
  return %0 : tensor<1x24x128x128xbf16>
}

// =============================================================================
// SDPA decomposition — Llama 3.2 3B prefill: GQA 3:1, is_causal=true
// (from sdpa_llama.mlir)
// =============================================================================

// SDPA-LABEL: func.func @sdpa_llama_3b_gqa_causal
// SDPA-NOT: ttir.scaled_dot_product_attention
// SDPA-NOT: ttir.softmax
// SDPA: "ttir.reshape"{{.*}}tensor<1x8x384x128xbf16>
// SDPA: "ttir.permute"
// SDPA: "ttir.matmul"
// SDPA: "ttir.reshape"{{.*}}tensor<1x24x128x128xbf16>
// SDPA: "ttir.arange"
// SDPA: "ttir.ge"
// SDPA: "ttir.where"
// SDPA: "ttir.add"
// SDPA: "ttir.multiply"
// SDPA: "ttir.max"
// SDPA: "ttir.subtract"
// SDPA: "ttir.exp"
// SDPA: "ttir.sum"
// SDPA: "ttir.div"
// SDPA: "ttir.reshape"{{.*}}tensor<1x8x384x128xbf16>
// SDPA: "ttir.matmul"
// SDPA: "ttir.reshape"{{.*}}tensor<1x24x128x128xbf16>
// SDPA: return
func.func @sdpa_llama_3b_gqa_causal(%q: tensor<1x24x128x128xbf16>, %k: tensor<1x8x128x128xbf16>, %v: tensor<1x8x128x128xbf16>) -> tensor<1x24x128x128xbf16> {
  %0 = "ttir.scaled_dot_product_attention"(%q, %k, %v) <{is_causal = true, operandSegmentSizes = array<i32: 1, 1, 1, 0, 0>}> : (tensor<1x24x128x128xbf16>, tensor<1x8x128x128xbf16>, tensor<1x8x128x128xbf16>) -> tensor<1x24x128x128xbf16>
  return %0 : tensor<1x24x128x128xbf16>
}

// =============================================================================
// SDPA decomposition — Qwen GQA 4:1 with mask, non-causal
// (from sdpa.mlir)
// =============================================================================

// SDPA-LABEL: func.func @sdpa_qwen_gqa_mask
// SDPA-NOT: ttir.scaled_dot_product_attention
// SDPA-NOT: ttir.softmax
// SDPA-NOT: ttir.arange
// SDPA-NOT: ttir.ge
// SDPA: "ttir.reshape"{{.*}}tensor<1x8x512x128xbf16>
// SDPA: "ttir.permute"
// SDPA: "ttir.matmul"
// SDPA: "ttir.reshape"{{.*}}tensor<1x32x128x128xbf16>
// SDPA: "ttir.add"
// SDPA: "ttir.multiply"
// SDPA: "ttir.max"
// SDPA: "ttir.subtract"
// SDPA: "ttir.exp"
// SDPA: "ttir.sum"
// SDPA: "ttir.div"
// SDPA: "ttir.reshape"{{.*}}tensor<1x8x512x128xbf16>
// SDPA: "ttir.matmul"
// SDPA: "ttir.reshape"{{.*}}tensor<1x32x128x128xbf16>
// SDPA: return
func.func @sdpa_qwen_gqa_mask(%q: tensor<1x32x128x128xbf16>, %k: tensor<1x8x128x128xbf16>, %v: tensor<1x8x128x128xbf16>, %mask: tensor<1x1x128x128xbf16>) -> tensor<1x32x128x128xbf16> {
  %0 = "ttir.scaled_dot_product_attention"(%q, %k, %v, %mask) <{is_causal = false, operandSegmentSizes = array<i32: 1, 1, 1, 1, 0>}> : (tensor<1x32x128x128xbf16>, tensor<1x8x128x128xbf16>, tensor<1x8x128x128xbf16>, tensor<1x1x128x128xbf16>) -> tensor<1x32x128x128xbf16>
  return %0 : tensor<1x32x128x128xbf16>
}

// =============================================================================
// SDPA decomposition — with custom scale
// =============================================================================

// SDPA-LABEL: func.func @sdpa_custom_scale
// SDPA-NOT: ttir.scaled_dot_product_attention
// SDPA-NOT: ttir.softmax
// SDPA: "ttir.full"() <{fill_value = 2.500000e-01 : f32
// SDPA: "ttir.multiply"
// SDPA: "ttir.max"
// SDPA: "ttir.subtract"
// SDPA: "ttir.exp"
// SDPA: "ttir.sum"
// SDPA: "ttir.div"
func.func @sdpa_custom_scale(%q: tensor<1x8x64x64xbf16>, %k: tensor<1x8x64x64xbf16>, %v: tensor<1x8x64x64xbf16>) -> tensor<1x8x64x64xbf16> {
  %0 = "ttir.scaled_dot_product_attention"(%q, %k, %v) <{is_causal = false, scale = 0.25 : f32, operandSegmentSizes = array<i32: 1, 1, 1, 0, 0>}> : (tensor<1x8x64x64xbf16>, tensor<1x8x64x64xbf16>, tensor<1x8x64x64xbf16>) -> tensor<1x8x64x64xbf16>
  return %0 : tensor<1x8x64x64xbf16>
}

// =============================================================================
// SDPA decomposition — with explicit attention mask
// =============================================================================

// SDPA-LABEL: func.func @sdpa_with_mask
// SDPA-NOT: ttir.scaled_dot_product_attention
// SDPA-NOT: ttir.softmax
// SDPA: "ttir.matmul"
// SDPA: "ttir.add"
// SDPA: "ttir.multiply"
// SDPA: "ttir.max"
// SDPA: "ttir.subtract"
// SDPA: "ttir.exp"
// SDPA: "ttir.sum"
// SDPA: "ttir.div"
func.func @sdpa_with_mask(%q: tensor<1x8x64x64xbf16>, %k: tensor<1x8x64x64xbf16>, %v: tensor<1x8x64x64xbf16>, %mask: tensor<1x1x64x64xbf16>) -> tensor<1x8x64x64xbf16> {
  %0 = "ttir.scaled_dot_product_attention"(%q, %k, %v, %mask) <{is_causal = false, operandSegmentSizes = array<i32: 1, 1, 1, 1, 0>}> : (tensor<1x8x64x64xbf16>, tensor<1x8x64x64xbf16>, tensor<1x8x64x64xbf16>, tensor<1x1x64x64xbf16>) -> tensor<1x8x64x64xbf16>
  return %0 : tensor<1x8x64x64xbf16>
}

// =============================================================================
// RMS norm decomposition — with weight and bias
// =============================================================================

// RMS-LABEL: func.func @rms_norm_weight_bias
// RMS-NOT: ttir.rms_norm
// RMS: "ttir.multiply"
// RMS: "ttir.mean"
// RMS: "ttir.add"
// RMS: "ttir.rsqrt"
// RMS: "ttir.multiply"
// RMS: "ttir.multiply"
// RMS: "ttir.add"
// RMS: return
func.func @rms_norm_weight_bias(%input: tensor<2x4x64xf32>, %weight: tensor<64xf32>, %bias: tensor<64xf32>) -> tensor<2x4x64xf32> {
  %0 = "ttir.rms_norm"(%input, %weight, %bias) <{normalized_shape = array<i64: 64>, epsilon = 1.000000e-05 : f32, operandSegmentSizes = array<i32: 1, 1, 1>}> : (tensor<2x4x64xf32>, tensor<64xf32>, tensor<64xf32>) -> tensor<2x4x64xf32>
  return %0 : tensor<2x4x64xf32>
}

// =============================================================================
// RMS norm decomposition — no weight, no bias
// =============================================================================

// RMS-LABEL: func.func @rms_norm_no_weight_no_bias
// RMS-NOT: ttir.rms_norm
// RMS: "ttir.multiply"
// RMS: "ttir.mean"
// RMS: "ttir.add"
// RMS: "ttir.rsqrt"
// RMS: "ttir.multiply"
// RMS-NOT: "ttir.multiply"
// RMS: return
func.func @rms_norm_no_weight_no_bias(%input: tensor<32x128xf32>) -> tensor<32x128xf32> {
  %0 = "ttir.rms_norm"(%input) <{normalized_shape = array<i64: 128>, epsilon = 1.000000e-06 : f32, operandSegmentSizes = array<i32: 1, 0, 0>}> : (tensor<32x128xf32>) -> tensor<32x128xf32>
  return %0 : tensor<32x128xf32>
}

// =============================================================================
// RMS norm decomposition — weight only (no bias)
// =============================================================================

// RMS-LABEL: func.func @rms_norm_weight_only
// RMS-NOT: ttir.rms_norm
// RMS: "ttir.multiply"
// RMS: "ttir.mean"
// RMS: "ttir.rsqrt"
// RMS: "ttir.multiply"
// RMS: "ttir.multiply"
// RMS-NOT: "ttir.add"
// RMS: return
func.func @rms_norm_weight_only(%input: tensor<2x4x64xf32>, %weight: tensor<64xf32>) -> tensor<2x4x64xf32> {
  %0 = "ttir.rms_norm"(%input, %weight) <{normalized_shape = array<i64: 64>, epsilon = 1.000000e-05 : f32, operandSegmentSizes = array<i32: 1, 1, 0>}> : (tensor<2x4x64xf32>, tensor<64xf32>) -> tensor<2x4x64xf32>
  return %0 : tensor<2x4x64xf32>
}

// =============================================================================
// RMS norm decomposition — custom epsilon
// =============================================================================

// RMS-LABEL: func.func @rms_norm_custom_epsilon
// RMS-NOT: ttir.rms_norm
// RMS: "ttir.full"() <{fill_value = 9.99999997E-7 : f32
// RMS: return
func.func @rms_norm_custom_epsilon(%input: tensor<32x128xf32>) -> tensor<32x128xf32> {
  %0 = "ttir.rms_norm"(%input) <{normalized_shape = array<i64: 128>, epsilon = 1.000000e-06 : f32, operandSegmentSizes = array<i32: 1, 0, 0>}> : (tensor<32x128xf32>) -> tensor<32x128xf32>
  return %0 : tensor<32x128xf32>
}

// =============================================================================
// Layer norm decomposition — with weight and bias
// =============================================================================

// LAYERNORM-LABEL: func.func @layer_norm_weight_bias
// LAYERNORM-NOT: ttir.layer_norm
// LAYERNORM: "ttir.mean"
// LAYERNORM: "ttir.subtract"
// LAYERNORM: "ttir.multiply"
// LAYERNORM: "ttir.mean"
// LAYERNORM: "ttir.add"
// LAYERNORM: "ttir.rsqrt"
// LAYERNORM: "ttir.multiply"
// LAYERNORM: "ttir.multiply"
// LAYERNORM: "ttir.add"
// LAYERNORM: return
func.func @layer_norm_weight_bias(%input: tensor<2x4x64xf32>, %weight: tensor<64xf32>, %bias: tensor<64xf32>) -> tensor<2x4x64xf32> {
  %0 = "ttir.layer_norm"(%input, %weight, %bias) <{normalized_shape = array<i64: 64>, epsilon = 1.000000e-05 : f32, operandSegmentSizes = array<i32: 1, 1, 1>}> : (tensor<2x4x64xf32>, tensor<64xf32>, tensor<64xf32>) -> tensor<2x4x64xf32>
  return %0 : tensor<2x4x64xf32>
}

// =============================================================================
// Layer norm decomposition — no weight, no bias
// =============================================================================

// LAYERNORM-LABEL: func.func @layer_norm_no_weight_no_bias
// LAYERNORM-NOT: ttir.layer_norm
// LAYERNORM: "ttir.mean"
// LAYERNORM: "ttir.subtract"
// LAYERNORM: "ttir.multiply"
// LAYERNORM: "ttir.mean"
// LAYERNORM: "ttir.add"
// LAYERNORM: "ttir.rsqrt"
// LAYERNORM: "ttir.multiply"
// LAYERNORM: return
func.func @layer_norm_no_weight_no_bias(%input: tensor<2x4x64xf32>) -> tensor<2x4x64xf32> {
  %0 = "ttir.layer_norm"(%input) <{normalized_shape = array<i64: 4, 64>, epsilon = 1.000000e-06 : f32, operandSegmentSizes = array<i32: 1, 0, 0>}> : (tensor<2x4x64xf32>) -> tensor<2x4x64xf32>
  return %0 : tensor<2x4x64xf32>
}

// =============================================================================
// Softmax decomposition — numericStable=true (max-subtract path)
// =============================================================================

// SOFTMAX-LABEL: func.func @softmax_stable_last_dim
// SOFTMAX-NOT: ttir.softmax
// SOFTMAX: "ttir.max"{{.*}}dim_arg = [2 : i32]
// SOFTMAX: "ttir.subtract"
// SOFTMAX: "ttir.exp"
// SOFTMAX: "ttir.sum"{{.*}}dim_arg = [2 : i32]
// SOFTMAX: "ttir.div"
// SOFTMAX: return
func.func @softmax_stable_last_dim(%input: tensor<4x32x128xbf16>) -> tensor<4x32x128xbf16> {
  %0 = "ttir.softmax"(%input) <{dimension = 2 : si32, numericStable = true}> : (tensor<4x32x128xbf16>) -> tensor<4x32x128xbf16>
  return %0 : tensor<4x32x128xbf16>
}

// =============================================================================
// Softmax decomposition — numericStable=false, last dimension (dim=2)
// =============================================================================

// SOFTMAX-LABEL: func.func @softmax_last_dim
// SOFTMAX-NOT: ttir.softmax
// SOFTMAX-NOT: "ttir.max"
// SOFTMAX-NOT: "ttir.subtract"
// SOFTMAX: "ttir.exp"
// SOFTMAX: "ttir.sum"{{.*}}dim_arg = [2 : i32]
// SOFTMAX: "ttir.div"
// SOFTMAX: return
func.func @softmax_last_dim(%input: tensor<4x32x128xbf16>) -> tensor<4x32x128xbf16> {
  %0 = "ttir.softmax"(%input) <{dimension = 2 : si32, numericStable = false}> : (tensor<4x32x128xbf16>) -> tensor<4x32x128xbf16>
  return %0 : tensor<4x32x128xbf16>
}

// =============================================================================
// Softmax decomposition — numericStable=false, first dimension (dim=0)
// =============================================================================

// SOFTMAX-LABEL: func.func @softmax_first_dim
// SOFTMAX-NOT: ttir.softmax
// SOFTMAX-NOT: "ttir.max"
// SOFTMAX-NOT: "ttir.subtract"
// SOFTMAX: "ttir.exp"
// SOFTMAX: "ttir.sum"{{.*}}dim_arg = [0 : i32]
// SOFTMAX: "ttir.div"
// SOFTMAX: return
func.func @softmax_first_dim(%input: tensor<4x32x128xbf16>) -> tensor<4x32x128xbf16> {
  %0 = "ttir.softmax"(%input) <{dimension = 0 : si32, numericStable = false}> : (tensor<4x32x128xbf16>) -> tensor<4x32x128xbf16>
  return %0 : tensor<4x32x128xbf16>
}

// =============================================================================
// Softmax decomposition — numericStable=false, middle dimension (dim=1)
// =============================================================================

// SOFTMAX-LABEL: func.func @softmax_mid_dim
// SOFTMAX-NOT: ttir.softmax
// SOFTMAX-NOT: "ttir.max"
// SOFTMAX-NOT: "ttir.subtract"
// SOFTMAX: "ttir.exp"
// SOFTMAX: "ttir.sum"{{.*}}dim_arg = [1 : i32]
// SOFTMAX: "ttir.div"
// SOFTMAX: return
func.func @softmax_mid_dim(%input: tensor<4x32x128xbf16>) -> tensor<4x32x128xbf16> {
  %0 = "ttir.softmax"(%input) <{dimension = 1 : si32, numericStable = false}> : (tensor<4x32x128xbf16>) -> tensor<4x32x128xbf16>
  return %0 : tensor<4x32x128xbf16>
}
