// RUN: ttmlir-opt --ttcore-register-device --ttnn-workaround -o %t %s
// RUN: FileCheck %s --input-file=%t

#dram = #ttnn.buffer_type<dram>
#l_qkv = #ttnn.ttnn_layout<(d0, d1, d2, d3) -> (d0 * 6 * 256 + d1 * 256 + d2, d3), <1x1>, memref<48x4x!ttcore.tile<32x32, bf16>, #dram>, <interleaved>>
#l_qkv_f32 = #ttnn.ttnn_layout<(d0, d1, d2, d3) -> (d0 * 256 * 6 + d1 * 6 + d2, d3), <1x1>, memref<48x4x!ttcore.tile<32x32, f32>, #dram>, <interleaved>>
#l_qkv_f32_permuted = #ttnn.ttnn_layout<(d0, d1, d2, d3) -> (d0 * 6 * 256 + d1 * 256 + d2, d3), <1x1>, memref<48x4x!ttcore.tile<32x32, f32>, #dram>, <interleaved>>
#l_scalar = #ttnn.ttnn_layout<(d0, d1, d2, d3) -> (d0 + d1 + d2, d3), <1x1>, memref<1x1x!ttcore.tile<32x32, f32>, #dram>, <interleaved>>

module attributes {} {
  // CHECK-LABEL: q_side_multiply_only
  func.func @q_side_multiply_only(
      %q: tensor<1x6x256x128xbf16, #l_qkv>,
      %k: tensor<1x6x256x128xbf16, #l_qkv>,
      %v: tensor<1x6x256x128xbf16, #l_qkv>) -> tensor<1x6x256x128xbf16, #l_qkv> {
    %device = "ttnn.get_device"() <{mesh_offset = #ttnn<mesh_offset 0x0>, mesh_shape = #ttnn<mesh_shape 1x1>}> : () -> !ttnn.device
    %scalar = "ttnn.full"(%device) <{dtype = #ttcore.supportedDataTypes<f32>, fill_value = 8.838834e-02 : f32, layout = #ttnn.layout<tile>, shape = #ttnn.shape<1x1x1x1>}> : (!ttnn.device) -> tensor<1x1x1x1xf32, #l_scalar>
    %q_scaled = "ttnn.multiply"(%q, %scalar) <{dtype = #ttcore.supportedDataTypes<bf16>}> : (tensor<1x6x256x128xbf16, #l_qkv>, tensor<1x1x1x1xf32, #l_scalar>) -> tensor<1x6x256x128xbf16, #l_qkv>

    // CHECK-NOT: ttnn.multiply
    // CHECK: %[[OUT:.*]] = "ttnn.scaled_dot_product_attention"(%arg0, %arg1, %arg2)
    // CHECK-SAME: scale = 0.0883883{{[0-9]*}} : f32
    %out = "ttnn.scaled_dot_product_attention"(%q_scaled, %k, %v)
        <{is_causal = false, scale = 1.000000e+00 : f32, operandSegmentSizes = array<i32: 1, 1, 1, 0, 0>}>
        : (tensor<1x6x256x128xbf16, #l_qkv>, tensor<1x6x256x128xbf16, #l_qkv>, tensor<1x6x256x128xbf16, #l_qkv>) -> tensor<1x6x256x128xbf16, #l_qkv>
    return %out : tensor<1x6x256x128xbf16, #l_qkv>
  }

  // CHECK-LABEL: k_side_dit_pattern
  func.func @k_side_dit_pattern(
      %q: tensor<1x6x256x128xbf16, #l_qkv>,
      %k_pre_perm: tensor<1x256x6x128xf32, #l_qkv_f32>,
      %v: tensor<1x6x256x128xbf16, #l_qkv>,
      %device: !ttnn.device) -> tensor<1x6x256x128xbf16, #l_qkv> {
    %scalar = "ttnn.full"(%device) <{fill_value = 8.838834e-02 : f32, shape = #ttnn.shape<1x1x1x1>, dtype = #ttcore.supportedDataTypes<f32>, layout = #ttnn.layout<tile>}> : (!ttnn.device) -> tensor<1x1x1x1xf32, #l_scalar>
    %k_scaled = "ttnn.multiply"(%k_pre_perm, %scalar) <{dtype = #ttcore.supportedDataTypes<f32>}> : (tensor<1x256x6x128xf32, #l_qkv_f32>, tensor<1x1x1x1xf32, #l_scalar>) -> tensor<1x256x6x128xf32, #l_qkv_f32>
    %k_perm = "ttnn.permute"(%k_scaled) <{permutation = array<i64: 0, 2, 1, 3>}> : (tensor<1x256x6x128xf32, #l_qkv_f32>) -> tensor<1x6x256x128xf32, #l_qkv_f32_permuted>
    %k_bf16 = "ttnn.typecast"(%k_perm) <{dtype = #ttcore.supportedDataTypes<bf16>}> : (tensor<1x6x256x128xf32, #l_qkv_f32_permuted>) -> tensor<1x6x256x128xbf16, #l_qkv>

    // CHECK-NOT: ttnn.multiply
    // CHECK: %[[OUT:.*]] = "ttnn.scaled_dot_product_attention"
    // CHECK-SAME: scale = 0.0883883{{[0-9]*}}
    %out = "ttnn.scaled_dot_product_attention"(%q, %k_bf16, %v)
        <{is_causal = false, scale = 1.000000e+00 : f32, operandSegmentSizes = array<i32: 1, 1, 1, 0, 0>}>
        : (tensor<1x6x256x128xbf16, #l_qkv>, tensor<1x6x256x128xbf16, #l_qkv>, tensor<1x6x256x128xbf16, #l_qkv>) -> tensor<1x6x256x128xbf16, #l_qkv>
    return %out : tensor<1x6x256x128xbf16, #l_qkv>
  }

  // CHECK-LABEL: both_sides_combine
  func.func @both_sides_combine(
      %q: tensor<1x6x256x128xbf16, #l_qkv>,
      %k: tensor<1x6x256x128xbf16, #l_qkv>,
      %v: tensor<1x6x256x128xbf16, #l_qkv>,
      %device: !ttnn.device) -> tensor<1x6x256x128xbf16, #l_qkv> {
    %s_q = "ttnn.full"(%device) <{fill_value = 5.000000e-01 : f32, shape = #ttnn.shape<1x1x1x1>, dtype = #ttcore.supportedDataTypes<f32>, layout = #ttnn.layout<tile>}> : (!ttnn.device) -> tensor<1x1x1x1xf32, #l_scalar>
    %s_k = "ttnn.full"(%device) <{fill_value = 5.000000e-01 : f32, shape = #ttnn.shape<1x1x1x1>, dtype = #ttcore.supportedDataTypes<f32>, layout = #ttnn.layout<tile>}> : (!ttnn.device) -> tensor<1x1x1x1xf32, #l_scalar>
    %q_scaled = "ttnn.multiply"(%q, %s_q) <{dtype = #ttcore.supportedDataTypes<bf16>}> : (tensor<1x6x256x128xbf16, #l_qkv>, tensor<1x1x1x1xf32, #l_scalar>) -> tensor<1x6x256x128xbf16, #l_qkv>
    %k_scaled = "ttnn.multiply"(%k, %s_k) <{dtype = #ttcore.supportedDataTypes<bf16>}> : (tensor<1x6x256x128xbf16, #l_qkv>, tensor<1x1x1x1xf32, #l_scalar>) -> tensor<1x6x256x128xbf16, #l_qkv>

    // CHECK-NOT: ttnn.multiply
    // CHECK: %[[OUT:.*]] = "ttnn.scaled_dot_product_attention"
    // CHECK-SAME: scale = 2.500000e-01 : f32
    %out = "ttnn.scaled_dot_product_attention"(%q_scaled, %k_scaled, %v)
        <{is_causal = false, scale = 1.000000e+00 : f32, operandSegmentSizes = array<i32: 1, 1, 1, 0, 0>}>
        : (tensor<1x6x256x128xbf16, #l_qkv>, tensor<1x6x256x128xbf16, #l_qkv>, tensor<1x6x256x128xbf16, #l_qkv>) -> tensor<1x6x256x128xbf16, #l_qkv>
    return %out : tensor<1x6x256x128xbf16, #l_qkv>
  }

  // CHECK-LABEL: existing_scale_combines
  func.func @existing_scale_combines(
      %q: tensor<1x6x256x128xbf16, #l_qkv>,
      %k: tensor<1x6x256x128xbf16, #l_qkv>,
      %v: tensor<1x6x256x128xbf16, #l_qkv>,
      %device: !ttnn.device) -> tensor<1x6x256x128xbf16, #l_qkv> {
    %scalar = "ttnn.full"(%device) <{fill_value = 5.000000e-01 : f32, shape = #ttnn.shape<1x1x1x1>, dtype = #ttcore.supportedDataTypes<f32>, layout = #ttnn.layout<tile>}> : (!ttnn.device) -> tensor<1x1x1x1xf32, #l_scalar>
    %q_scaled = "ttnn.multiply"(%q, %scalar) <{dtype = #ttcore.supportedDataTypes<bf16>}> : (tensor<1x6x256x128xbf16, #l_qkv>, tensor<1x1x1x1xf32, #l_scalar>) -> tensor<1x6x256x128xbf16, #l_qkv>

    // CHECK-NOT: ttnn.multiply
    // CHECK: %[[OUT:.*]] = "ttnn.scaled_dot_product_attention"
    // CHECK-SAME: scale = 2.500000e-01 : f32
    %out = "ttnn.scaled_dot_product_attention"(%q_scaled, %k, %v)
        <{is_causal = false, scale = 5.000000e-01 : f32, operandSegmentSizes = array<i32: 1, 1, 1, 0, 0>}>
        : (tensor<1x6x256x128xbf16, #l_qkv>, tensor<1x6x256x128xbf16, #l_qkv>, tensor<1x6x256x128xbf16, #l_qkv>) -> tensor<1x6x256x128xbf16, #l_qkv>
    return %out : tensor<1x6x256x128xbf16, #l_qkv>
  }

  // CHECK-LABEL: multiply_multi_use
  // The multiply feeds both SDPA and a second function return; the workaround
  // must not fire because bypassing would corrupt the second consumer.
  // CHECK: ttnn.multiply
  // CHECK: scale = 1.000000e+00 : f32
  func.func @multiply_multi_use(
      %q: tensor<1x6x256x128xbf16, #l_qkv>,
      %k: tensor<1x6x256x128xbf16, #l_qkv>,
      %v: tensor<1x6x256x128xbf16, #l_qkv>,
      %device: !ttnn.device)
      -> (tensor<1x6x256x128xbf16, #l_qkv>, tensor<1x6x256x128xbf16, #l_qkv>) {
    %scalar = "ttnn.full"(%device) <{fill_value = 5.000000e-01 : f32, shape = #ttnn.shape<1x1x1x1>, dtype = #ttcore.supportedDataTypes<f32>, layout = #ttnn.layout<tile>}> : (!ttnn.device) -> tensor<1x1x1x1xf32, #l_scalar>
    %q_scaled = "ttnn.multiply"(%q, %scalar) <{dtype = #ttcore.supportedDataTypes<bf16>}> : (tensor<1x6x256x128xbf16, #l_qkv>, tensor<1x1x1x1xf32, #l_scalar>) -> tensor<1x6x256x128xbf16, #l_qkv>
    %out = "ttnn.scaled_dot_product_attention"(%q_scaled, %k, %v)
        <{is_causal = false, scale = 1.000000e+00 : f32, operandSegmentSizes = array<i32: 1, 1, 1, 0, 0>}>
        : (tensor<1x6x256x128xbf16, #l_qkv>, tensor<1x6x256x128xbf16, #l_qkv>, tensor<1x6x256x128xbf16, #l_qkv>) -> tensor<1x6x256x128xbf16, #l_qkv>
    return %out, %q_scaled : tensor<1x6x256x128xbf16, #l_qkv>, tensor<1x6x256x128xbf16, #l_qkv>
  }

  // CHECK-LABEL: multiply_non_constant
  // CHECK: ttnn.multiply
  // CHECK: scale = 1.000000e+00 : f32
  func.func @multiply_non_constant(
      %q: tensor<1x6x256x128xbf16, #l_qkv>,
      %k: tensor<1x6x256x128xbf16, #l_qkv>,
      %v: tensor<1x6x256x128xbf16, #l_qkv>,
      %dyn: tensor<1x1x1x1xf32, #l_scalar>) -> tensor<1x6x256x128xbf16, #l_qkv> {
    %q_scaled = "ttnn.multiply"(%q, %dyn) <{dtype = #ttcore.supportedDataTypes<bf16>}> : (tensor<1x6x256x128xbf16, #l_qkv>, tensor<1x1x1x1xf32, #l_scalar>) -> tensor<1x6x256x128xbf16, #l_qkv>
    %out = "ttnn.scaled_dot_product_attention"(%q_scaled, %k, %v)
        <{is_causal = false, scale = 1.000000e+00 : f32, operandSegmentSizes = array<i32: 1, 1, 1, 0, 0>}>
        : (tensor<1x6x256x128xbf16, #l_qkv>, tensor<1x6x256x128xbf16, #l_qkv>, tensor<1x6x256x128xbf16, #l_qkv>) -> tensor<1x6x256x128xbf16, #l_qkv>
    return %out : tensor<1x6x256x128xbf16, #l_qkv>
  }

  // CHECK-LABEL: no_upstream_multiply
  // CHECK-NOT: ttnn.multiply
  // CHECK: scale = 1.000000e+00 : f32
  func.func @no_upstream_multiply(
      %q: tensor<1x6x256x128xbf16, #l_qkv>,
      %k: tensor<1x6x256x128xbf16, #l_qkv>,
      %v: tensor<1x6x256x128xbf16, #l_qkv>) -> tensor<1x6x256x128xbf16, #l_qkv> {
    %out = "ttnn.scaled_dot_product_attention"(%q, %k, %v)
        <{is_causal = false, scale = 1.000000e+00 : f32, operandSegmentSizes = array<i32: 1, 1, 1, 0, 0>}>
        : (tensor<1x6x256x128xbf16, #l_qkv>, tensor<1x6x256x128xbf16, #l_qkv>, tensor<1x6x256x128xbf16, #l_qkv>) -> tensor<1x6x256x128xbf16, #l_qkv>
    return %out : tensor<1x6x256x128xbf16, #l_qkv>
  }
}
