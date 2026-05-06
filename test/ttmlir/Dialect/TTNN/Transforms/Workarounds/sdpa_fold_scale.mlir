// RUN: ttmlir-opt --ttcore-register-device --ttnn-workaround -o %t %s
// RUN: FileCheck %s --input-file=%t

#dram = #ttnn.buffer_type<dram>
#l_qkv = #ttnn.ttnn_layout<(d0, d1, d2, d3) -> (d0 * 6 * 256 + d1 * 256 + d2, d3), <1x1>, memref<48x4x!ttcore.tile<32x32, bf16>, #dram>, <interleaved>>
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
}
