// TODO(dmilinkovic): re-enable CPU-hoisted const-eval once EmitC support for CPU-hoisted ops lands - issue #6100.
// RUN: ttmlir-opt --ttir-to-ttnn-common-pipeline="enable-cpu-hoisted-const-eval=false system-desc-path=%system_desc_path%" %s > %t.mlir
// RUN: ttmlir-opt --ttnn-common-to-flatbuffer-pipeline -o %t_fb.mlir %t.mlir
// RUN: ttmlir-translate --ttnn-to-flatbuffer %t_fb.mlir > %basename_t.ttnn
// RUN: ttmlir-opt --ttnn-common-to-emitc-pipeline %t.mlir > %t2.mlir
// RUN: ttmlir-translate --mlir-to-cpp %t2.mlir > %basename_t.cpp

module {
    func.func @qkv_causal_sdpa(%query: tensor<8x12x32x32xbf16>, %key: tensor<8x3x32x32xbf16>, %value: tensor<8x3x32x32xbf16>) -> tensor<8x12x32x32xbf16> {
        // CHECK: "ttnn.scaled_dot_product_attention"
        %0 = "ttir.scaled_dot_product_attention"(%query, %key, %value) <{is_causal = true, scale = 1.0 : f32 }> : (tensor<8x12x32x32xbf16>, tensor<8x3x32x32xbf16>, tensor<8x3x32x32xbf16>) -> tensor<8x12x32x32xbf16>
        return %0 : tensor<8x12x32x32xbf16>
    }

    func.func @qkv_attn_mask_sdpa(%query: tensor<8x12x32x32xbf16>, %key: tensor<8x3x32x32xbf16>, %value: tensor<8x3x32x32xbf16>, %attn_mask: tensor<8x1x32x32xbf16>) -> tensor<8x12x32x32xbf16> {
        // CHECK: "ttnn.scaled_dot_product_attention"
        %0 = "ttir.scaled_dot_product_attention"(%query, %key, %value, %attn_mask) <{is_causal = false, scale = 1.0 : f32 }> : (tensor<8x12x32x32xbf16>, tensor<8x3x32x32xbf16>, tensor<8x3x32x32xbf16>, tensor<8x1x32x32xbf16>) -> tensor<8x12x32x32xbf16>
        return %0 : tensor<8x12x32x32xbf16>
    }
}
