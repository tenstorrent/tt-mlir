// TODO(dmilinkovic): re-enable CPU-hoisted const-eval once EmitC support for CPU-hoisted ops lands - issue #6100.
// RUN: ttmlir-opt --ttir-to-ttnn-backend-pipeline="enable-cpu-hoisted-const-eval=false system-desc-path=%system_desc_path%" %s > %t.mlir
// RUN: ttmlir-translate --ttnn-to-flatbuffer %t.mlir > %basename_t.ttnn
// RUN: ttmlir-opt --ttnn-to-emitc-device-pipeline %t.mlir > %t2.mlir
// RUN: ttmlir-translate --mlir-to-cpp %t2.mlir > %basename_t.cpp

module {
    func.func @qkv_causal_sdpa(%query: tensor<1x8x12x32xbf16>, %key: tensor<8x3x32x32xbf16>, %value: tensor<8x3x32x32xbf16>, %cur_pos: tensor<8xi32>) -> tensor<1x8x12x32xbf16> {
        // CHECK: "ttnn.scaled_dot_product_attention_decode"
        %0 = "ttir.scaled_dot_product_attention_decode"(%query, %key, %value, %cur_pos) <{ operandSegmentSizes = array<i32: 1, 1, 1, 0, 1, 0>, is_causal = true, scale = 1.0 : f32 }> : (tensor<1x8x12x32xbf16>, tensor<8x3x32x32xbf16>, tensor<8x3x32x32xbf16>, tensor<8xi32>) -> tensor<1x8x12x32xbf16>
        return %0 : tensor<1x8x12x32xbf16>
    }

    func.func @qkv_attn_mask_sdpa(%query: tensor<1x8x12x32xbf16>, %key: tensor<8x3x32x32xbf16>, %value: tensor<8x3x32x32xbf16>, %cur_pos: tensor<8xi32>, %attn_mask: tensor<8x1x12x32xbf16>) -> tensor<1x8x12x32xbf16> {
        // CHECK: "ttnn.scaled_dot_product_attention_decode"
        %0 = "ttir.scaled_dot_product_attention_decode"(%query, %key, %value, %attn_mask, %cur_pos) <{ operandSegmentSizes = array<i32: 1, 1, 1, 1, 1, 0>, is_causal = false, scale = 1.0 : f32 }> : (tensor<1x8x12x32xbf16>, tensor<8x3x32x32xbf16>, tensor<8x3x32x32xbf16>, tensor<8x1x12x32xbf16>, tensor<8xi32>) -> tensor<1x8x12x32xbf16>
        return %0 : tensor<1x8x12x32xbf16>
    }
}
