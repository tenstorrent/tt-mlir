// RUN: ttmlir-opt --ttir-to-ttnn-backend-pipeline="system-desc-path=%system_desc_path%" %s > %t.mlir
// RUN: FileCheck %s --input-file=%t.mlir
// RUN: ttmlir-translate --ttnn-to-flatbuffer %t.mlir > %t.ttnn

module {
    func.func @qkv_causal_sdpa(%query: tensor<8x12x32x32xbf16>, %key: tensor<8x3x32x32xbf16>, %value: tensor<8x3x32x32xbf16>) -> tensor<8x12x32x32xbf16> {
        %output = ttir.empty() : tensor<8x12x32x32xbf16>
        // CHECK: "ttnn.scaled_dot_product_attention"
        %1 = "ttir.scaled_dot_product_attention"(%query, %key, %value, %output) <{is_causal = true, scale = 1.0 : f32 }> : (tensor<8x12x32x32xbf16>, tensor<8x3x32x32xbf16>, tensor<8x3x32x32xbf16>, tensor<8x12x32x32xbf16>) -> tensor<8x12x32x32xbf16>
        return %1 : tensor<8x12x32x32xbf16>
    }

    func.func @qkv_attn_mask_sdpa(%query: tensor<8x12x32x32xbf16>, %key: tensor<8x3x32x32xbf16>, %value: tensor<8x3x32x32xbf16>, %attn_mask: tensor<8x1x32x32xbf16>) -> tensor<8x12x32x32xbf16> {
        %output = ttir.empty() : tensor<8x12x32x32xbf16>
        // CHECK: "ttnn.scaled_dot_product_attention"
        %1 = "ttir.scaled_dot_product_attention"(%query, %key, %value, %attn_mask, %output) <{is_causal = false, scale = 1.0 : f32 }> : (tensor<8x12x32x32xbf16>, tensor<8x3x32x32xbf16>, tensor<8x3x32x32xbf16>, tensor<8x1x32x32xbf16>, tensor<8x12x32x32xbf16>) -> tensor<8x12x32x32xbf16>
        return %1 : tensor<8x12x32x32xbf16>
    }
}
