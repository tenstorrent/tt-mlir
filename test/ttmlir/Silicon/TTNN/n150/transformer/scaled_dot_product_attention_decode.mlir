// RUN: ttmlir-opt --ttir-to-ttnn-backend-pipeline="system-desc-path=%system_desc_path%" %s > %t.mlir
// RUN: FileCheck %s --input-file=%t.mlir
// RUN: ttmlir-translate --ttnn-to-flatbuffer %t.mlir > %t.ttnn

module {
    func.func @qkv_only_sdpa(%query: tensor<1x1x1x32xbf16>, %key: tensor<1x1x32x32xbf16>, %value: tensor<1x1x32x32xbf16>) -> tensor<1x1x1x32xbf16> {
        %output = ttir.empty() : tensor<1x1x1x32xbf16>
        // CHECK: "ttnn.scaled_dot_product_attention_decode"
        %1 = "ttir.scaled_dot_product_attention_decode"(%query, %key, %value, %output) <{ operandSegmentSizes = array<i32: 1, 1, 1, 1, 0, 0, 0>, is_causal = false, scale = 1.0 : f32 }> : (tensor<1x1x1x32xbf16>, tensor<1x1x32x32xbf16>, tensor<1x1x32x32xbf16>, tensor<1x1x1x32xbf16>) -> tensor<1x1x1x32xbf16>
        return %1 : tensor<1x1x1x32xbf16>
    }
}
