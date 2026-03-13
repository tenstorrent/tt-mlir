// RUN: ttmlir-opt --ttir-to-ttnn-backend-pipeline="system-desc-path=%system_desc_path%" -o %t.mlir %s
// RUN: ttmlir-translate --ttnn-to-flatbuffer -o %basename_t.ttnn %t.mlir
// RUN: ttmlir-opt --ttnn-to-emitc-device-pipeline -o %t2.mlir %t.mlir
// RUN: ttmlir-translate --mlir-to-cpp -o %basename_t.cpp %t2.mlir

module {
  func.func @forward(%query: tensor<1x1x32x64xbf16>, %key: tensor<1x1x2048x64xbf16>, %page_table: tensor<1x64xui32>) -> tensor<1x1x32x64xbf16> {
    %1 = "ttir.paged_flash_multi_latent_attention_decode"(%query, %key, %page_table) <{head_dim_v = 64 : ui32, is_causal = true, operandSegmentSizes = array<i32: 1, 1, 0, 1, 0, 0, 0>}> : (tensor<1x1x32x64xbf16>, tensor<1x1x2048x64xbf16>, tensor<1x64xui32>) -> tensor<1x1x32x64xbf16>
    return %1 : tensor<1x1x32x64xbf16>
  }
}
