// RUN: ttmlir-opt --ttir-to-ttnn-backend-pipeline -o %t %s
// RUN: FileCheck %s --input-file=%t

module {
  func.func @forward(%query: tensor<1x1x32x64xbf16>, %key: tensor<1x1x2048x64xbf16>, %page_table: tensor<1x64xui32>, %cur_pos: tensor<1xsi32>) -> tensor<1x1x32x64xbf16> {
    // CHECK: "ttnn.paged_flash_multi_latent_attention_decode"
    %1 = "ttir.paged_flash_multi_latent_attention_decode"(%query, %key, %page_table, %cur_pos) <{head_dim_v = 64 : ui32, is_causal = true, operandSegmentSizes = array<i32: 1, 1, 0, 1, 0, 1, 0>}> : (tensor<1x1x32x64xbf16>, tensor<1x1x2048x64xbf16>, tensor<1x64xui32>, tensor<1xsi32>) -> tensor<1x1x32x64xbf16>
    return %1 : tensor<1x1x32x64xbf16>
  }
}
