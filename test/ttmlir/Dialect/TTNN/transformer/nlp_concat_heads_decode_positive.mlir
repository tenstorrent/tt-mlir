// RUN: ttmlir-opt --ttir-to-ttnn-backend-pipeline %s | FileCheck %s
module {
  func.func @nlp_concat_heads_decode_basic_1(%arg0: tensor<1x1x32x128xbf16>) -> tensor<1x1x1x4096xbf16> {
    // CHECK: "ttnn.nlp_concat_heads_decode"(%arg0)
    %0 = "ttnn.nlp_concat_heads_decode"(%arg0) {num_heads = 32 : ui32} : (tensor<1x1x32x128xbf16>) -> tensor<1x1x1x4096xbf16>
    return %0 : tensor<1x1x1x4096xbf16>
  }

  func.func @nlp_concat_heads_decode_batch_2(%arg0: tensor<1x2x32x128xbf16>) -> tensor<1x1x2x4096xbf16> {
    // CHECK: "ttnn.nlp_concat_heads_decode"(%arg0)
    %0 = "ttnn.nlp_concat_heads_decode"(%arg0) {num_heads = 32 : ui32} : (tensor<1x2x32x128xbf16>) -> tensor<1x1x2x4096xbf16>
    return %0 : tensor<1x1x2x4096xbf16>
  }

  func.func @nlp_concat_heads_decode_seq_2(%arg0: tensor<2x1x32x128xbf16>) -> tensor<2x1x1x4096xbf16> {
    // CHECK: "ttnn.nlp_concat_heads_decode"(%arg0)
    %0 = "ttnn.nlp_concat_heads_decode"(%arg0) {num_heads = 32 : ui32} : (tensor<2x1x32x128xbf16>) -> tensor<2x1x1x4096xbf16>
    return %0 : tensor<2x1x1x4096xbf16>
  }

  func.func @nlp_concat_heads_decode_16_heads(%arg0: tensor<1x1x32x128xbf16>) -> tensor<1x1x1x2048xbf16> {
    // CHECK: "ttnn.nlp_concat_heads_decode"(%arg0)
    %0 = "ttnn.nlp_concat_heads_decode"(%arg0) {num_heads = 16 : ui32} : (tensor<1x1x32x128xbf16>) -> tensor<1x1x1x2048xbf16>
    return %0 : tensor<1x1x1x2048xbf16>
  }

  func.func @nlp_concat_heads_decode_8_heads(%arg0: tensor<1x1x32x128xbf16>) -> tensor<1x1x1x1024xbf16> {
    // CHECK: "ttnn.nlp_concat_heads_decode"(%arg0)
    %0 = "ttnn.nlp_concat_heads_decode"(%arg0) {num_heads = 8 : ui32} : (tensor<1x1x32x128xbf16>) -> tensor<1x1x1x1024xbf16>
    return %0 : tensor<1x1x1x1024xbf16>
  }

  func.func @nlp_concat_heads_decode_head_size_64(%arg0: tensor<1x1x32x64xbf16>) -> tensor<1x1x1x2048xbf16> {
    // CHECK: "ttnn.nlp_concat_heads_decode"(%arg0)
    %0 = "ttnn.nlp_concat_heads_decode"(%arg0) {num_heads = 32 : ui32} : (tensor<1x1x32x64xbf16>) -> tensor<1x1x1x2048xbf16>
    return %0 : tensor<1x1x1x2048xbf16>
  }

  func.func @nlp_concat_heads_decode_combined(%arg0: tensor<2x4x32x64xbf16>) -> tensor<2x1x4x768xbf16> {
    // CHECK: "ttnn.nlp_concat_heads_decode"(%arg0)
    %0 = "ttnn.nlp_concat_heads_decode"(%arg0) {num_heads = 12 : ui32} : (tensor<2x4x32x64xbf16>) -> tensor<2x1x4x768xbf16>
    return %0 : tensor<2x1x4x768xbf16>
  }

  func.func @nlp_concat_heads_decode_head_size_256(%arg0: tensor<1x1x32x256xbf16>) -> tensor<1x1x1x8192xbf16> {
    // CHECK: "ttnn.nlp_concat_heads_decode"(%arg0)
    %0 = "ttnn.nlp_concat_heads_decode"(%arg0) {num_heads = 32 : ui32} : (tensor<1x1x32x256xbf16>) -> tensor<1x1x1x8192xbf16>
    return %0 : tensor<1x1x1x8192xbf16>
  }

  func.func @nlp_concat_heads_decode_single_head(%arg0: tensor<1x1x32x128xbf16>) -> tensor<1x1x1x128xbf16> {
    // CHECK: "ttnn.nlp_concat_heads_decode"(%arg0)
    %0 = "ttnn.nlp_concat_heads_decode"(%arg0) {num_heads = 1 : ui32} : (tensor<1x1x32x128xbf16>) -> tensor<1x1x1x128xbf16>
    return %0 : tensor<1x1x1x128xbf16>
  }

  func.func @nlp_concat_heads_decode_max_batch_seq(%arg0: tensor<8x8x32x128xbf16>) -> tensor<8x1x8x4096xbf16> {
    // CHECK: "ttnn.nlp_concat_heads_decode"(%arg0)
    %0 = "ttnn.nlp_concat_heads_decode"(%arg0) {num_heads = 32 : ui32} : (tensor<8x8x32x128xbf16>) -> tensor<8x1x8x4096xbf16>
    return %0 : tensor<8x1x8x4096xbf16>
  }
}
