// RUN: ttmlir-opt --ttir-to-ttnn-backend-pipeline %s | FileCheck %s
module {
  // Test case 1: Basic case with num_heads=32, head_size=128
  // Input: [seq=1, batch=1, num_heads=32, head_size=128]
  // Output: [seq=1, 1, batch=1, hidden=4096] where hidden = 32 * 128
  func.func @nlp_concat_heads_decode_basic_1(%arg0: tensor<1x1x32x128xbf16>) -> tensor<1x1x1x4096xbf16> {
    // CHECK: "ttnn.nlp_concat_heads_decode"(%arg0)
    %0 = "ttnn.nlp_concat_heads_decode"(%arg0) {num_heads = 32 : ui32} : (tensor<1x1x32x128xbf16>) -> tensor<1x1x1x4096xbf16>
    return %0 : tensor<1x1x1x4096xbf16>
  }

  // Test case 2: Different batch size (batch=2)
  // Input: [seq=1, batch=2, num_heads=32, head_size=128]
  // Output: [seq=1, 1, batch=2, hidden=4096]
  func.func @nlp_concat_heads_decode_batch_2(%arg0: tensor<1x2x32x128xbf16>) -> tensor<1x1x2x4096xbf16> {
    // CHECK: "ttnn.nlp_concat_heads_decode"(%arg0)
    %0 = "ttnn.nlp_concat_heads_decode"(%arg0) {num_heads = 32 : ui32} : (tensor<1x2x32x128xbf16>) -> tensor<1x1x2x4096xbf16>
    return %0 : tensor<1x1x2x4096xbf16>
  }

  // Test case 3: Different sequence length (seq=2)
  // Input: [seq=2, batch=1, num_heads=32, head_size=128]
  // Output: [seq=2, 1, batch=1, hidden=4096]
  func.func @nlp_concat_heads_decode_seq_2(%arg0: tensor<2x1x32x128xbf16>) -> tensor<2x1x1x4096xbf16> {
    // CHECK: "ttnn.nlp_concat_heads_decode"(%arg0)
    %0 = "ttnn.nlp_concat_heads_decode"(%arg0) {num_heads = 32 : ui32} : (tensor<2x1x32x128xbf16>) -> tensor<2x1x1x4096xbf16>
    return %0 : tensor<2x1x1x4096xbf16>
  }

  // Test case 4: Smaller num_heads=16 (input still padded to 32)
  // Input: [seq=1, batch=1, num_heads=32 (padded), head_size=128]
  // Output: [seq=1, 1, batch=1, hidden=2048] where hidden = 16 * 128
  func.func @nlp_concat_heads_decode_16_heads(%arg0: tensor<1x1x32x128xbf16>) -> tensor<1x1x1x2048xbf16> {
    // CHECK: "ttnn.nlp_concat_heads_decode"(%arg0)
    %0 = "ttnn.nlp_concat_heads_decode"(%arg0) {num_heads = 16 : ui32} : (tensor<1x1x32x128xbf16>) -> tensor<1x1x1x2048xbf16>
    return %0 : tensor<1x1x1x2048xbf16>
  }

  // Test case 5: Smaller num_heads=8 (input still padded to 32)
  // Input: [seq=1, batch=1, num_heads=32 (padded), head_size=128]
  // Output: [seq=1, 1, batch=1, hidden=1024] where hidden = 8 * 128
  func.func @nlp_concat_heads_decode_8_heads(%arg0: tensor<1x1x32x128xbf16>) -> tensor<1x1x1x1024xbf16> {
    // CHECK: "ttnn.nlp_concat_heads_decode"(%arg0)
    %0 = "ttnn.nlp_concat_heads_decode"(%arg0) {num_heads = 8 : ui32} : (tensor<1x1x32x128xbf16>) -> tensor<1x1x1x1024xbf16>
    return %0 : tensor<1x1x1x1024xbf16>
  }

  // Test case 6: Different head size (head_size=64)
  // Input: [seq=1, batch=1, num_heads=32, head_size=64]
  // Output: [seq=1, 1, batch=1, hidden=2048] where hidden = 32 * 64
  func.func @nlp_concat_heads_decode_head_size_64(%arg0: tensor<1x1x32x64xbf16>) -> tensor<1x1x1x2048xbf16> {
    // CHECK: "ttnn.nlp_concat_heads_decode"(%arg0)
    %0 = "ttnn.nlp_concat_heads_decode"(%arg0) {num_heads = 32 : ui32} : (tensor<1x1x32x64xbf16>) -> tensor<1x1x1x2048xbf16>
    return %0 : tensor<1x1x1x2048xbf16>
  }

  // Test case 7: Combined different parameters
  // Input: [seq=2, batch=4, num_heads=32 (padded), head_size=64]
  // Output: [seq=2, 1, batch=4, hidden=768] where hidden = 12 * 64
  func.func @nlp_concat_heads_decode_combined(%arg0: tensor<2x4x32x64xbf16>) -> tensor<2x1x4x768xbf16> {
    // CHECK: "ttnn.nlp_concat_heads_decode"(%arg0)
    %0 = "ttnn.nlp_concat_heads_decode"(%arg0) {num_heads = 12 : ui32} : (tensor<2x4x32x64xbf16>) -> tensor<2x1x4x768xbf16>
    return %0 : tensor<2x1x4x768xbf16>
  }

  // Test case 8: Larger head size (head_size=256)
  // Input: [seq=1, batch=1, num_heads=32, head_size=256]
  // Output: [seq=1, 1, batch=1, hidden=8192] where hidden = 32 * 256
  func.func @nlp_concat_heads_decode_head_size_256(%arg0: tensor<1x1x32x256xbf16>) -> tensor<1x1x1x8192xbf16> {
    // CHECK: "ttnn.nlp_concat_heads_decode"(%arg0)
    %0 = "ttnn.nlp_concat_heads_decode"(%arg0) {num_heads = 32 : ui32} : (tensor<1x1x32x256xbf16>) -> tensor<1x1x1x8192xbf16>
    return %0 : tensor<1x1x1x8192xbf16>
  }

  // Test case 9: Single head (num_heads=1)
  // Input: [seq=1, batch=1, num_heads=32 (padded), head_size=128]
  // Output: [seq=1, 1, batch=1, hidden=128] where hidden = 1 * 128
  func.func @nlp_concat_heads_decode_single_head(%arg0: tensor<1x1x32x128xbf16>) -> tensor<1x1x1x128xbf16> {
    // CHECK: "ttnn.nlp_concat_heads_decode"(%arg0)
    %0 = "ttnn.nlp_concat_heads_decode"(%arg0) {num_heads = 1 : ui32} : (tensor<1x1x32x128xbf16>) -> tensor<1x1x1x128xbf16>
    return %0 : tensor<1x1x1x128xbf16>
  }

  // Test case 10: Maximum batch and sequence
  // Input: [seq=8, batch=8, num_heads=32, head_size=128]
  // Output: [seq=8, 1, batch=8, hidden=4096] where hidden = 32 * 128
  func.func @nlp_concat_heads_decode_max_batch_seq(%arg0: tensor<8x8x32x128xbf16>) -> tensor<8x1x8x4096xbf16> {
    // CHECK: "ttnn.nlp_concat_heads_decode"(%arg0)
    %0 = "ttnn.nlp_concat_heads_decode"(%arg0) {num_heads = 32 : ui32} : (tensor<8x8x32x128xbf16>) -> tensor<8x1x8x4096xbf16>
    return %0 : tensor<8x1x8x4096xbf16>
  }
}