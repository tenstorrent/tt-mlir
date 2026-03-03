// RUN: ttmlir-opt --convert-ttir-to-linalg %s | FileCheck %s

module {
  // Test basic concatenate_heads operation (Meta Llama example)
  func.func @concatenate_heads_basic(%arg0: tensor<1x24x32x128xbf16>) -> tensor<1x32x3072xbf16> {
    // CHECK-LABEL: func.func @concatenate_heads_basic
    // CHECK: tosa.transpose
    // CHECK-SAME: perms = array<i32: 0, 2, 1, 3>
    // CHECK: tosa.reshape
    // CHECK-NOT: ttir.concatenate_heads
    %0 = "ttir.concatenate_heads"(%arg0) : (tensor<1x24x32x128xbf16>) -> tensor<1x32x3072xbf16>
    return %0 : tensor<1x32x3072xbf16>
  }

  // Test with batch size > 1
  func.func @concatenate_heads_batch2(%arg0: tensor<2x24x32x128xbf16>) -> tensor<2x32x3072xbf16> {
    // CHECK-LABEL: func.func @concatenate_heads_batch2
    // CHECK: tosa.transpose
    // CHECK-SAME: perms = array<i32: 0, 2, 1, 3>
    // CHECK: tosa.reshape
    // CHECK-NOT: ttir.concatenate_heads
    %0 = "ttir.concatenate_heads"(%arg0) : (tensor<2x24x32x128xbf16>) -> tensor<2x32x3072xbf16>
    return %0 : tensor<2x32x3072xbf16>
  }

  // Test BERT base example (12 heads, 256 seq, 64 head_size)
  func.func @concatenate_heads_bert(%arg0: tensor<1x12x256x64xbf16>) -> tensor<1x256x768xbf16> {
    // CHECK-LABEL: func.func @concatenate_heads_bert
    // CHECK: tosa.transpose
    // CHECK-SAME: perms = array<i32: 0, 2, 1, 3>
    // CHECK: tosa.reshape
    // CHECK-NOT: ttir.concatenate_heads
    %0 = "ttir.concatenate_heads"(%arg0) : (tensor<1x12x256x64xbf16>) -> tensor<1x256x768xbf16>
    return %0 : tensor<1x256x768xbf16>
  }

  // Test ViT example (12 heads, 197 seq, 64 head_size)
  func.func @concatenate_heads_vit(%arg0: tensor<1x12x197x64xbf16>) -> tensor<1x197x768xbf16> {
    // CHECK-LABEL: func.func @concatenate_heads_vit
    // CHECK: tosa.transpose
    // CHECK-SAME: perms = array<i32: 0, 2, 1, 3>
    // CHECK: tosa.reshape
    // CHECK-NOT: ttir.concatenate_heads
    %0 = "ttir.concatenate_heads"(%arg0) : (tensor<1x12x197x64xbf16>) -> tensor<1x197x768xbf16>
    return %0 : tensor<1x197x768xbf16>
  }

  // Test with f32 element type
  func.func @concatenate_heads_f32(%arg0: tensor<1x8x64x32xf32>) -> tensor<1x64x256xf32> {
    // CHECK-LABEL: func.func @concatenate_heads_f32
    // CHECK: tosa.transpose
    // CHECK-SAME: perms = array<i32: 0, 2, 1, 3>
    // CHECK: tosa.reshape
    // CHECK-NOT: ttir.concatenate_heads
    %0 = "ttir.concatenate_heads"(%arg0) : (tensor<1x8x64x32xf32>) -> tensor<1x64x256xf32>
    return %0 : tensor<1x64x256xf32>
  }
}
