// RUN: ttmlir-opt --ttcore-register-device --ttnn-layout --ttnn-workaround -o %t %s
// RUN: FileCheck %s --input-file=%t

module {
  // Large num_heads * head_dim would cause tt-metal's nlp_concat_heads kernel
  // to reserve per-core circular buffers (~2 MB) that exceed L1, so the
  // workaround must decompose the op into permute + reshape even when
  // head_size is tile aligned. Shape matches the Gemma-4 31B prefill that
  // triggered the OOM.
  func.func @test_concatenate_heads_l1_overflow(%arg0: tensor<1x32x12x512xbf16>) -> tensor<1x12x16384xbf16> {
    // CHECK-LABEL: func.func @test_concatenate_heads_l1_overflow
    // CHECK: "ttnn.permute"
    // CHECK: "ttnn.reshape"
    // CHECK-NOT: "ttnn.concatenate_heads"
    %result = "ttnn.concatenate_heads"(%arg0) : (tensor<1x32x12x512xbf16>) -> tensor<1x12x16384xbf16>
    return %result : tensor<1x12x16384xbf16>
  }

  // Small num_heads * head_dim and tile-aligned head_size: neither trigger
  // applies, so the op must remain a fused concatenate_heads.
  func.func @test_concatenate_heads_no_rewrite(%arg0: tensor<1x8x12x128xbf16>) -> tensor<1x12x1024xbf16> {
    // CHECK-LABEL: func.func @test_concatenate_heads_no_rewrite
    // CHECK: "ttnn.concatenate_heads"
    // CHECK-NOT: "ttnn.permute"
    %result = "ttnn.concatenate_heads"(%arg0) : (tensor<1x8x12x128xbf16>) -> tensor<1x12x1024xbf16>
    return %result : tensor<1x12x1024xbf16>
  }
}
