// RUN: ttmlir-opt --split-input-file --ttcore-register-device --ttnn-layout --ttnn-workaround -o %t %s
// RUN: FileCheck %s --input-file=%t
// At opt-level>=1 the workaround bails out; OpValidationAndFallback owns the
// program_config decision in that path.
// RUN: ttmlir-opt --split-input-file --ttcore-register-device --ttnn-layout --ttnn-workaround="ttnn-optimization-level=1" -o %t.opt1 %s
// RUN: FileCheck %s --check-prefix=OPT1 --input-file=%t.opt1

func.func @test_paged_sdpa_large_head_dim(
    %arg0: tensor<1x1x4x512xbf16>,
    %arg1: tensor<128x4x32x512xbf16>,
    %arg2: tensor<128x4x32x512xbf16>,
    %arg3: tensor<1x64xi32>,
    %arg4: tensor<1xi32>) -> tensor<1x1x4x512xbf16> {
  // CHECK-LABEL: func.func @test_paged_sdpa_large_head_dim
  // CHECK: "ttnn.paged_scaled_dot_product_attention_decode"
  // CHECK-SAME: program_config = #ttnn.sdpa_program_config
  // CHECK-SAME: k_chunk_size = 32
  // OPT1-LABEL: func.func @test_paged_sdpa_large_head_dim
  // OPT1: "ttnn.paged_scaled_dot_product_attention_decode"
  // OPT1-NOT: program_config
  %result = "ttnn.paged_scaled_dot_product_attention_decode"(%arg0, %arg1, %arg2, %arg3, %arg4)
      <{is_causal = true, operandSegmentSizes = array<i32: 1, 1, 1, 1, 0, 1, 0>}>
      : (tensor<1x1x4x512xbf16>, tensor<128x4x32x512xbf16>, tensor<128x4x32x512xbf16>,
         tensor<1x64xi32>, tensor<1xi32>) -> tensor<1x1x4x512xbf16>
  return %result : tensor<1x1x4x512xbf16>
}

// -----

func.func @test_paged_sdpa_small_head_dim(
    %arg0: tensor<1x1x12x128xbf16>,
    %arg1: tensor<128x12x32x128xbf16>,
    %arg2: tensor<128x12x32x128xbf16>,
    %arg3: tensor<1x4xi32>,
    %arg4: tensor<1xi32>) -> tensor<1x1x12x128xbf16> {
  // CHECK-LABEL: func.func @test_paged_sdpa_small_head_dim
  // CHECK: "ttnn.paged_scaled_dot_product_attention_decode"
  // CHECK-NOT: program_config
  %result = "ttnn.paged_scaled_dot_product_attention_decode"(%arg0, %arg1, %arg2, %arg3, %arg4)
      <{is_causal = true, operandSegmentSizes = array<i32: 1, 1, 1, 1, 0, 1, 0>}>
      : (tensor<1x1x12x128xbf16>, tensor<128x12x32x128xbf16>, tensor<128x12x32x128xbf16>,
         tensor<1x4xi32>, tensor<1xi32>) -> tensor<1x1x12x128xbf16>
  return %result : tensor<1x1x12x128xbf16>
}
