// RUN: ttmlir-opt --ttir-to-ttnn-runtime-pipeline="composite-resolution=force-promote" -o %t %s
// RUN: FileCheck %s --input-file=%t
// RUN: ttmlir-translate --ttnn-to-flatbuffer -o %t.ttnn %t

module {
  // CHECK-LABEL: func.func @chunk_gated_delta_rule
  // CHECK: "ttnn.chunk_gated_delta_rule"
  func.func @chunk_gated_delta_rule(
      %q: tensor<1x64x4x32xbf16>,
      %k: tensor<1x64x4x32xbf16>,
      %v: tensor<1x64x8x32xbf16>,
      %g: tensor<1x64x8xf32>,
      %beta: tensor<1x64x8xf32>) -> tensor<1x64x8x32xbf16> {
    %output = "ttcore.composite"(%q, %k, %v, %g, %beta) <{
        composite_name = "chunk_gated_delta_rule",
        decomposition = @chunk_gated_delta_rule_decomp,
        composite_attributes = {
          has_initial_state = false,
          has_eye = false,
          has_tril = false,
          has_ones = false,
          has_masks = false,
          output_final_state = false,
          chunk_size = 64 : ui32,
          use_qk_l2norm = false,
          output_head_major = false}}>
        : (tensor<1x64x4x32xbf16>, tensor<1x64x4x32xbf16>,
           tensor<1x64x8x32xbf16>, tensor<1x64x8xf32>,
           tensor<1x64x8xf32>) -> tensor<1x64x8x32xbf16>
    return %output : tensor<1x64x8x32xbf16>
  }

  // CHECK-LABEL: func.func @chunk_gated_delta_rule_with_state
  // CHECK: "ttnn.chunk_gated_delta_rule"
  func.func @chunk_gated_delta_rule_with_state(
      %q: tensor<1x64x4x32xbf16>,
      %k: tensor<1x64x4x32xbf16>,
      %v: tensor<1x64x8x32xbf16>,
      %g: tensor<1x64x8xf32>,
      %beta: tensor<1x64x8xf32>,
      %initial_state: tensor<1x8x32x32xf32>)
      -> (tensor<1x64x8x32xbf16>, tensor<1x8x32x32xf32>) {
    %output, %final_state =
        "ttcore.composite"(%q, %k, %v, %g, %beta, %initial_state) <{
          composite_name = "chunk_gated_delta_rule",
          decomposition = @chunk_gated_delta_rule_state_decomp,
          composite_attributes = {
            has_initial_state = true,
            has_eye = false,
            has_tril = false,
            has_ones = false,
            has_masks = false,
            output_final_state = true,
            chunk_size = 64 : ui32,
            use_qk_l2norm = false,
            output_head_major = false}}>
          : (tensor<1x64x4x32xbf16>, tensor<1x64x4x32xbf16>,
             tensor<1x64x8x32xbf16>, tensor<1x64x8xf32>,
             tensor<1x64x8xf32>, tensor<1x8x32x32xf32>)
            -> (tensor<1x64x8x32xbf16>, tensor<1x8x32x32xf32>)
    return %output, %final_state
        : tensor<1x64x8x32xbf16>, tensor<1x8x32x32xf32>
  }

  func.func private @chunk_gated_delta_rule_decomp(
      %q: tensor<1x64x4x32xbf16>,
      %k: tensor<1x64x4x32xbf16>,
      %v: tensor<1x64x8x32xbf16>,
      %g: tensor<1x64x8xf32>,
      %beta: tensor<1x64x8xf32>) -> tensor<1x64x8x32xbf16> {
    return %v : tensor<1x64x8x32xbf16>
  }

  func.func private @chunk_gated_delta_rule_state_decomp(
      %q: tensor<1x64x4x32xbf16>,
      %k: tensor<1x64x4x32xbf16>,
      %v: tensor<1x64x8x32xbf16>,
      %g: tensor<1x64x8xf32>,
      %beta: tensor<1x64x8xf32>,
      %initial_state: tensor<1x8x32x32xf32>)
      -> (tensor<1x64x8x32xbf16>, tensor<1x8x32x32xf32>) {
    return %v, %initial_state
        : tensor<1x64x8x32xbf16>, tensor<1x8x32x32xf32>
  }
}
