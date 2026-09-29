// RUN: ttmlir-opt --ttir-to-emitpy-pipeline="system-desc-path=%system_desc_path% composite-resolution=force-promote" -o %t.mlir %s
// RUN: ttmlir-translate --mlir-to-python %t.mlir | FileCheck %s

module {
  // The Python API always returns a pair. Verify callResultCount=2 forces
  // unpacking even when the MLIR op exposes only the output tensor.
  // CHECK-LABEL: def chunk_gated_delta_rule(
  // CHECK: [[OUTPUT:[a-z_0-9]+]], [[IGNORED:[a-z_0-9]+]] = ttnn.transformer.chunk_gated_delta_rule(
  // CHECK-SAME: output_final_state=False
  // CHECK: return {{\[}}[[OUTPUT]]{{\]}}
  func.func @chunk_gated_delta_rule(
      %q: tensor<1x64x4x32xbf16>, %k: tensor<1x64x4x32xbf16>,
      %v: tensor<1x64x8x32xbf16>, %g: tensor<1x64x8xf32>,
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
        output_head_major = false
      }
    }> : (tensor<1x64x4x32xbf16>, tensor<1x64x4x32xbf16>,
          tensor<1x64x8x32xbf16>, tensor<1x64x8xf32>,
          tensor<1x64x8xf32>) -> tensor<1x64x8x32xbf16>
    return %output : tensor<1x64x8x32xbf16>
  }

  // CHECK-LABEL: def chunk_gated_delta_rule_with_state(
  // CHECK: [[OUTPUT:[a-z_0-9]+]], [[FINAL_STATE:[a-z_0-9]+]] = ttnn.transformer.chunk_gated_delta_rule(
  // CHECK-SAME: output_final_state=True
  // CHECK: return {{\[}}[[OUTPUT]], [[FINAL_STATE]]{{\]}}
  func.func @chunk_gated_delta_rule_with_state(
      %q: tensor<1x64x4x32xbf16>, %k: tensor<1x64x4x32xbf16>,
      %v: tensor<1x64x8x32xbf16>, %g: tensor<1x64x8xf32>,
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
            output_head_major = false
          }
        }> : (tensor<1x64x4x32xbf16>, tensor<1x64x4x32xbf16>,
              tensor<1x64x8x32xbf16>, tensor<1x64x8xf32>,
              tensor<1x64x8xf32>, tensor<1x8x32x32xf32>)
            -> (tensor<1x64x8x32xbf16>, tensor<1x8x32x32xf32>)
    return %output, %final_state
        : tensor<1x64x8x32xbf16>, tensor<1x8x32x32xf32>
  }

  func.func private @chunk_gated_delta_rule_decomp(
      %q: tensor<1x64x4x32xbf16>, %k: tensor<1x64x4x32xbf16>,
      %v: tensor<1x64x8x32xbf16>, %g: tensor<1x64x8xf32>,
      %beta: tensor<1x64x8xf32>) -> tensor<1x64x8x32xbf16> {
    return %v : tensor<1x64x8x32xbf16>
  }

  func.func private @chunk_gated_delta_rule_state_decomp(
      %q: tensor<1x64x4x32xbf16>, %k: tensor<1x64x4x32xbf16>,
      %v: tensor<1x64x8x32xbf16>, %g: tensor<1x64x8xf32>,
      %beta: tensor<1x64x8xf32>,
      %initial_state: tensor<1x8x32x32xf32>)
      -> (tensor<1x64x8x32xbf16>, tensor<1x8x32x32xf32>) {
    return %v, %initial_state
        : tensor<1x64x8x32xbf16>, tensor<1x8x32x32xf32>
  }
}
