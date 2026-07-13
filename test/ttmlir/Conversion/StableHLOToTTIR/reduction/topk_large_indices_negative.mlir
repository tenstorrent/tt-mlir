// REQUIRES: stablehlo
// RUN: not ttmlir-opt --split-input-file --stablehlo-to-ttir-pipeline %s 2>&1 | FileCheck %s

// topk_large_indices expects exactly 1 operand (input). With the wrong operand
// count the custom_call fails to legalize (the k-value and shape constraints
// are enforced later by the ttnn.topk_large_indices verifier; see
// test/ttmlir/Dialect/TTNN/reduction/topk_large_indices_negative.mlir).
module {
  func.func public @topk_large_indices_bad_operand_count(%input: tensor<2x64xbf16>, %extra: tensor<2x64xbf16>) -> tensor<2x16xui32> {
    // CHECK: error: failed to legalize operation 'stablehlo.custom_call'
    %0 = stablehlo.custom_call @tt.topk_large_indices(%input, %extra) {api_version = 0 : i32, mhlo.frontend_attributes = {k = "16"}} : (tensor<2x64xbf16>, tensor<2x64xbf16>) -> tensor<2x16xui32>
    return %0 : tensor<2x16xui32>
  }
}
