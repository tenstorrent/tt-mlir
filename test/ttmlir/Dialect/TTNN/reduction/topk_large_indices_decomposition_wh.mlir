// REQUIRES: stablehlo
// RUN: ttmlir-opt --stablehlo-to-ttir-pipeline -o %t.mlir %s
// RUN: ttmlir-opt --ttir-to-ttnn-backend-pipeline="mock-system-desc-arch=wormhole_b0" -o %t2.mlir %t.mlir
// RUN: FileCheck %s --input-file=%t2.mlir --implicit-check-not="ttnn.topk_large_indices"

// End-to-end decomposition path: on a non-Blackhole target (Wormhole), the
// ttcore.composite "topk_large_indices" is replaced by its decomposition of
// ttnn primitives (a plain ttnn.topk whose indices are kept).

module @topk_large_indices {
  func.func public @topk_large_indices(%input: tensor<2x64xbf16>) -> tensor<2x16xui32> {
    // CHECK: "ttnn.topk"
    %0 = stablehlo.custom_call @tt.topk_large_indices(%input) {api_version = 0 : i32, mhlo.frontend_attributes = {k = "16"}} : (tensor<2x64xbf16>) -> tensor<2x16xui32>
    return %0 : tensor<2x16xui32>
  }
}
