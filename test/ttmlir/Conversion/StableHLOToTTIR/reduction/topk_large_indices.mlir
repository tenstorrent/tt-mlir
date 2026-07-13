// REQUIRES: stablehlo
// RUN: ttmlir-opt --stablehlo-to-ttir-pipeline -o %t %s
// RUN: FileCheck %s --input-file=%t

module @topk_large_indices attributes {} {
  // stablehlo.custom_call @tt.topk_large_indices lowers to a
  // ttcore.composite "topk_large_indices" carrying k, plus a synthesized
  // primitive decomposition function (a ttir.topk keeping only the indices).
  func.func public @topk_large_indices(%input: tensor<2x64xbf16>) -> tensor<2x16xui32> {
    // CHECK-LABEL: @topk_large_indices
    // CHECK: "ttcore.composite"(%arg0)
    // CHECK-SAME: k = 16 : ui32
    // CHECK-SAME: composite_name = "topk_large_indices"
    // CHECK-SAME: decomposition = @topk_large_indices_decomp
    // CHECK-SAME: (tensor<2x64xbf16>) -> tensor<2x16xui32>
    %0 = stablehlo.custom_call @tt.topk_large_indices(%input) {api_version = 0 : i32, mhlo.frontend_attributes = {k = "16"}} : (tensor<2x64xbf16>) -> tensor<2x16xui32>
    return %0 : tensor<2x16xui32>
  }

  // The synthesized decomposition holds the primitive lowering: a single
  // ttir.topk over the last dimension whose (UINT32) indices result is
  // returned and whose values result is discarded.
  // CHECK: func.func private @topk_large_indices_decomp
  // CHECK: "ttir.topk"
  // CHECK-SAME: k = 16 : i32
  // CHECK-SAME: (tensor<2x64xbf16>) -> (tensor<2x16xbf16>, tensor<2x16xui32>)
}
