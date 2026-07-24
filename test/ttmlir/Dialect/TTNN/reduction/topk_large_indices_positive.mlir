// RUN: ttmlir-opt --ttir-to-ttnn-backend-pipeline="mock-system-desc-arch=blackhole composite-resolution=force-promote" %s | FileCheck %s

// Resolves a ttcore.composite "topk_large_indices" through the TTNN backend
// pipeline (TTNNResolveComposites) on a Blackhole target and verifies it is
// promoted to the typed ttnn.topk_large_indices op carrying k. The synthesized
// decomposition function is the fallback body and is deleted once the typed
// promotion succeeds.

module {
  func.func @topk_large_indices(%input: tensor<1x512xbf16>) -> tensor<1x512xui32> {
    // CHECK-LABEL: @topk_large_indices
    // CHECK: "ttnn.topk_large_indices"
    // CHECK-SAME: k = 512 : ui32
    // CHECK-NOT: "ttcore.composite"
    %0 = "ttcore.composite"(%input) <{composite_name = "topk_large_indices", decomposition = @decomp, composite_attributes = {k = 512 : ui32}}> : (tensor<1x512xbf16>) -> tensor<1x512xui32>
    return %0 : tensor<1x512xui32>
  }
  func.func private @decomp(%input: tensor<1x512xbf16>) -> tensor<1x512xui32> {
    %values, %indices = "ttir.topk"(%input) <{k = 512 : i32, dim = -1 : i32, largest = true, sorted = true}> : (tensor<1x512xbf16>) -> (tensor<1x512xbf16>, tensor<1x512xui32>)
    return %indices : tensor<1x512xui32>
  }
}
