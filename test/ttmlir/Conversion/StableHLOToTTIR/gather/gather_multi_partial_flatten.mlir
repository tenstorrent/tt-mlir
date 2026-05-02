// REQUIRES: stablehlo
// RUN: ttmlir-opt --stablehlo-to-ttir-pipeline -o %t %s
// RUN: FileCheck %s --input-file=%t

// Gather with two partial-slice indexed dims (slice_sizes = [1, 3, 3, 768]
// on operand [1, 12, 12, 768]). Verifies that the multi-partial-flatten
// pattern fires: operand is flattened to 2-D via the shared permuteInput /
// reshapeInput helpers, the 2-D index vector is flattened and expanded over
// the 3x3 window in a single shot, and a single ttir.embedding is emitted.

// CHECK-LABEL: func.func @gather_multi_partial_flatten
module @test_gather_multi_partial {
  func.func @gather_multi_partial_flatten(%arg0: tensor<1x12x12x768xbf16>, %arg1: tensor<16x12x2xi32>) -> (tensor<1x3x3x768x16x12xbf16>) {
    // Operand flattened via permute + reshape to 2-D weights.
    // CHECK: "ttir.permute"
    // CHECK-SAME: (tensor<1x12x12x768xbf16>) -> tensor<12x12x1x768xbf16>
    // CHECK: "ttir.reshape"
    // CHECK-SAME: (tensor<12x12x1x768xbf16>) -> tensor<144x768xbf16>
    // Multi-component indices combined into a flat 1-D index.
    // CHECK: "ttir.matmul"
    // Window expansion producing a single set of expanded indices.
    // CHECK: "ttir.constant"
    // CHECK: "ttir.broadcast"
    // CHECK: "ttir.add"
    // Exactly one embedding op.
    // CHECK: "ttir.embedding"
    // Final reshape + permute to the original result shape.
    // CHECK: "ttir.reshape"
    // CHECK: "ttir.permute"
    // CHECK: return {{.*}} : tensor<1x3x3x768x16x12xbf16>
    // CHECK-NOT: stablehlo.gather
    %0 = "stablehlo.gather"(%arg0, %arg1) <{dimension_numbers = #stablehlo.gather<offset_dims = [0, 1, 2, 3], start_index_map = [1, 2], index_vector_dim = 2>, indices_are_sorted = false, slice_sizes = array<i64: 1, 3, 3, 768>}> : (tensor<1x12x12x768xbf16>, tensor<16x12x2xi32>) -> tensor<1x3x3x768x16x12xbf16>
    return %0 : tensor<1x3x3x768x16x12xbf16>
  }
}
