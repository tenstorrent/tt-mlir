// REQUIRES: stablehlo
// RUN: ttmlir-opt --stablehlo-to-ttir-pipeline -o %t %s
// RUN: FileCheck %s --input-file=%t

// CHECK-LABEL: func.func @repeat_front
module @test_gather_repeat_front {
  func.func @repeat_front(%arg0: tensor<1x768x4x60x106xbf16>) -> (tensor<1x768x6x60x106xbf16>) {
    // Front pad must slice the *first* element along the indexed dim ([0,1)).
    // CHECK: "ttir.slice_static"
    // CHECK-SAME: begins = [0 : i32, 0 : i32, 0 : i32, 0 : i32, 0 : i32]
    // CHECK-SAME: ends = [1 : i32, 768 : i32, 1 : i32, 60 : i32, 106 : i32]
    // CHECK-SAME: (tensor<1x768x4x60x106xbf16>) -> tensor<1x768x1x60x106xbf16>
    // CHECK: "ttir.repeat"
    // CHECK-SAME: repeat_dimensions = array<i64: 1, 1, 2, 1, 1>
    // CHECK-SAME: (tensor<1x768x1x60x106xbf16>) -> tensor<1x768x2x60x106xbf16>
    // CHECK: "ttir.concat"
    // CHECK-SAME: dim = 2
    // CHECK-SAME: (tensor<1x768x2x60x106xbf16>, tensor<1x768x4x60x106xbf16>) -> tensor<1x768x6x60x106xbf16>
    // CHECK-NOT: stablehlo.gather
    %1 = "stablehlo.constant"() <{value = dense<[[0], [0], [0], [1], [2], [3]]> : tensor<6x1xi64>}> : () -> tensor<6x1xi64>
    %2 = "stablehlo.gather"(%arg0, %1) <{dimension_numbers = #stablehlo.gather<offset_dims = [0, 1, 3, 4], collapsed_slice_dims = [2], start_index_map = [2], index_vector_dim = 1>, indices_are_sorted = false, slice_sizes = array<i64: 1, 768, 1, 60, 106>}> : (tensor<1x768x4x60x106xbf16>, tensor<6x1xi64>) -> tensor<1x768x6x60x106xbf16>
    return %2 : tensor<1x768x6x60x106xbf16>
  }
}

// CHECK-LABEL: func.func @repeat_back
module @test_gather_repeat_back {
  func.func @repeat_back(%arg0: tensor<1x768x4x60x106xbf16>) -> (tensor<1x768x6x60x106xbf16>) {
    // Back pad must slice the *last* element along the indexed dim ([3,4)).
    // CHECK: "ttir.slice_static"
    // CHECK-SAME: begins = [0 : i32, 0 : i32, 3 : i32, 0 : i32, 0 : i32]
    // CHECK-SAME: ends = [1 : i32, 768 : i32, 4 : i32, 60 : i32, 106 : i32]
    // CHECK-SAME: (tensor<1x768x4x60x106xbf16>) -> tensor<1x768x1x60x106xbf16>
    // CHECK: "ttir.repeat"
    // CHECK-SAME: repeat_dimensions = array<i64: 1, 1, 2, 1, 1>
    // CHECK-SAME: (tensor<1x768x1x60x106xbf16>) -> tensor<1x768x2x60x106xbf16>
    // CHECK: "ttir.concat"
    // CHECK-SAME: dim = 2
    // CHECK-SAME: (tensor<1x768x4x60x106xbf16>, tensor<1x768x2x60x106xbf16>) -> tensor<1x768x6x60x106xbf16>
    // CHECK-NOT: stablehlo.gather
    %1 = "stablehlo.constant"() <{value = dense<[[0], [1], [2], [3], [3], [3]]> : tensor<6x1xi64>}> : () -> tensor<6x1xi64>
    %2 = "stablehlo.gather"(%arg0, %1) <{dimension_numbers = #stablehlo.gather<offset_dims = [0, 1, 3, 4], collapsed_slice_dims = [2], start_index_map = [2], index_vector_dim = 1>, indices_are_sorted = false, slice_sizes = array<i64: 1, 768, 1, 60, 106>}> : (tensor<1x768x4x60x106xbf16>, tensor<6x1xi64>) -> tensor<1x768x6x60x106xbf16>
    return %2 : tensor<1x768x6x60x106xbf16>
  }
}

// CHECK-LABEL: func.func @repeat_both
module @test_gather_slice_no_repeat_both {
  func.func @repeat_both(%arg0: tensor<1x768x4x60x106xbf16>) -> (tensor<1x768x6x60x106xbf16>) {
    // Front pad: slice [0,1) along indexed dim.
    // CHECK: "ttir.slice_static"
    // CHECK-SAME: begins = [0 : i32, 0 : i32, 0 : i32, 0 : i32, 0 : i32]
    // CHECK-SAME: ends = [1 : i32, 768 : i32, 1 : i32, 60 : i32, 106 : i32]
    // CHECK-SAME: (tensor<1x768x4x60x106xbf16>) -> tensor<1x768x1x60x106xbf16>
    // CHECK: "ttir.repeat"
    // CHECK-SAME: repeat_dimensions = array<i64: 1, 1, 1, 1, 1>
    // Back pad: slice [3,4) along indexed dim.
    // CHECK: "ttir.slice_static"
    // CHECK-SAME: begins = [0 : i32, 0 : i32, 3 : i32, 0 : i32, 0 : i32]
    // CHECK-SAME: ends = [1 : i32, 768 : i32, 4 : i32, 60 : i32, 106 : i32]
    // CHECK-SAME: (tensor<1x768x4x60x106xbf16>) -> tensor<1x768x1x60x106xbf16>
    // CHECK: "ttir.repeat"
    // CHECK-SAME: repeat_dimensions = array<i64: 1, 1, 1, 1, 1>
    // CHECK: "ttir.concat"
    // CHECK-SAME: dim = 2
    // CHECK-SAME: (tensor<1x768x1x60x106xbf16>, tensor<1x768x4x60x106xbf16>, tensor<1x768x1x60x106xbf16>) -> tensor<1x768x6x60x106xbf16>
    // CHECK-NOT: stablehlo.gather
    %1 = "stablehlo.constant"() <{value = dense<[[0], [0], [1], [2], [3], [3]]> : tensor<6x1xi64>}> : () -> tensor<6x1xi64>
    %2 = "stablehlo.gather"(%arg0, %1) <{dimension_numbers = #stablehlo.gather<offset_dims = [0, 1, 3, 4], collapsed_slice_dims = [2], start_index_map = [2], index_vector_dim = 1>, indices_are_sorted = false, slice_sizes = array<i64: 1, 768, 1, 60, 106>}> : (tensor<1x768x4x60x106xbf16>, tensor<6x1xi64>) -> tensor<1x768x6x60x106xbf16>
    return %2 : tensor<1x768x6x60x106xbf16>
  }
}


// CHECK-LABEL: func.func @repeat_both_multiple
module @test_gather_repeat_both_multiple {
  func.func @repeat_both_multiple(%arg0: tensor<1x768x4x60x106xbf16>) -> (tensor<1x768x8x60x106xbf16>) {
    // Front pad: slice [0,1) along indexed dim, repeated 2x.
    // CHECK: "ttir.slice_static"
    // CHECK-SAME: begins = [0 : i32, 0 : i32, 0 : i32, 0 : i32, 0 : i32]
    // CHECK-SAME: ends = [1 : i32, 768 : i32, 1 : i32, 60 : i32, 106 : i32]
    // CHECK-SAME: (tensor<1x768x4x60x106xbf16>) -> tensor<1x768x1x60x106xbf16>
    // CHECK: "ttir.repeat"
    // CHECK-SAME: repeat_dimensions = array<i64: 1, 1, 2, 1, 1>
    // CHECK-SAME: (tensor<1x768x1x60x106xbf16>) -> tensor<1x768x2x60x106xbf16>
    // Back pad: slice [3,4) along indexed dim, repeated 2x.
    // CHECK: "ttir.slice_static"
    // CHECK-SAME: begins = [0 : i32, 0 : i32, 3 : i32, 0 : i32, 0 : i32]
    // CHECK-SAME: ends = [1 : i32, 768 : i32, 4 : i32, 60 : i32, 106 : i32]
    // CHECK-SAME: (tensor<1x768x4x60x106xbf16>) -> tensor<1x768x1x60x106xbf16>
    // CHECK: "ttir.repeat"
    // CHECK-SAME: repeat_dimensions = array<i64: 1, 1, 2, 1, 1>
    // CHECK-SAME: (tensor<1x768x1x60x106xbf16>) -> tensor<1x768x2x60x106xbf16>
    // CHECK: "ttir.concat"
    // CHECK-SAME: dim = 2
    // CHECK-SAME: (tensor<1x768x2x60x106xbf16>, tensor<1x768x4x60x106xbf16>, tensor<1x768x2x60x106xbf16>) -> tensor<1x768x8x60x106xbf16>
    // CHECK-NOT: stablehlo.gather
    %1 = "stablehlo.constant"() <{value = dense<[[0], [0], [0], [1], [2], [3], [3], [3]]> : tensor<8x1xi64>}> : () -> tensor<8x1xi64>
    %2 = "stablehlo.gather"(%arg0, %1) <{dimension_numbers = #stablehlo.gather<offset_dims = [0, 1, 3, 4], collapsed_slice_dims = [2], start_index_map = [2], index_vector_dim = 1>, indices_are_sorted = false, slice_sizes = array<i64: 1, 768, 1, 60, 106>}> : (tensor<1x768x4x60x106xbf16>, tensor<8x1xi64>) -> tensor<1x768x8x60x106xbf16>
    return %2 : tensor<1x768x8x60x106xbf16>
  }
}

// Constant-uniform indices like [1, 1, ..., 1] are not replicate-padding;
// must fall through to the embedding lowering instead of slice/repeat/concat.
// CHECK-LABEL: func.func @uniform_max_indices_falls_through
module @test_gather_uniform_max_indices {
  func.func @uniform_max_indices_falls_through(%arg0: tensor<2x768xf32>) -> tensor<193x768xf32> {
    // CHECK-NOT: "ttir.concat"
    // CHECK: "ttir.embedding"
    // CHECK-NOT: stablehlo.gather
    %1 = "stablehlo.constant"() <{value = dense<1> : tensor<193xui32>}> : () -> tensor<193xui32>
    %2 = "stablehlo.gather"(%arg0, %1) <{dimension_numbers = #stablehlo.gather<offset_dims = [1], collapsed_slice_dims = [0], start_index_map = [0], index_vector_dim = 1>, indices_are_sorted = false, slice_sizes = array<i64: 1, 768>}> : (tensor<2x768xf32>, tensor<193xui32>) -> tensor<193x768xf32>
    return %2 : tensor<193x768xf32>
  }
}

// When the indexed dim has size 1 (maxIndex == 0), 0 and maxIndex collapse;
// all surplus indices become front padding without double-counting them as
// back padding.
// CHECK-LABEL: func.func @repeat_front_singleton_indexed_dim
module @test_gather_repeat_front_singleton_indexed_dim {
  func.func @repeat_front_singleton_indexed_dim(%arg0: tensor<1x16x1x40x64xbf16>) -> (tensor<1x16x3x40x64xbf16>) {
    // CHECK: "ttir.slice_static"
    // CHECK-SAME: begins = [0 : i32, 0 : i32, 0 : i32, 0 : i32, 0 : i32]
    // CHECK-SAME: ends = [1 : i32, 16 : i32, 1 : i32, 40 : i32, 64 : i32]
    // CHECK-SAME: (tensor<1x16x1x40x64xbf16>) -> tensor<1x16x1x40x64xbf16>
    // CHECK: "ttir.repeat"
    // CHECK-SAME: repeat_dimensions = array<i64: 1, 1, 2, 1, 1>
    // CHECK-SAME: (tensor<1x16x1x40x64xbf16>) -> tensor<1x16x2x40x64xbf16>
    // CHECK: "ttir.concat"
    // CHECK-SAME: dim = 2
    // CHECK-SAME: (tensor<1x16x2x40x64xbf16>, tensor<1x16x1x40x64xbf16>) -> tensor<1x16x3x40x64xbf16>
    // CHECK-NOT: stablehlo.gather
    %1 = "stablehlo.constant"() <{value = dense<[[0], [0], [0]]> : tensor<3x1xi64>}> : () -> tensor<3x1xi64>
    %2 = "stablehlo.gather"(%arg0, %1) <{dimension_numbers = #stablehlo.gather<offset_dims = [0, 1, 3, 4], collapsed_slice_dims = [2], start_index_map = [2], index_vector_dim = 1>, indices_are_sorted = false, slice_sizes = array<i64: 1, 16, 1, 40, 64>}> : (tensor<1x16x1x40x64xbf16>, tensor<3x1xi64>) -> tensor<1x16x3x40x64xbf16>
    return %2 : tensor<1x16x3x40x64xbf16>
  }
}

// Consecutive ascending indices [0, 1, ..., N-1] would otherwise match the
// replicate-padding heuristic, but a batched index vector (start_indices shape
// [1, N]) makes the gather output rank (3) exceed the operand rank (2). The
// slice/repeat/concat rewrite reconstructs operand-shaped frames, so it cannot
// represent that output and must fall through to the embedding lowering. This
// is the InternLM2 RoPE `cos[position_ids]` pattern (operand [S, D], indices
// [1, S], output [1, S, D]); matching it here produced an invalid concat
// ('ttir.concat' output dim mismatch "1 vs. N").
//
// This covers only the output-rank-mismatch leg of the legality guard; see
// @gather_offset_dim_mismatch_falls_through below for the same-rank leg.
// CHECK-LABEL: func.func @rope_cos_gather_falls_through
module @test_gather_rope_cos {
  func.func @rope_cos_gather_falls_through(%arg0: tensor<8x4xbf16>) -> tensor<1x8x4xbf16> {
    // CHECK-NOT: "ttir.concat"
    // CHECK: "ttir.embedding"
    // CHECK-NOT: stablehlo.gather
    %1 = "stablehlo.constant"() <{value = dense<[[0, 1, 2, 3, 4, 5, 6, 7]]> : tensor<1x8xi64>}> : () -> tensor<1x8xi64>
    %2 = "stablehlo.gather"(%arg0, %1) <{dimension_numbers = #stablehlo.gather<offset_dims = [2], collapsed_slice_dims = [0], start_index_map = [0], index_vector_dim = 2>, indices_are_sorted = false, slice_sizes = array<i64: 1, 4>}> : (tensor<8x4xbf16>, tensor<1x8xi64>) -> tensor<1x8x4xbf16>
    return %2 : tensor<1x8x4xbf16>
  }
}

// Same output rank as the operand (2 == 2), but placing the offset dim first
// (offset_dims = [0]) puts the batch dim in output position 1 instead of
// position 0. Indices [0, 1, ..., 7] are consecutive ascending, so this would
// otherwise match the replicate-padding heuristic, but the guard's positional
// non-indexed-dim comparison catches that output dim 1 (size 8, from the
// batch) does not match operand dim 1 (size 4, D) and rejects the match. A
// literal slice/repeat/concat of operand-shaped [8, 4] frames cannot produce
// a [4, 8] result, so this must fall through to the embedding lowering
// (which permutes/reshapes to handle arbitrary offset_dims ordering).
// CHECK-LABEL: func.func @gather_offset_dim_mismatch_falls_through
module @test_gather_offset_dim_mismatch {
  func.func @gather_offset_dim_mismatch_falls_through(%arg0: tensor<8x4xbf16>) -> tensor<4x8xbf16> {
    // CHECK-NOT: "ttir.concat"
    // CHECK: "ttir.embedding"
    // CHECK-NOT: stablehlo.gather
    %1 = "stablehlo.constant"() <{value = dense<[0, 1, 2, 3, 4, 5, 6, 7]> : tensor<8xi64>}> : () -> tensor<8xi64>
    %2 = "stablehlo.gather"(%arg0, %1) <{dimension_numbers = #stablehlo.gather<offset_dims = [0], collapsed_slice_dims = [0], start_index_map = [0], index_vector_dim = 1>, indices_are_sorted = false, slice_sizes = array<i64: 1, 4>}> : (tensor<8x4xbf16>, tensor<8xi64>) -> tensor<4x8xbf16>
    return %2 : tensor<4x8xbf16>
  }
}
