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
