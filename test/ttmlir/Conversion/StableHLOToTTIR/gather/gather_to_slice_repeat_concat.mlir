// REQUIRES: stablehlo
// RUN: ttmlir-opt --stablehlo-to-ttir-pipeline -o %t %s
// RUN: FileCheck %s --input-file=%t

// CHECK-LABEL: func.func @repeat_front
module @test_gather_repeat_front {
  func.func @repeat_front(%arg0: tensor<1x768x4x60x106xbf16>) -> (tensor<1x768x6x60x106xbf16>) {
    // CHECK: "ttir.slice_static"
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
    // CHECK: "ttir.slice_static"
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
    // CHECK: "ttir.slice_static"
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
    // CHECK: "ttir.slice_static"
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
