// REQUIRES: stablehlo
// RUN: ttmlir-opt --stablehlo-to-ttir-pipeline -o %t %s
// RUN: FileCheck %s --input-file=%t
module attributes {} {
  // Gather as dynamic slice: operand 1x256x256x128, indices 3xi32 -> 1x128x256x128
  // start_index_map = [1, 2, 3], offset_dims = [0, 1, 2, 3], no collapsed dims.
  // CHECK-LABEL: func.func @gather_dynamic_slice_3d_index
  func.func @gather_dynamic_slice_3d_index(%operand: tensor<1x256x256x128xbf16>, %start_indices: tensor<3xi32>) -> tensor<1x128x256x128xbf16> {
    // CHECK: "ttir.constant"
    // CHECK: "ttir.slice_static"
    // CHECK: "ttir.slice_static"
    // CHECK: "ttir.slice_static"
    // CHECK: "ttir.concat"
    // CHECK: "ttir.constant"
    // CHECK: "ttir.add"
    // CHECK: "ttir.slice_dynamic"
    // CHECK-SAME: (tensor<1x256x256x128xbf16>, tensor<4xi32>, tensor<4xi32>) -> tensor<1x128x256x128xbf16>
    %0 = "stablehlo.gather"(%operand, %start_indices) <{dimension_numbers = #stablehlo.gather<offset_dims = [0, 1, 2, 3], start_index_map = [1, 2, 3]>, indices_are_sorted = true, slice_sizes = array<i64: 1, 128, 256, 128>}> : (tensor<1x256x256x128xbf16>, tensor<3xi32>) -> tensor<1x128x256x128xbf16>
    return %0 : tensor<1x128x256x128xbf16>
  }

  // Gather as dynamic slice: simple 2D case with single index dim.
  // CHECK-LABEL: func.func @gather_dynamic_slice_1d_index
  func.func @gather_dynamic_slice_1d_index(%operand: tensor<10x20xf32>, %start_indices: tensor<1xi32>) -> tensor<10x5xf32> {
    // CHECK: "ttir.constant"
    // CHECK: "ttir.concat"
    // CHECK: "ttir.constant"
    // CHECK: "ttir.add"
    // CHECK: "ttir.slice_dynamic"
    // CHECK-SAME: (tensor<10x20xf32>, tensor<2xi32>, tensor<2xi32>) -> tensor<10x5xf32>
    %0 = "stablehlo.gather"(%operand, %start_indices) <{dimension_numbers = #stablehlo.gather<offset_dims = [0, 1], start_index_map = [1]>, indices_are_sorted = true, slice_sizes = array<i64: 10, 5>}> : (tensor<10x20xf32>, tensor<1xi32>) -> tensor<10x5xf32>
    return %0 : tensor<10x5xf32>
  }

  // Gather as dynamic slice: all dims indexed.
  // CHECK-LABEL: func.func @gather_dynamic_slice_all_dims
  func.func @gather_dynamic_slice_all_dims(%operand: tensor<8x16x32xf32>, %start_indices: tensor<3xi32>) -> tensor<4x8x16xf32> {
    // CHECK: "ttir.constant"
    // CHECK: "ttir.add"
    // CHECK: "ttir.slice_dynamic"
    // CHECK-SAME: (tensor<8x16x32xf32>, tensor<3xi32>, tensor<3xi32>) -> tensor<4x8x16xf32>
    %0 = "stablehlo.gather"(%operand, %start_indices) <{dimension_numbers = #stablehlo.gather<offset_dims = [0, 1, 2], start_index_map = [0, 1, 2]>, indices_are_sorted = true, slice_sizes = array<i64: 4, 8, 16>}> : (tensor<8x16x32xf32>, tensor<3xi32>) -> tensor<4x8x16xf32>
    return %0 : tensor<4x8x16xf32>
  }
}
