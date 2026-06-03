// RUN: ttmlir-opt --stablehlo-complex-data-type-conversion -o %t %s
// RUN: FileCheck %s --input-file=%t

// RoPE-style row gather: table[pos_ids[:, axis]] (Z-Image axis-1 table shape).
module {
  func.func @test_complex_gather_axis0(
      %table: tensor<512x24xcomplex<f32>>,
      %indices: tensor<3616x1xi32>) -> tensor<3616x24xcomplex<f32>> {
    // CHECK: stablehlo.gather
    // CHECK-SAME: offset_dims = [1, 2]
    // CHECK-SAME: slice_sizes = array<i64: 1, 24, 2>
    // CHECK-SAME: (tensor<512x24x2xf32>, tensor<3616x1xi32>) -> tensor<3616x24x2xf32>
    %0 = "stablehlo.gather"(%table, %indices) {
      dimension_numbers = #stablehlo.gather<
        offset_dims = [1],
        collapsed_slice_dims = [0],
        operand_batching_dims = [],
        start_indices_batching_dims = [],
        start_index_map = [0],
        index_vector_dim = 1
      >,
      slice_sizes = array<i64: 1, 24>,
      indices_are_sorted = false
    } : (tensor<512x24xcomplex<f32>>, tensor<3616x1xi32>) -> tensor<3616x24xcomplex<f32>>
    return %0 : tensor<3616x24xcomplex<f32>>
  }
}
