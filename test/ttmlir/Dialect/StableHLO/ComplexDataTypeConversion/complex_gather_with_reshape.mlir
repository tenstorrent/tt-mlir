// RUN: ttmlir-opt --stablehlo-complex-math-expander --stablehlo-complex-data-type-conversion -o %t %s
// RUN: FileCheck %s --input-file=%t

// XLA often emits reshapes around the polar table before gather (Z-Image repro).
module {
  func.func @test_complex_gather_after_reshape(
      %arg0: tensor<3616x3xi32>,
      %arg1: tensor<512x24xcomplex<f32>>) -> tensor<3616x24xcomplex<f32>> {
    // CHECK: stablehlo.gather
    // CHECK-SAME: offset_dims = [1, 2]
    // CHECK-SAME: slice_sizes = array<i64: 1, 24, 2>
    // CHECK-NOT: complex<f32>
    %0 = stablehlo.reshape %arg1 : (tensor<512x24xcomplex<f32>>) -> tensor<1x512x24xcomplex<f32>>
    %1 = stablehlo.reshape %0 : (tensor<1x512x24xcomplex<f32>>) -> tensor<512x24xcomplex<f32>>
    %2 = stablehlo.slice %arg0 [0:3616, 1:2] : (tensor<3616x3xi32>) -> tensor<3616x1xi32>
    %3 = stablehlo.convert %2 : (tensor<3616x1xi32>) -> tensor<3616x1xi64>
    %4 = "stablehlo.gather"(%1, %3) <{
      dimension_numbers = #stablehlo.gather<
        offset_dims = [1],
        collapsed_slice_dims = [0],
        start_index_map = [0],
        index_vector_dim = 1
      >,
      slice_sizes = array<i64: 1, 24>,
      indices_are_sorted = false
    }> : (tensor<512x24xcomplex<f32>>, tensor<3616x1xi64>) -> tensor<3616x24xcomplex<f32>>
    return %4 : tensor<3616x24xcomplex<f32>>
  }
}
