// RUN: ttmlir-opt --stablehlo-complex-data-type-conversion -o %t %s
// RUN: FileCheck %s --input-file=%t
// REQUIRES: stablehlo

// A complex gather composite is converted in the StableHLO domain: the
// composite and its decomposition are both retyped to the float-pair form.

module @complex_composite_gather attributes {mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} {

  // The composite stays a composite, retyped to the float-pair form.
  // CHECK-LABEL: func.func @main
  // CHECK-SAME: %arg0: tensor<1x512x16x2xf32>
  // CHECK-SAME: %arg1: tensor<1x4352x16xi64>
  // CHECK-SAME: -> tensor<1x4352x16x2xf32>
  func.func @main(%arg0: tensor<1x512x16xcomplex<f32>>, %arg1: tensor<1x4352x16xi64>) -> tensor<1x4352x16xcomplex<f32>> {
    // CHECK: stablehlo.composite "tenstorrent.gather"
    // CHECK-SAME: decomposition = @tenstorrent.gather.impl
    // CHECK-SAME: (tensor<1x512x16x2xf32>, tensor<1x4352x16xi64>) -> tensor<1x4352x16x2xf32>
    // CHECK-NOT: ttir.
    %0 = stablehlo.composite "tenstorrent.gather" %arg0, %arg1 {composite_attributes = {dim = 1 : i64, sparse_grad = false}, decomposition = @tenstorrent.gather.impl} : (tensor<1x512x16xcomplex<f32>>, tensor<1x4352x16xi64>) -> tensor<1x4352x16xcomplex<f32>>
    return %0 : tensor<1x4352x16xcomplex<f32>>
  }

  // The decomposition is converted too; its inner gather gains the trailing dim.
  // CHECK-LABEL: func.func private @tenstorrent.gather.impl
  // CHECK-SAME: %arg0: tensor<1x512x16x2xf32>
  // CHECK-SAME: -> tensor<1x4352x16x2xf32>
  func.func private @tenstorrent.gather.impl(%arg0: tensor<1x512x16xcomplex<f32>>, %arg1: tensor<1x4352x16xi64>) -> tensor<1x4352x16xcomplex<f32>> {
    %c = stablehlo.constant dense<0> : tensor<1x4352x16x1xui32>
    %0 = stablehlo.convert %arg1 : (tensor<1x4352x16xi64>) -> tensor<1x4352x16xui32>
    %1 = stablehlo.reshape %0 : (tensor<1x4352x16xui32>) -> tensor<1x4352x16x1xui32>
    %2 = stablehlo.iota dim = 0 : tensor<16xui32>
    %3 = stablehlo.broadcast_in_dim %2, dims = [2] : (tensor<16xui32>) -> tensor<1x4352x16x1xui32>
    %4 = stablehlo.concatenate %c, %1, %3, dim = 3 : (tensor<1x4352x16x1xui32>, tensor<1x4352x16x1xui32>, tensor<1x4352x16x1xui32>) -> tensor<1x4352x16x3xui32>
    // CHECK: stablehlo.gather
    // CHECK-SAME: slice_sizes = array<i64: 1, 1, 1, 2>
    // CHECK-SAME: -> tensor<1x4352x16x2xf32>
    %5 = "stablehlo.gather"(%arg0, %4) <{dimension_numbers = #stablehlo.gather<collapsed_slice_dims = [0, 1, 2], start_index_map = [0, 1, 2], index_vector_dim = 3>, slice_sizes = array<i64: 1, 1, 1>}> : (tensor<1x512x16xcomplex<f32>>, tensor<1x4352x16x3xui32>) -> tensor<1x4352x16xcomplex<f32>>
    return %5 : tensor<1x4352x16xcomplex<f32>>
  }
}
