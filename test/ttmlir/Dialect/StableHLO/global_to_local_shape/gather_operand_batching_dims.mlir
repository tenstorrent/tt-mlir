// REQUIRES: stablehlo
// RUN: rm -rf %t.mlir
// RUN: ttmlir-opt -split-input-file --stablehlo-pipeline -o %t.mlir %s
// RUN: FileCheck %s --input-file=%t.mlir

// Regression test for `UpdateGlobalToLocalShapes`' handling of stablehlo.gather
// ops that carry `operand_batching_dims`

sdy.mesh @mesh = <["batch"=2]>

func.func @gather_sharded_batching_dim(
    %arg0: tensor<4x2xf32> {sdy.sharding = #sdy.sharding<@mesh, [{}, {"batch"}]>},
    %arg1: tensor<3x2x1xi32> {sdy.sharding = #sdy.sharding<@mesh, [{}, {"batch"}, {}]>}
) -> tensor<3x2xf32> {
  %0 = "stablehlo.gather"(%arg0, %arg1) <{
      dimension_numbers = #stablehlo.gather<
          collapsed_slice_dims = [0],
          operand_batching_dims = [1],
          start_indices_batching_dims = [1],
          start_index_map = [0],
          index_vector_dim = 2>,
      indices_are_sorted = false,
      slice_sizes = array<i64: 1, 1>
  }> : (tensor<4x2xf32>, tensor<3x2x1xi32>) -> tensor<3x2xf32>
  return %0 : tensor<3x2xf32>
}

// CHECK-LABEL: func.func @gather_sharded_batching_dim
// CHECK:       sdy.manual_computation
// CHECK:       "stablehlo.gather"
// CHECK-SAME:  collapsed_slice_dims = [0]
// CHECK-SAME:  operand_batching_dims = [1]
// CHECK-SAME:  start_indices_batching_dims = [1]
// CHECK-SAME:  start_index_map = [0]
// CHECK-SAME:  index_vector_dim = 2
// CHECK-SAME:  slice_sizes = array<i64: 1, 1>
// CHECK-SAME:  (tensor<4x1xf32>, tensor<3x1x1xi32>) -> tensor<3x1xf32>

// -----

sdy.mesh @mesh = <["batch"=2]>

func.func @gather_3d_sharded_batching_dim(
    %arg0: tensor<2x4x6xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"batch"}, {}, {}]>},
    %arg1: tensor<2x4x3x1xi32> {sdy.sharding = #sdy.sharding<@mesh, [{"batch"}, {}, {}, {}]>}
) -> tensor<2x4x3xf32> {
  %0 = "stablehlo.gather"(%arg0, %arg1) <{
      dimension_numbers = #stablehlo.gather<
          collapsed_slice_dims = [2],
          operand_batching_dims = [0, 1],
          start_indices_batching_dims = [0, 1],
          start_index_map = [2],
          index_vector_dim = 3>,
      indices_are_sorted = false,
      slice_sizes = array<i64: 1, 1, 1>
  }> : (tensor<2x4x6xf32>, tensor<2x4x3x1xi32>) -> tensor<2x4x3xf32>
  return %0 : tensor<2x4x3xf32>
}

// CHECK-LABEL: func.func @gather_3d_sharded_batching_dim
// CHECK:       sdy.manual_computation
// CHECK:       "stablehlo.gather"
// CHECK-SAME:  operand_batching_dims = [0, 1]
// CHECK-SAME:  start_indices_batching_dims = [0, 1]
// CHECK-SAME:  slice_sizes = array<i64: 1, 1, 1>
// CHECK-SAME:  (tensor<1x4x6xf32>, tensor<1x4x3x1xi32>) -> tensor<1x4x3xf32>
