// REQUIRES: stablehlo
// RUN: not ttmlir-opt --split-input-file --stablehlo-to-ttir-pipeline %s 2>&1 | FileCheck %s
// Negative tests for the rank-reducing single-dim scatter path. These are
// valid stablehlo but fall outside the tightly-gated form accepted by
// checkBasicLegality, so conversion must reject them (fail to legalize) rather
// than silently miscompile.

// The update dropped more than one axis (operand rank 3, update rank 1), i.e.
// two inserted window dims. Only a single collapsed axis is supported.
// -----
module attributes {} {
  func.func @scatter_rank_reducing_drops_two_dims(%operand: tensor<4x5x6xf32>, %indices: tensor<1xi64>, %update: tensor<6xf32>) -> tensor<4x5x6xf32> {
    // CHECK: error: failed to legalize operation 'stablehlo.scatter'
    %0 = "stablehlo.scatter"(%operand, %indices, %update) <{
      scatter_dimension_numbers = #stablehlo.scatter<
        update_window_dims = [0],
        inserted_window_dims = [0, 1],
        scatter_dims_to_operand_dims = [0],
        index_vector_dim = 0>
    }> ({
    ^bb0(%a: tensor<f32>, %b: tensor<f32>):
      stablehlo.return %b : tensor<f32>
    }) : (tensor<4x5x6xf32>, tensor<1xi64>, tensor<6xf32>) -> tensor<4x5x6xf32>
    return %0 : tensor<4x5x6xf32>
  }
}

// The collapsed (inserted) window dim is dim 1, but the scattered axis is dim 0
// they must be the same axis for the rank-reducing form.
// -----
module attributes {} {
  func.func @scatter_rank_reducing_axis_mismatch(%operand: tensor<3x4xf32>, %indices: tensor<1xi64>, %update: tensor<3xf32>) -> tensor<3x4xf32> {
    // CHECK: error: failed to legalize operation 'stablehlo.scatter'
    %0 = "stablehlo.scatter"(%operand, %indices, %update) <{
      scatter_dimension_numbers = #stablehlo.scatter<
        update_window_dims = [0],
        inserted_window_dims = [1],
        scatter_dims_to_operand_dims = [0],
        index_vector_dim = 0>
    }> ({
    ^bb0(%a: tensor<f32>, %b: tensor<f32>):
      stablehlo.return %b : tensor<f32>
    }) : (tensor<3x4xf32>, tensor<1xi64>, tensor<3xf32>) -> tensor<3x4xf32>
    return %0 : tensor<3x4xf32>
  }
}
