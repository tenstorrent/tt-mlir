// REQUIRES: stablehlo
// RUN: ttmlir-opt --stablehlo-simplify -o %t %s
// RUN: FileCheck %s --input-file=%t

module @collapse_row_scatter {
  // A row scatter wearing a multi-dim costume: index = concat(row_idx, column
  // iota), scatter_dims=[0,1]. The pass must collapse it to a single-axis
  // scatter (dims=[0], column as window) so it is not flattened to 1D downstream.
  func.func @disguised_row_scatter(%operand: tensor<4x8xbf16>,
                                   %rows: tensor<2x8x1xi64>,
                                   %update: tensor<2x8xbf16>) -> tensor<4x8xbf16> {
    %iota = stablehlo.iota dim = 1 : tensor<2x8x1xi64>
    %idx = stablehlo.concatenate %rows, %iota, dim = 2
        : (tensor<2x8x1xi64>, tensor<2x8x1xi64>) -> tensor<2x8x2xi64>
    %r = "stablehlo.scatter"(%operand, %idx, %update) <{scatter_dimension_numbers = #stablehlo.scatter<inserted_window_dims = [0, 1], scatter_dims_to_operand_dims = [0, 1], index_vector_dim = 2>}> ({
    ^bb0(%a: tensor<bf16>, %b: tensor<bf16>):
      %s = stablehlo.add %a, %b : tensor<bf16>
      stablehlo.return %s : tensor<bf16>
    }) : (tensor<4x8xbf16>, tensor<2x8x2xi64>, tensor<2x8xbf16>) -> tensor<4x8xbf16>
    return %r : tensor<4x8xbf16>
    // CHECK-LABEL: func.func @disguised_row_scatter
    // CHECK: stablehlo.reshape
    // CHECK-SAME: -> tensor<2x1xi64>
    // CHECK: "stablehlo.scatter"
    // CHECK-SAME: update_window_dims = [1]
    // CHECK-SAME: scatter_dims_to_operand_dims = [0]
  }

  // A genuine multi-coordinate scatter (both index components are real, no iota)
  // must be left unchanged.
  func.func @genuine_2d_scatter(%operand: tensor<4x8xbf16>,
                                %rows: tensor<3x1xi64>,
                                %cols: tensor<3x1xi64>,
                                %update: tensor<3xbf16>) -> tensor<4x8xbf16> {
    %idx = stablehlo.concatenate %rows, %cols, dim = 1
        : (tensor<3x1xi64>, tensor<3x1xi64>) -> tensor<3x2xi64>
    %r = "stablehlo.scatter"(%operand, %idx, %update) <{scatter_dimension_numbers = #stablehlo.scatter<inserted_window_dims = [0, 1], scatter_dims_to_operand_dims = [0, 1], index_vector_dim = 1>}> ({
    ^bb0(%a: tensor<bf16>, %b: tensor<bf16>):
      %s = stablehlo.add %a, %b : tensor<bf16>
      stablehlo.return %s : tensor<bf16>
    }) : (tensor<4x8xbf16>, tensor<3x2xi64>, tensor<3xbf16>) -> tensor<4x8xbf16>
    return %r : tensor<4x8xbf16>
    // CHECK-LABEL: func.func @genuine_2d_scatter
    // CHECK: scatter_dims_to_operand_dims = [0, 1]
  }

  // Same row scatter, but the column iota is materialized as
  // `iota -> broadcast_in_dim` — exactly how the frontend lowers the positional
  // column. Verifies tracesToIota looks through broadcast_in_dim.
  func.func @disguised_row_scatter_broadcast_iota(%operand: tensor<4x8xbf16>,
                                                  %rows: tensor<2x8x1xi64>,
                                                  %update: tensor<2x8xbf16>) -> tensor<4x8xbf16> {
    %iota = stablehlo.iota dim = 0 : tensor<8xi64>
    %bcast = stablehlo.broadcast_in_dim %iota, dims = [1]
        : (tensor<8xi64>) -> tensor<2x8x1xi64>
    %idx = stablehlo.concatenate %rows, %bcast, dim = 2
        : (tensor<2x8x1xi64>, tensor<2x8x1xi64>) -> tensor<2x8x2xi64>
    %r = "stablehlo.scatter"(%operand, %idx, %update) <{scatter_dimension_numbers = #stablehlo.scatter<inserted_window_dims = [0, 1], scatter_dims_to_operand_dims = [0, 1], index_vector_dim = 2>}> ({
    ^bb0(%a: tensor<bf16>, %b: tensor<bf16>):
      %s = stablehlo.add %a, %b : tensor<bf16>
      stablehlo.return %s : tensor<bf16>
    }) : (tensor<4x8xbf16>, tensor<2x8x2xi64>, tensor<2x8xbf16>) -> tensor<4x8xbf16>
    return %r : tensor<4x8xbf16>
    // CHECK-LABEL: func.func @disguised_row_scatter_broadcast_iota
    // CHECK: "stablehlo.scatter"
    // CHECK-SAME: update_window_dims = [1]
    // CHECK-SAME: scatter_dims_to_operand_dims = [0]
  }

  // Same row scatter, but the column iota goes through a reshape. Verifies
  // tracesToIota looks through reshape.
  func.func @disguised_row_scatter_reshape_iota(%operand: tensor<4x8xbf16>,
                                                %rows: tensor<2x8x1xi64>,
                                                %update: tensor<2x8xbf16>) -> tensor<4x8xbf16> {
    %iota = stablehlo.iota dim = 1 : tensor<2x8xi64>
    %rs = stablehlo.reshape %iota : (tensor<2x8xi64>) -> tensor<2x8x1xi64>
    %idx = stablehlo.concatenate %rows, %rs, dim = 2
        : (tensor<2x8x1xi64>, tensor<2x8x1xi64>) -> tensor<2x8x2xi64>
    %r = "stablehlo.scatter"(%operand, %idx, %update) <{scatter_dimension_numbers = #stablehlo.scatter<inserted_window_dims = [0, 1], scatter_dims_to_operand_dims = [0, 1], index_vector_dim = 2>}> ({
    ^bb0(%a: tensor<bf16>, %b: tensor<bf16>):
      %s = stablehlo.add %a, %b : tensor<bf16>
      stablehlo.return %s : tensor<bf16>
    }) : (tensor<4x8xbf16>, tensor<2x8x2xi64>, tensor<2x8xbf16>) -> tensor<4x8xbf16>
    return %r : tensor<4x8xbf16>
    // CHECK-LABEL: func.func @disguised_row_scatter_reshape_iota
    // CHECK: "stablehlo.scatter"
    // CHECK-SAME: update_window_dims = [1]
    // CHECK-SAME: scatter_dims_to_operand_dims = [0]
  }

  // The second index component traces to an iota, but the iota varies along the
  // row axis (broadcast dims = [0]), not the column axis — so it is NOT the
  // identity column map. Collapsing would miscompile, so the scatter must be
  // left unchanged at dims=[0,1].
  func.func @row_axis_iota_not_collapsed(%operand: tensor<4x8xbf16>,
                                         %rows: tensor<2x8x1xi64>,
                                         %update: tensor<2x8xbf16>) -> tensor<4x8xbf16> {
    %iota = stablehlo.iota dim = 0 : tensor<2xi64>
    %bcast = stablehlo.broadcast_in_dim %iota, dims = [0]
        : (tensor<2xi64>) -> tensor<2x8x1xi64>
    %idx = stablehlo.concatenate %rows, %bcast, dim = 2
        : (tensor<2x8x1xi64>, tensor<2x8x1xi64>) -> tensor<2x8x2xi64>
    %r = "stablehlo.scatter"(%operand, %idx, %update) <{scatter_dimension_numbers = #stablehlo.scatter<inserted_window_dims = [0, 1], scatter_dims_to_operand_dims = [0, 1], index_vector_dim = 2>}> ({
    ^bb0(%a: tensor<bf16>, %b: tensor<bf16>):
      %s = stablehlo.add %a, %b : tensor<bf16>
      stablehlo.return %s : tensor<bf16>
    }) : (tensor<4x8xbf16>, tensor<2x8x2xi64>, tensor<2x8xbf16>) -> tensor<4x8xbf16>
    return %r : tensor<4x8xbf16>
    // CHECK-LABEL: func.func @row_axis_iota_not_collapsed
    // CHECK: scatter_dims_to_operand_dims = [0, 1]
  }
}
