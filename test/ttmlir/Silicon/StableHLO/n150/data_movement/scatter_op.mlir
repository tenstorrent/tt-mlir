// REQUIRES: stablehlo
// RUN: rm -rf %t.ttnn
// RUN: rm -rf %t.mlir
// RUN: ttmlir-opt --stablehlo-to-ttir-pipeline %s -o %t.mlir --ttir-to-ttnn-backend-pipeline="system-desc-path=%system_desc_path%"
// RUN: ttmlir-translate --ttnn-to-flatbuffer -o %t.ttnn %t.mlir
// RUN: FileCheck --input-file=%t.mlir %s

// Rank-reducing single-dim scatter `x.at[i, :].set(v)`: the StableHLO update
// drops the scattered axis, so the conversion promotes source/index back to
// operand rank with a size-1 scattered axis and lowers to a single ttnn.scatter.

func.func @rank_reducing_scatter_row(%operand: tensor<32x32xbf16>, %indices: tensor<1xi64>, %update: tensor<32xbf16>) -> tensor<32x32xbf16> {
  // CHECK-LABEL: func.func @rank_reducing_scatter_row
  // CHECK: "ttnn.scatter"
  %result = "stablehlo.scatter"(%operand, %indices, %update) <{
    scatter_dimension_numbers = #stablehlo.scatter<
      update_window_dims = [0],
      inserted_window_dims = [0],
      scatter_dims_to_operand_dims = [0],
      index_vector_dim = 0>
  }> ({
  ^bb0(%a: tensor<bf16>, %b: tensor<bf16>):
    stablehlo.return %b : tensor<bf16>
  }) : (tensor<32x32xbf16>, tensor<1xi64>, tensor<32xbf16>) -> tensor<32x32xbf16>
  return %result : tensor<32x32xbf16>
}

// Scatter into a non-leading axis `x.at[:, i, :].set(v)` (scattered axis is dim 1).
func.func @rank_reducing_scatter_middle_dim(%operand: tensor<2x32x32xbf16>, %indices: tensor<1xi64>, %update: tensor<2x32xbf16>) -> tensor<2x32x32xbf16> {
  // CHECK-LABEL: func.func @rank_reducing_scatter_middle_dim
  // CHECK: "ttnn.scatter"
  %result = "stablehlo.scatter"(%operand, %indices, %update) <{
    scatter_dimension_numbers = #stablehlo.scatter<
      update_window_dims = [0, 1], 
      inserted_window_dims = [1], 
      scatter_dims_to_operand_dims = [1], 
      index_vector_dim = 0>
  }> ({
  ^bb0(%a: tensor<bf16>, %b: tensor<bf16>):
    stablehlo.return %b : tensor<bf16>
  }) : (tensor<2x32x32xbf16>, tensor<1xi64>, tensor<2x32xbf16>) -> tensor<2x32x32xbf16>
  return %result : tensor<2x32x32xbf16>
}

// Scatter into a 1-D operand `x.at[i].set(v)`: the update collapses all the way
// to a scalar, so source/index are promoted to a size-1 axis and it lowers to a
// single ttnn.scatter along dim 0.
func.func @rank_reducing_scatter_1d(%operand: tensor<32xbf16>, %indices: tensor<1xi64>, %update: tensor<bf16>) -> tensor<32xbf16> {
  // CHECK-LABEL: func.func @rank_reducing_scatter_1d
  // CHECK: "ttnn.scatter"
  %result = "stablehlo.scatter"(%operand, %indices, %update) <{
    scatter_dimension_numbers = #stablehlo.scatter<
      update_window_dims = [],
      inserted_window_dims = [0],
      scatter_dims_to_operand_dims = [0],
      index_vector_dim = 0>
  }> ({
  ^bb0(%a: tensor<bf16>, %b: tensor<bf16>):
    stablehlo.return %b : tensor<bf16>
  }) : (tensor<32xbf16>, tensor<1xi64>, tensor<bf16>) -> tensor<32xbf16>
  return %result : tensor<32xbf16>
}
