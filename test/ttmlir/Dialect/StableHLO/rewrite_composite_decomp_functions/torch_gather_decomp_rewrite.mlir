// REQUIRES: stablehlo
// RUN: ttmlir-opt -split-input-file --rewrite-composite-decomp-functions -o %t %s
// RUN: FileCheck %s --input-file=%t

// Exercise rewriteTenstorrentGatherDecomp across the full space of valid
// torch.gather inputs: every combination of
//   - input rank (1, 2, 3, 4),
//   - `dim` position (first, middle, last, negative with different magnitudes,
//     default/missing attribute),
//   - non-dim axis shape relation (all equal vs. one/all strictly larger on
//     input, which forces a leading stablehlo.slice),
//   - gather-dim axis relation (input.size(dim) vs. index.size(dim): larger,
//     equal, smaller — torch.gather allows any),
//   - index element type (i32, i64, ui32, ui16),
//   - other composite_attributes (sparse_grad) coexisting with `dim`.
// Function signature preservation is covered at the end.

// -----

// Rank-2, dim=0, non-dim axis sizes match: no slice, single reshape + gather.
module @Rank2_Dim0_MatchingAxes {
  func.func @main(%arg0: tensor<5x3xf32>, %arg1: tensor<2x3xi32>) -> tensor<2x3xf32> {
    %0 = stablehlo.composite "tenstorrent.gather" %arg0, %arg1 {composite_attributes = {dim = 0 : i64}, decomposition = @tenstorrent.gather.impl_d0_match} : (tensor<5x3xf32>, tensor<2x3xi32>) -> tensor<2x3xf32>
    return %0 : tensor<2x3xf32>
  }
  // CHECK-LABEL: func.func private @tenstorrent.gather.impl_d0_match
  // CHECK-SAME: (%[[IN:.*]]: tensor<5x3xf32>, %[[IDX:.*]]: tensor<2x3xi32>) -> tensor<2x3xf32>
  // CHECK-NOT:  stablehlo.slice
  // CHECK:      %[[R:.*]] = stablehlo.reshape %[[IDX]] : (tensor<2x3xi32>) -> tensor<2x3x1xi32>
  // CHECK:      %[[G:.*]] = "stablehlo.gather"(%[[IN]], %[[R]])
  // CHECK-SAME: collapsed_slice_dims = [0]
  // CHECK-SAME: operand_batching_dims = [1]
  // CHECK-SAME: start_indices_batching_dims = [1]
  // CHECK-SAME: start_index_map = [0]
  // CHECK-SAME: index_vector_dim = 2
  // CHECK-SAME: slice_sizes = array<i64: 1, 1>
  // CHECK-SAME: (tensor<5x3xf32>, tensor<2x3x1xi32>) -> tensor<2x3xf32>
  // CHECK:      return %[[G]] : tensor<2x3xf32>
  func.func private @tenstorrent.gather.impl_d0_match(%arg0: tensor<5x3xf32>, %arg1: tensor<2x3xi32>) -> tensor<2x3xf32> {
    %0 = stablehlo.constant dense<0.000000e+00> : tensor<2x3xf32>
    return %0 : tensor<2x3xf32>
  }
}

// -----

// Rank-2, dim=1, non-dim axis sizes match: batching dim moves to axis 0.
module @Rank2_Dim1_MatchingAxes {
  func.func @main(%arg0: tensor<3x5xbf16>, %arg1: tensor<3x2xi64>) -> tensor<3x2xbf16> {
    %0 = stablehlo.composite "tenstorrent.gather" %arg0, %arg1 {composite_attributes = {dim = 1 : i64}, decomposition = @tenstorrent.gather.impl_d1_match} : (tensor<3x5xbf16>, tensor<3x2xi64>) -> tensor<3x2xbf16>
    return %0 : tensor<3x2xbf16>
  }
  // CHECK-LABEL: func.func private @tenstorrent.gather.impl_d1_match
  // CHECK-NOT:  stablehlo.slice
  // CHECK:      stablehlo.reshape {{.*}} : (tensor<3x2xi64>) -> tensor<3x2x1xi64>
  // CHECK:      "stablehlo.gather"
  // CHECK-SAME: collapsed_slice_dims = [1]
  // CHECK-SAME: operand_batching_dims = [0]
  // CHECK-SAME: start_indices_batching_dims = [0]
  // CHECK-SAME: start_index_map = [1]
  // CHECK-SAME: index_vector_dim = 2
  // CHECK-SAME: slice_sizes = array<i64: 1, 1>
  func.func private @tenstorrent.gather.impl_d1_match(%arg0: tensor<3x5xbf16>, %arg1: tensor<3x2xi64>) -> tensor<3x2xbf16> {
    %0 = stablehlo.constant dense<0.000000e+00> : tensor<3x2xbf16>
    return %0 : tensor<3x2xbf16>
  }
}

// -----

// Rank-2, dim=0, input's non-dim axis is strictly larger than the index's:
// the rewrite must emit a leading stablehlo.slice that trims axis 1 from
// 32 down to 16 so the batching-dim sizes match.
module @Rank2_Dim0_NonDimAxisLarger {
  func.func @main(%arg0: tensor<5x32xf32>, %arg1: tensor<2x16xi32>) -> tensor<2x16xf32> {
    %0 = stablehlo.composite "tenstorrent.gather" %arg0, %arg1 {composite_attributes = {dim = 0 : i64}, decomposition = @tenstorrent.gather.impl_d0_slice} : (tensor<5x32xf32>, tensor<2x16xi32>) -> tensor<2x16xf32>
    return %0 : tensor<2x16xf32>
  }
  // CHECK-LABEL: func.func private @tenstorrent.gather.impl_d0_slice
  // CHECK-SAME: (%[[IN:.*]]: tensor<5x32xf32>, %[[IDX:.*]]: tensor<2x16xi32>) -> tensor<2x16xf32>
  // CHECK:      %[[S:.*]] = stablehlo.slice %[[IN]] [0:5, 0:16] : (tensor<5x32xf32>) -> tensor<5x16xf32>
  // CHECK:      %[[R:.*]] = stablehlo.reshape %[[IDX]] : (tensor<2x16xi32>) -> tensor<2x16x1xi32>
  // CHECK:      %[[G:.*]] = "stablehlo.gather"(%[[S]], %[[R]])
  // CHECK-SAME: collapsed_slice_dims = [0]
  // CHECK-SAME: operand_batching_dims = [1]
  // CHECK-SAME: start_indices_batching_dims = [1]
  // CHECK-SAME: start_index_map = [0]
  // CHECK-SAME: index_vector_dim = 2
  // CHECK-SAME: slice_sizes = array<i64: 1, 1>
  // CHECK-SAME: (tensor<5x16xf32>, tensor<2x16x1xi32>) -> tensor<2x16xf32>
  // CHECK:      return %[[G]]
  func.func private @tenstorrent.gather.impl_d0_slice(%arg0: tensor<5x32xf32>, %arg1: tensor<2x16xi32>) -> tensor<2x16xf32> {
    %0 = stablehlo.constant dense<0.000000e+00> : tensor<2x16xf32>
    return %0 : tensor<2x16xf32>
  }
}

// -----

// Rank-2, dim=1, input's non-dim axis is strictly larger: slice trims
// axis 0 instead of axis 1.
module @Rank2_Dim1_NonDimAxisLarger {
  func.func @main(%arg0: tensor<10x5xbf16>, %arg1: tensor<6x3xi64>) -> tensor<6x3xbf16> {
    %0 = stablehlo.composite "tenstorrent.gather" %arg0, %arg1 {composite_attributes = {dim = 1 : i64}, decomposition = @tenstorrent.gather.impl_d1_slice} : (tensor<10x5xbf16>, tensor<6x3xi64>) -> tensor<6x3xbf16>
    return %0 : tensor<6x3xbf16>
  }
  // CHECK-LABEL: func.func private @tenstorrent.gather.impl_d1_slice
  // CHECK:      %[[S:.*]] = stablehlo.slice %{{.*}} [0:6, 0:5] : (tensor<10x5xbf16>) -> tensor<6x5xbf16>
  // CHECK:      stablehlo.reshape {{.*}} : (tensor<6x3xi64>) -> tensor<6x3x1xi64>
  // CHECK:      "stablehlo.gather"(%[[S]]
  // CHECK-SAME: collapsed_slice_dims = [1]
  // CHECK-SAME: operand_batching_dims = [0]
  // CHECK-SAME: start_indices_batching_dims = [0]
  // CHECK-SAME: start_index_map = [1]
  // CHECK-SAME: (tensor<6x5xbf16>, tensor<6x3x1xi64>) -> tensor<6x3xbf16>
  func.func private @tenstorrent.gather.impl_d1_slice(%arg0: tensor<10x5xbf16>, %arg1: tensor<6x3xi64>) -> tensor<6x3xbf16> {
    %0 = stablehlo.constant dense<0.000000e+00> : tensor<6x3xbf16>
    return %0 : tensor<6x3xbf16>
  }
}

// -----

// Negative dim = -1 on rank-2 normalizes to dim = 1.
module @NegativeDim_NegOne_Rank2 {
  func.func @main(%arg0: tensor<4x6xf32>, %arg1: tensor<4x3xi32>) -> tensor<4x3xf32> {
    %0 = stablehlo.composite "tenstorrent.gather" %arg0, %arg1 {composite_attributes = {dim = -1 : i64}, decomposition = @tenstorrent.gather.impl_neg1} : (tensor<4x6xf32>, tensor<4x3xi32>) -> tensor<4x3xf32>
    return %0 : tensor<4x3xf32>
  }
  // CHECK-LABEL: func.func private @tenstorrent.gather.impl_neg1
  // CHECK-NOT:  stablehlo.slice
  // CHECK:      "stablehlo.gather"
  // CHECK-SAME: collapsed_slice_dims = [1]
  // CHECK-SAME: operand_batching_dims = [0]
  // CHECK-SAME: start_indices_batching_dims = [0]
  // CHECK-SAME: start_index_map = [1]
  // CHECK-SAME: index_vector_dim = 2
  func.func private @tenstorrent.gather.impl_neg1(%arg0: tensor<4x6xf32>, %arg1: tensor<4x3xi32>) -> tensor<4x3xf32> {
    %0 = stablehlo.constant dense<0.000000e+00> : tensor<4x3xf32>
    return %0 : tensor<4x3xf32>
  }
}

// -----

// Negative dim = -2 on rank-3 normalizes to dim = 1. Exercises negation
// beyond the trailing-axis case.
module @NegativeDim_NegTwo_Rank3 {
  func.func @main(%arg0: tensor<2x5x4xf32>, %arg1: tensor<2x3x4xi32>) -> tensor<2x3x4xf32> {
    %0 = stablehlo.composite "tenstorrent.gather" %arg0, %arg1 {composite_attributes = {dim = -2 : i64}, decomposition = @tenstorrent.gather.impl_neg2} : (tensor<2x5x4xf32>, tensor<2x3x4xi32>) -> tensor<2x3x4xf32>
    return %0 : tensor<2x3x4xf32>
  }
  // CHECK-LABEL: func.func private @tenstorrent.gather.impl_neg2
  // CHECK-NOT:  stablehlo.slice
  // CHECK:      stablehlo.reshape {{.*}} : (tensor<2x3x4xi32>) -> tensor<2x3x4x1xi32>
  // CHECK:      "stablehlo.gather"
  // CHECK-SAME: collapsed_slice_dims = [1]
  // CHECK-SAME: operand_batching_dims = [0, 2]
  // CHECK-SAME: start_indices_batching_dims = [0, 2]
  // CHECK-SAME: start_index_map = [1]
  // CHECK-SAME: index_vector_dim = 3
  // CHECK-SAME: slice_sizes = array<i64: 1, 1, 1>
  func.func private @tenstorrent.gather.impl_neg2(%arg0: tensor<2x5x4xf32>, %arg1: tensor<2x3x4xi32>) -> tensor<2x3x4xf32> {
    %0 = stablehlo.constant dense<0.000000e+00> : tensor<2x3x4xf32>
    return %0 : tensor<2x3x4xf32>
  }
}

// -----

// Rank-3 with dim = 0 (first axis): batching dims are the trailing two.
module @Rank3_DimFirst {
  func.func @main(%arg0: tensor<6x4x3xf32>, %arg1: tensor<2x4x3xi32>) -> tensor<2x4x3xf32> {
    %0 = stablehlo.composite "tenstorrent.gather" %arg0, %arg1 {composite_attributes = {dim = 0 : i64}, decomposition = @tenstorrent.gather.impl_3dfirst} : (tensor<6x4x3xf32>, tensor<2x4x3xi32>) -> tensor<2x4x3xf32>
    return %0 : tensor<2x4x3xf32>
  }
  // CHECK-LABEL: func.func private @tenstorrent.gather.impl_3dfirst
  // CHECK-NOT:  stablehlo.slice
  // CHECK:      "stablehlo.gather"
  // CHECK-SAME: collapsed_slice_dims = [0]
  // CHECK-SAME: operand_batching_dims = [1, 2]
  // CHECK-SAME: start_indices_batching_dims = [1, 2]
  // CHECK-SAME: start_index_map = [0]
  // CHECK-SAME: index_vector_dim = 3
  // CHECK-SAME: slice_sizes = array<i64: 1, 1, 1>
  func.func private @tenstorrent.gather.impl_3dfirst(%arg0: tensor<6x4x3xf32>, %arg1: tensor<2x4x3xi32>) -> tensor<2x4x3xf32> {
    %0 = stablehlo.constant dense<0.000000e+00> : tensor<2x4x3xf32>
    return %0 : tensor<2x4x3xf32>
  }
}

// -----

// Rank-3 with dim = 1 (middle axis): batching dims straddle the collapsed
// axis.
module @Rank3_DimMiddle {
  func.func @main(%arg0: tensor<2x7x4xf32>, %arg1: tensor<2x3x4xi32>) -> tensor<2x3x4xf32> {
    %0 = stablehlo.composite "tenstorrent.gather" %arg0, %arg1 {composite_attributes = {dim = 1 : i64}, decomposition = @tenstorrent.gather.impl_3dmid} : (tensor<2x7x4xf32>, tensor<2x3x4xi32>) -> tensor<2x3x4xf32>
    return %0 : tensor<2x3x4xf32>
  }
  // CHECK-LABEL: func.func private @tenstorrent.gather.impl_3dmid
  // CHECK-NOT:  stablehlo.slice
  // CHECK:      "stablehlo.gather"
  // CHECK-SAME: collapsed_slice_dims = [1]
  // CHECK-SAME: operand_batching_dims = [0, 2]
  // CHECK-SAME: start_indices_batching_dims = [0, 2]
  // CHECK-SAME: start_index_map = [1]
  // CHECK-SAME: index_vector_dim = 3
  func.func private @tenstorrent.gather.impl_3dmid(%arg0: tensor<2x7x4xf32>, %arg1: tensor<2x3x4xi32>) -> tensor<2x3x4xf32> {
    %0 = stablehlo.constant dense<0.000000e+00> : tensor<2x3x4xf32>
    return %0 : tensor<2x3x4xf32>
  }
}

// -----

// Rank-3 with dim = 2 (last axis): batching dims are the leading two.
module @Rank3_DimLast {
  func.func @main(%arg0: tensor<2x4x6xf32>, %arg1: tensor<2x4x3xi32>) -> tensor<2x4x3xf32> {
    %0 = stablehlo.composite "tenstorrent.gather" %arg0, %arg1 {composite_attributes = {dim = 2 : i64}, decomposition = @tenstorrent.gather.impl_3dlast} : (tensor<2x4x6xf32>, tensor<2x4x3xi32>) -> tensor<2x4x3xf32>
    return %0 : tensor<2x4x3xf32>
  }
  // CHECK-LABEL: func.func private @tenstorrent.gather.impl_3dlast
  // CHECK-NOT:  stablehlo.slice
  // CHECK:      stablehlo.reshape {{.*}} : (tensor<2x4x3xi32>) -> tensor<2x4x3x1xi32>
  // CHECK:      "stablehlo.gather"
  // CHECK-SAME: collapsed_slice_dims = [2]
  // CHECK-SAME: operand_batching_dims = [0, 1]
  // CHECK-SAME: start_indices_batching_dims = [0, 1]
  // CHECK-SAME: start_index_map = [2]
  // CHECK-SAME: index_vector_dim = 3
  // CHECK-SAME: slice_sizes = array<i64: 1, 1, 1>
  func.func private @tenstorrent.gather.impl_3dlast(%arg0: tensor<2x4x6xf32>, %arg1: tensor<2x4x3xi32>) -> tensor<2x4x3xf32> {
    %0 = stablehlo.constant dense<0.000000e+00> : tensor<2x4x3xf32>
    return %0 : tensor<2x4x3xf32>
  }
}

// -----

// Rank-3 mixed: two non-dim axes both need trimming but by different
// amounts. The dim axis itself has matching sizes (5 == 5), which must
// not trigger a slice on that axis.
module @Rank3_MixedNonDimAxisSizes {
  func.func @main(%arg0: tensor<5x8x7xf32>, %arg1: tensor<5x6x3xi32>) -> tensor<5x6x3xf32> {
    %0 = stablehlo.composite "tenstorrent.gather" %arg0, %arg1 {composite_attributes = {dim = 0 : i64}, decomposition = @tenstorrent.gather.impl_mixed} : (tensor<5x8x7xf32>, tensor<5x6x3xi32>) -> tensor<5x6x3xf32>
    return %0 : tensor<5x6x3xf32>
  }
  // CHECK-LABEL: func.func private @tenstorrent.gather.impl_mixed
  // CHECK:      %[[S:.*]] = stablehlo.slice %{{.*}} [0:5, 0:6, 0:3] : (tensor<5x8x7xf32>) -> tensor<5x6x3xf32>
  // CHECK:      stablehlo.reshape {{.*}} : (tensor<5x6x3xi32>) -> tensor<5x6x3x1xi32>
  // CHECK:      "stablehlo.gather"(%[[S]]
  // CHECK-SAME: collapsed_slice_dims = [0]
  // CHECK-SAME: operand_batching_dims = [1, 2]
  // CHECK-SAME: start_indices_batching_dims = [1, 2]
  // CHECK-SAME: (tensor<5x6x3xf32>, tensor<5x6x3x1xi32>) -> tensor<5x6x3xf32>
  func.func private @tenstorrent.gather.impl_mixed(%arg0: tensor<5x8x7xf32>, %arg1: tensor<5x6x3xi32>) -> tensor<5x6x3xf32> {
    %0 = stablehlo.constant dense<0.000000e+00> : tensor<5x6x3xf32>
    return %0 : tensor<5x6x3xf32>
  }
}

// -----

// The dim axis on input may be strictly smaller than on index
// (torch.gather allows index.size(dim) > input.size(dim) — values in
// `index` just have to be valid positions in [0, input.size(dim))). The
// rewrite must not slice on the dim axis in that case.
module @Rank2_IndexLargerOnDim {
  func.func @main(%arg0: tensor<4x3xf32>, %arg1: tensor<4x9xi32>) -> tensor<4x9xf32> {
    %0 = stablehlo.composite "tenstorrent.gather" %arg0, %arg1 {composite_attributes = {dim = 1 : i64}, decomposition = @tenstorrent.gather.impl_bigidx} : (tensor<4x3xf32>, tensor<4x9xi32>) -> tensor<4x9xf32>
    return %0 : tensor<4x9xf32>
  }
  // CHECK-LABEL: func.func private @tenstorrent.gather.impl_bigidx
  // CHECK-NOT:  stablehlo.slice
  // CHECK:      stablehlo.reshape {{.*}} : (tensor<4x9xi32>) -> tensor<4x9x1xi32>
  // CHECK:      "stablehlo.gather"
  // CHECK-SAME: collapsed_slice_dims = [1]
  // CHECK-SAME: operand_batching_dims = [0]
  // CHECK-SAME: start_indices_batching_dims = [0]
  // CHECK-SAME: start_index_map = [1]
  // CHECK-SAME: (tensor<4x3xf32>, tensor<4x9x1xi32>) -> tensor<4x9xf32>
  func.func private @tenstorrent.gather.impl_bigidx(%arg0: tensor<4x3xf32>, %arg1: tensor<4x9xi32>) -> tensor<4x9xf32> {
    %0 = stablehlo.constant dense<0.000000e+00> : tensor<4x9xf32>
    return %0 : tensor<4x9xf32>
  }
}

// -----

// Rank-4 with dim in the middle: batching dims skip the collapsed axis
// and cover every other axis.
module @Rank4_DimMiddle {
  func.func @main(%arg0: tensor<2x3x8x5xf32>, %arg1: tensor<2x3x4x5xi32>) -> tensor<2x3x4x5xf32> {
    %0 = stablehlo.composite "tenstorrent.gather" %arg0, %arg1 {composite_attributes = {dim = 2 : i64}, decomposition = @tenstorrent.gather.impl_4d} : (tensor<2x3x8x5xf32>, tensor<2x3x4x5xi32>) -> tensor<2x3x4x5xf32>
    return %0 : tensor<2x3x4x5xf32>
  }
  // CHECK-LABEL: func.func private @tenstorrent.gather.impl_4d
  // CHECK-NOT:  stablehlo.slice
  // CHECK:      stablehlo.reshape {{.*}} : (tensor<2x3x4x5xi32>) -> tensor<2x3x4x5x1xi32>
  // CHECK:      "stablehlo.gather"
  // CHECK-SAME: collapsed_slice_dims = [2]
  // CHECK-SAME: operand_batching_dims = [0, 1, 3]
  // CHECK-SAME: start_indices_batching_dims = [0, 1, 3]
  // CHECK-SAME: start_index_map = [2]
  // CHECK-SAME: index_vector_dim = 4
  // CHECK-SAME: slice_sizes = array<i64: 1, 1, 1, 1>
  func.func private @tenstorrent.gather.impl_4d(%arg0: tensor<2x3x8x5xf32>, %arg1: tensor<2x3x4x5xi32>) -> tensor<2x3x4x5xf32> {
    %0 = stablehlo.constant dense<0.000000e+00> : tensor<2x3x4x5xf32>
    return %0 : tensor<2x3x4x5xf32>
  }
}

// -----

// Rank-1: no non-dim axes exist, so operand_batching_dims and
// start_indices_batching_dims stay empty and the output is a plain
// embedding-style gather. torch.gather on rank-1 is only valid with dim=0.
module @Rank1 {
  func.func @main(%arg0: tensor<5xf32>, %arg1: tensor<2xi32>) -> tensor<2xf32> {
    %0 = stablehlo.composite "tenstorrent.gather" %arg0, %arg1 {composite_attributes = {dim = 0 : i64}, decomposition = @tenstorrent.gather.impl_rank1} : (tensor<5xf32>, tensor<2xi32>) -> tensor<2xf32>
    return %0 : tensor<2xf32>
  }
  // CHECK-LABEL: func.func private @tenstorrent.gather.impl_rank1
  // CHECK-NOT:  stablehlo.slice
  // CHECK:      stablehlo.reshape {{.*}} : (tensor<2xi32>) -> tensor<2x1xi32>
  // CHECK:      "stablehlo.gather"
  // CHECK-SAME: collapsed_slice_dims = [0]
  // CHECK-SAME: start_index_map = [0]
  // CHECK-SAME: index_vector_dim = 1
  // CHECK-SAME: slice_sizes = array<i64: 1>
  // CHECK-SAME: (tensor<5xf32>, tensor<2x1xi32>) -> tensor<2xf32>
  func.func private @tenstorrent.gather.impl_rank1(%arg0: tensor<5xf32>, %arg1: tensor<2xi32>) -> tensor<2xf32> {
    %0 = stablehlo.constant dense<0.000000e+00> : tensor<2xf32>
    return %0 : tensor<2xf32>
  }
}

// -----

// Composite with no composite_attributes at all: the rewrite falls back
// to dim = 0 and still produces a valid batching-dim gather.
module @GatherDefaultAttrs {
  func.func @main(%arg0: tensor<5x3xf32>, %arg1: tensor<2x3xi32>) -> tensor<2x3xf32> {
    %0 = stablehlo.composite "tenstorrent.gather" %arg0, %arg1 {decomposition = @tenstorrent.gather.impl_default} : (tensor<5x3xf32>, tensor<2x3xi32>) -> tensor<2x3xf32>
    return %0 : tensor<2x3xf32>
  }
  // CHECK-LABEL: func.func private @tenstorrent.gather.impl_default
  // CHECK:      "stablehlo.gather"
  // CHECK-SAME: collapsed_slice_dims = [0]
  // CHECK-SAME: operand_batching_dims = [1]
  // CHECK-SAME: start_index_map = [0]
  func.func private @tenstorrent.gather.impl_default(%arg0: tensor<5x3xf32>, %arg1: tensor<2x3xi32>) -> tensor<2x3xf32> {
    %0 = stablehlo.constant dense<0.000000e+00> : tensor<2x3xf32>
    return %0 : tensor<2x3xf32>
  }
}

// -----

// ui32 index element type is preserved through reshape and gather.
module @IndexType_Ui32 {
  func.func @main(%arg0: tensor<5x3xf32>, %arg1: tensor<2x3xui32>) -> tensor<2x3xf32> {
    %0 = stablehlo.composite "tenstorrent.gather" %arg0, %arg1 {composite_attributes = {dim = 0 : i64}, decomposition = @tenstorrent.gather.impl_ui32} : (tensor<5x3xf32>, tensor<2x3xui32>) -> tensor<2x3xf32>
    return %0 : tensor<2x3xf32>
  }
  // CHECK-LABEL: func.func private @tenstorrent.gather.impl_ui32
  // CHECK:      stablehlo.reshape {{.*}} : (tensor<2x3xui32>) -> tensor<2x3x1xui32>
  // CHECK:      "stablehlo.gather"
  // CHECK-SAME: (tensor<5x3xf32>, tensor<2x3x1xui32>) -> tensor<2x3xf32>
  func.func private @tenstorrent.gather.impl_ui32(%arg0: tensor<5x3xf32>, %arg1: tensor<2x3xui32>) -> tensor<2x3xf32> {
    %0 = stablehlo.constant dense<0.000000e+00> : tensor<2x3xf32>
    return %0 : tensor<2x3xf32>
  }
}

// -----

// ui16 index element type is also preserved.
module @IndexType_Ui16 {
  func.func @main(%arg0: tensor<5x3xf32>, %arg1: tensor<2x3xui16>) -> tensor<2x3xf32> {
    %0 = stablehlo.composite "tenstorrent.gather" %arg0, %arg1 {composite_attributes = {dim = 0 : i64}, decomposition = @tenstorrent.gather.impl_ui16} : (tensor<5x3xf32>, tensor<2x3xui16>) -> tensor<2x3xf32>
    return %0 : tensor<2x3xf32>
  }
  // CHECK-LABEL: func.func private @tenstorrent.gather.impl_ui16
  // CHECK:      stablehlo.reshape {{.*}} : (tensor<2x3xui16>) -> tensor<2x3x1xui16>
  // CHECK:      "stablehlo.gather"
  // CHECK-SAME: (tensor<5x3xf32>, tensor<2x3x1xui16>) -> tensor<2x3xf32>
  func.func private @tenstorrent.gather.impl_ui16(%arg0: tensor<5x3xf32>, %arg1: tensor<2x3xui16>) -> tensor<2x3xf32> {
    %0 = stablehlo.constant dense<0.000000e+00> : tensor<2x3xf32>
    return %0 : tensor<2x3xf32>
  }
}

// -----

// bf16 input with i64 indices and a non-dim axis that must be sliced —
// the common torch-xla-produced shape for torch.gather on an embedding
// table.
module @Bf16Input_I64Index_WithSlice {
  func.func @main(%arg0: tensor<8x32xbf16>, %arg1: tensor<8x16xi64>) -> tensor<8x16xbf16> {
    %0 = stablehlo.composite "tenstorrent.gather" %arg0, %arg1 {composite_attributes = {dim = 0 : i64}, decomposition = @tenstorrent.gather.impl_bf16} : (tensor<8x32xbf16>, tensor<8x16xi64>) -> tensor<8x16xbf16>
    return %0 : tensor<8x16xbf16>
  }
  // CHECK-LABEL: func.func private @tenstorrent.gather.impl_bf16
  // CHECK:      %[[S:.*]] = stablehlo.slice %{{.*}} [0:8, 0:16] : (tensor<8x32xbf16>) -> tensor<8x16xbf16>
  // CHECK:      stablehlo.reshape {{.*}} : (tensor<8x16xi64>) -> tensor<8x16x1xi64>
  // CHECK:      "stablehlo.gather"(%[[S]]
  // CHECK-SAME: (tensor<8x16xbf16>, tensor<8x16x1xi64>) -> tensor<8x16xbf16>
  func.func private @tenstorrent.gather.impl_bf16(%arg0: tensor<8x32xbf16>, %arg1: tensor<8x16xi64>) -> tensor<8x16xbf16> {
    %0 = stablehlo.constant dense<0.000000e+00> : tensor<8x16xbf16>
    return %0 : tensor<8x16xbf16>
  }
}

// -----

// A sibling `sparse_grad` attribute under composite_attributes must be
// silently ignored — only `dim` is consumed by the rewrite.
module @GatherSparseGradAttr {
  func.func @main(%arg0: tensor<5x3xf32>, %arg1: tensor<2x3xi32>) -> tensor<2x3xf32> {
    %0 = stablehlo.composite "tenstorrent.gather" %arg0, %arg1 {composite_attributes = {dim = 0 : i64, sparse_grad = false}, decomposition = @tenstorrent.gather.impl_sparsegrad} : (tensor<5x3xf32>, tensor<2x3xi32>) -> tensor<2x3xf32>
    return %0 : tensor<2x3xf32>
  }
  // CHECK-LABEL: func.func private @tenstorrent.gather.impl_sparsegrad
  // CHECK:      "stablehlo.gather"
  // CHECK-SAME: collapsed_slice_dims = [0]
  // CHECK-SAME: operand_batching_dims = [1]
  func.func private @tenstorrent.gather.impl_sparsegrad(%arg0: tensor<5x3xf32>, %arg1: tensor<2x3xi32>) -> tensor<2x3xf32> {
    %0 = stablehlo.constant dense<0.000000e+00> : tensor<2x3xf32>
    return %0 : tensor<2x3xf32>
  }
}

// -----

// Function signature (name, input types, output types) must be preserved
// exactly by the rewrite — the body changes but the FuncOp around it stays.
module @PreservesFunctionSignature {
  func.func @main(%arg0: tensor<5x3xf32>, %arg1: tensor<2x3xi32>) -> tensor<2x3xf32> {
    %0 = stablehlo.composite "tenstorrent.gather" %arg0, %arg1 {composite_attributes = {dim = 0 : i64}, decomposition = @tenstorrent.gather.impl_sig} : (tensor<5x3xf32>, tensor<2x3xi32>) -> tensor<2x3xf32>
    return %0 : tensor<2x3xf32>
  }
  // CHECK:      func.func private @tenstorrent.gather.impl_sig
  // CHECK-SAME: (%{{.*}}: tensor<5x3xf32>, %{{.*}}: tensor<2x3xi32>) -> tensor<2x3xf32>
  func.func private @tenstorrent.gather.impl_sig(%arg0: tensor<5x3xf32>, %arg1: tensor<2x3xi32>) -> tensor<2x3xf32> {
    %0 = stablehlo.constant dense<0.000000e+00> : tensor<2x3xf32>
    return %0 : tensor<2x3xf32>
  }
}
