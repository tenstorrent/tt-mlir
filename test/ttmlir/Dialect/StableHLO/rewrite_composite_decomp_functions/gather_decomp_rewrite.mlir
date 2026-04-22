// REQUIRES: stablehlo
// RUN: ttmlir-opt -split-input-file --rewrite-composite-decomp-functions -o %t %s
// RUN: FileCheck %s --input-file=%t

// Decomposition body for "tenstorrent.gather" with dim = 0 is rewritten to
// stablehlo.reshape + stablehlo.gather. Batching dims are all input axes
// except `dim`, and the reshaped start_indices gain a trailing size-1 dim.
module @GatherDim0 {
  func.func @main(%arg0: tensor<5x3xf32>, %arg1: tensor<2x3xi32>) -> tensor<2x3xf32> {
    %0 = stablehlo.composite "tenstorrent.gather" %arg0, %arg1 {composite_attributes = {dim = 0 : i64}, decomposition = @tenstorrent.gather.impl_dim0} : (tensor<5x3xf32>, tensor<2x3xi32>) -> tensor<2x3xf32>
    return %0 : tensor<2x3xf32>
  }
  // CHECK-LABEL: func.func private @tenstorrent.gather.impl_dim0
  // CHECK-SAME: (%[[IN:.*]]: tensor<5x3xf32>, %[[IDX:.*]]: tensor<2x3xi32>) -> tensor<2x3xf32>
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
  func.func private @tenstorrent.gather.impl_dim0(%arg0: tensor<5x3xf32>, %arg1: tensor<2x3xi32>) -> tensor<2x3xf32> {
    %0 = stablehlo.constant dense<0.000000e+00> : tensor<2x3xf32>
    return %0 : tensor<2x3xf32>
  }
}

// -----

// Decomposition body for "tenstorrent.gather" with dim = 1 collapses axis 1
// and batches axis 0.
module @GatherDim1 {
  func.func @main(%arg0: tensor<3x5xbf16>, %arg1: tensor<3x2xi64>) -> tensor<3x2xbf16> {
    %0 = stablehlo.composite "tenstorrent.gather" %arg0, %arg1 {composite_attributes = {dim = 1 : i64}, decomposition = @tenstorrent.gather.impl_dim1} : (tensor<3x5xbf16>, tensor<3x2xi64>) -> tensor<3x2xbf16>
    return %0 : tensor<3x2xbf16>
  }
  // CHECK-LABEL: func.func private @tenstorrent.gather.impl_dim1
  // CHECK:      stablehlo.reshape {{.*}} : (tensor<3x2xi64>) -> tensor<3x2x1xi64>
  // CHECK:      "stablehlo.gather"
  // CHECK-SAME: collapsed_slice_dims = [1]
  // CHECK-SAME: operand_batching_dims = [0]
  // CHECK-SAME: start_indices_batching_dims = [0]
  // CHECK-SAME: start_index_map = [1]
  // CHECK-SAME: index_vector_dim = 2
  // CHECK-SAME: slice_sizes = array<i64: 1, 1>
  func.func private @tenstorrent.gather.impl_dim1(%arg0: tensor<3x5xbf16>, %arg1: tensor<3x2xi64>) -> tensor<3x2xbf16> {
    %0 = stablehlo.constant dense<0.000000e+00> : tensor<3x2xbf16>
    return %0 : tensor<3x2xbf16>
  }
}

// -----

// Negative dim is normalized: dim = -1 on a rank-2 input becomes dim = 1.
module @GatherNegativeDim {
  func.func @main(%arg0: tensor<4x6xf32>, %arg1: tensor<4x3xi32>) -> tensor<4x3xf32> {
    %0 = stablehlo.composite "tenstorrent.gather" %arg0, %arg1 {composite_attributes = {dim = -1 : i64}, decomposition = @tenstorrent.gather.impl_neg} : (tensor<4x6xf32>, tensor<4x3xi32>) -> tensor<4x3xf32>
    return %0 : tensor<4x3xf32>
  }
  // CHECK-LABEL: func.func private @tenstorrent.gather.impl_neg
  // CHECK:      "stablehlo.gather"
  // CHECK-SAME: collapsed_slice_dims = [1]
  // CHECK-SAME: operand_batching_dims = [0]
  // CHECK-SAME: start_indices_batching_dims = [0]
  // CHECK-SAME: start_index_map = [1]
  // CHECK-SAME: index_vector_dim = 2
  func.func private @tenstorrent.gather.impl_neg(%arg0: tensor<4x6xf32>, %arg1: tensor<4x3xi32>) -> tensor<4x3xf32> {
    %0 = stablehlo.constant dense<0.000000e+00> : tensor<4x3xf32>
    return %0 : tensor<4x3xf32>
  }
}

// -----

// 3D input: batching dims are [0, 1] and index_vector_dim moves to rank (= 3).
module @Gather3D {
  func.func @main(%arg0: tensor<2x4x6xf32>, %arg1: tensor<2x4x3xi32>) -> tensor<2x4x3xf32> {
    %0 = stablehlo.composite "tenstorrent.gather" %arg0, %arg1 {composite_attributes = {dim = 2 : i64}, decomposition = @tenstorrent.gather.impl_3d} : (tensor<2x4x6xf32>, tensor<2x4x3xi32>) -> tensor<2x4x3xf32>
    return %0 : tensor<2x4x3xf32>
  }
  // CHECK-LABEL: func.func private @tenstorrent.gather.impl_3d
  // CHECK:      stablehlo.reshape {{.*}} : (tensor<2x4x3xi32>) -> tensor<2x4x3x1xi32>
  // CHECK:      "stablehlo.gather"
  // CHECK-SAME: collapsed_slice_dims = [2]
  // CHECK-SAME: operand_batching_dims = [0, 1]
  // CHECK-SAME: start_indices_batching_dims = [0, 1]
  // CHECK-SAME: start_index_map = [2]
  // CHECK-SAME: index_vector_dim = 3
  // CHECK-SAME: slice_sizes = array<i64: 1, 1, 1>
  func.func private @tenstorrent.gather.impl_3d(%arg0: tensor<2x4x6xf32>, %arg1: tensor<2x4x3xi32>) -> tensor<2x4x3xf32> {
    %0 = stablehlo.constant dense<0.000000e+00> : tensor<2x4x3xf32>
    return %0 : tensor<2x4x3xf32>
  }
}

// -----

// Composite op with no `composite_attributes` falls back to the default
// dim = 0 and still rewrites the decomposition body.
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

// Unsigned integer indices are preserved through the reshape and gather.
module @GatherUi32Index {
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
