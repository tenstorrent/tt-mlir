// REQUIRES: stablehlo
// RUN: ttmlir-opt --stablehlo-pipeline="enable-aggressive-simplification=true" --split-input-file -o %t.mlir %s
// RUN: FileCheck %s --input-file=%t.mlir

module {
  // Reshape unsqueezes at index_vector_dim, should fuse.
  // CHECK-LABEL: func.func @reshape_gather_basic
  func.func @reshape_gather_basic(%arg0: tensor<1993728x80xf32>, %arg1: tensor<1836732xi64>) -> tensor<1836732x80xf32> {
    %0 = stablehlo.reshape %arg1 : (tensor<1836732xi64>) -> tensor<1836732x1xi64>
    %1 = "stablehlo.gather"(%arg0, %0) <{
      dimension_numbers = #stablehlo.gather<
        offset_dims = [1],
        collapsed_slice_dims = [0],
        start_index_map = [0],
        index_vector_dim = 1>,
      slice_sizes = array<i64: 1, 80>,
      indices_are_sorted = false
    }> : (tensor<1993728x80xf32>, tensor<1836732x1xi64>) -> tensor<1836732x80xf32>
    // CHECK-NOT: stablehlo.reshape
    // CHECK: "stablehlo.gather"(%arg0, %arg1)
    return %1 : tensor<1836732x80xf32>
  }

  // Reshape unsqueezes at index_vector_dim with a batch dim, should fuse.
  // CHECK-LABEL: func.func @reshape_gather_batched
  func.func @reshape_gather_batched(%arg0: tensor<4x1024x64xf32>, %arg1: tensor<4x512xi64>) -> tensor<4x512x64xf32> {
    %0 = stablehlo.reshape %arg1 : (tensor<4x512xi64>) -> tensor<4x512x1xi64>
    %1 = "stablehlo.gather"(%arg0, %0) <{
      dimension_numbers = #stablehlo.gather<
        offset_dims = [2],
        collapsed_slice_dims = [1],
        operand_batching_dims = [0],
        start_indices_batching_dims = [0],
        start_index_map = [1],
        index_vector_dim = 2>,
      slice_sizes = array<i64: 1, 1, 64>,
      indices_are_sorted = false
    }> : (tensor<4x1024x64xf32>, tensor<4x512x1xi64>) -> tensor<4x512x64xf32>
    // CHECK-NOT: stablehlo.reshape
    // CHECK: "stablehlo.gather"(%arg0, %arg1)
    return %1 : tensor<4x512x64xf32>
  }
  // Reshape unsqueezes at a batch dim (not index_vector_dim), should NOT fuse.
  // CHECK-LABEL: func.func @reshape_gather_unsqueeze_wrong_dim
  func.func @reshape_gather_unsqueeze_wrong_dim(%arg0: tensor<1024x64xf32>, %arg1: tensor<512xi64>) -> tensor<1x512x64xf32> {
    %0 = stablehlo.reshape %arg1 : (tensor<512xi64>) -> tensor<1x512xi64>
    %1 = "stablehlo.gather"(%arg0, %0) <{
      dimension_numbers = #stablehlo.gather<
        offset_dims = [2],
        collapsed_slice_dims = [0],
        start_index_map = [0],
        index_vector_dim = 2>,
      slice_sizes = array<i64: 1, 64>,
      indices_are_sorted = false
    }> : (tensor<1024x64xf32>, tensor<1x512xi64>) -> tensor<1x512x64xf32>
    // CHECK: %[[IDX:.*]] = stablehlo.reshape
    // CHECK: "stablehlo.gather"(%arg0, %[[IDX]])
    return %1 : tensor<1x512x64xf32>
  }
  // Reshape unsqueezes at index_vector_dim, but index_vector_dim is not the
  // last dim (there is a batch dim after it), should NOT fuse.
  func.func @reshape_gather_index_vector_not_last(%arg0: tensor<1024x64xf32>, %arg1: tensor<4x512xi64>) -> tensor<4x512x64xf32> {
    %0 = stablehlo.reshape %arg1 : (tensor<4x512xi64>) -> tensor<4x1x512xi64>
    %1 = "stablehlo.gather"(%arg0, %0) <{
      dimension_numbers = #stablehlo.gather<
        offset_dims = [2],
        collapsed_slice_dims = [0],
        start_index_map = [0],
        index_vector_dim = 1>,
      slice_sizes = array<i64: 1, 64>,
      indices_are_sorted = false
    }> : (tensor<1024x64xf32>, tensor<4x1x512xi64>) -> tensor<4x512x64xf32>
    // CHECK: stablehlo.reshape
    // CHECK: "stablehlo.gather"
    return %1 : tensor<4x512x64xf32>
  }
}

// -----

// Sharded start indices: pattern skips fusion, reshape must remain.
module {
  sdy.mesh @mesh = <["x"=2, "y"=2]>
  func.func @reshape_gather_sharded(
      %arg0: tensor<1993728x80xf32>,
      %arg1: tensor<1836732xi64> {sdy.sharding = #sdy.sharding<@mesh, [{"x"}]>}
  ) -> tensor<1836732x80xf32> {
    %0 = stablehlo.reshape %arg1 : (tensor<1836732xi64>) -> tensor<1836732x1xi64>
    %1 = "stablehlo.gather"(%arg0, %0) <{
      dimension_numbers = #stablehlo.gather<
        offset_dims = [1],
        collapsed_slice_dims = [0],
        start_index_map = [0],
        index_vector_dim = 1>,
      slice_sizes = array<i64: 1, 80>,
      indices_are_sorted = false
    }> : (tensor<1993728x80xf32>, tensor<1836732x1xi64>) -> tensor<1836732x80xf32>
    // CHECK: stablehlo.reshape
    return %1 : tensor<1836732x80xf32>
  }
}
