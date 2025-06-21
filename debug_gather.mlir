module {
  func.func @gather_test(%arg0: tensor<32x128xf32>, %arg1: tensor<1x256xi32>) -> tensor<1x256x128xf32> {
    %0 = ttir.empty() : tensor<1x256x128xf32>
    %1 = "ttir.gather"(%arg0, %arg1, %0) {
        offset_dims = array<i64: 2>,
        collapsed_slice_dims = array<i64: 0>,
        operand_batching_dims = array<i64>,
        start_indices_batching_dims = array<i64>,
        start_index_map = array<i64: 0>,
        index_vector_dim = 1 : si64,
        slice_sizes = array<i64: 1, 128>,
        indices_are_sorted = false
    } : (tensor<32x128xf32>, tensor<1x256xi32>, tensor<1x256x128xf32>) -> tensor<1x256x128xf32>
    return %1 : tensor<1x256x128xf32>
  }
}
