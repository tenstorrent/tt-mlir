// RUN: ttmlir-opt --ttir-to-ttnn-backend-pipeline %s | FileCheck %s
module attributes {} {
  // Examples 0 to 4 are from models.
  // CHECK: ttnn.embedding
  func.func @gather_0(%operand: tensor<32000x1024xbf16>, %start_indices: tensor<1x32xi32>) -> tensor<1x32x1024xbf16> {
    %0 = ttir.empty() : tensor<1x32x1024xbf16>
    %1 = "ttir.gather"(%operand, %start_indices, %0) {
        offset_dims = array<i64: 2>,
        collapsed_slice_dims = array<i64: 0>,
        operand_batching_dims = array<i64>,
        start_indices_batching_dims = array<i64>,
        start_index_map = array<i64: 0>,
        index_vector_dim = 2 : si64,
        slice_sizes = array<i64: 1, 1024>,
        indices_are_sorted = false
    } : (tensor<32000x1024xbf16>, tensor<1x32xi32>, tensor<1x32x1024xbf16>) -> tensor<1x32x1024xbf16>
    return %1 : tensor<1x32x1024xbf16>
  }

  func.func @gather_1(%operand: tensor<448x384xbf16>, %start_indices: tensor<1x2x1xi32>) -> tensor<1x2x384xbf16> {
    %0 = ttir.empty() : tensor<1x2x384xbf16>
    %1 = "ttir.gather"(%operand, %start_indices, %0) <{
        offset_dims = array<i64: 2>,
        collapsed_slice_dims = array<i64: 0>,
        operand_batching_dims = array<i64>,
        start_indices_batching_dims = array<i64>,
        start_index_map = array<i64: 0>,
        index_vector_dim = 2 : si64,
        slice_sizes = array<i64: 1, 384>,
        indices_are_sorted = false
      }> : (tensor<448x384xbf16>, tensor<1x2x1xi32>, tensor<1x2x384xbf16>) -> tensor<1x2x384xbf16>
    return %1 : tensor<1x2x384xbf16>
  }

  func.func @gather_2(%operand: tensor<51864x384xbf16>, %start_indices: tensor<1x2xi32>) -> tensor<1x2x384xbf16> {
    %0 = ttir.empty() : tensor<1x2x384xbf16>
    %1 = "ttir.gather"(%operand, %start_indices, %0) <{
        offset_dims = array<i64: 2>,
        collapsed_slice_dims = array<i64: 0>,
        operand_batching_dims = array<i64>,
        start_indices_batching_dims = array<i64>,
        start_index_map = array<i64: 0>,
        index_vector_dim = 2 : si64,
        slice_sizes = array<i64: 1, 384>,
        indices_are_sorted = false
      }> : (tensor<51864x384xbf16>, tensor<1x2xi32>, tensor<1x2x384xbf16>) -> tensor<1x2x384xbf16>
    return %1 : tensor<1x2x384xbf16>
  }

  func.func @gather_3(%operand: tensor<732x12xf32>, %start_indices: tensor<38809x1xi32>) -> tensor<38809x12xf32> {
    %0 = ttir.empty() : tensor<38809x12xf32>
    %1 = "ttir.gather"(%operand, %start_indices, %0) <{
        offset_dims = array<i64: 1>,
        collapsed_slice_dims = array<i64: 0>,
        operand_batching_dims = array<i64>,
        start_indices_batching_dims = array<i64>,
        start_index_map = array<i64: 0>,
        index_vector_dim = 1 : si64,
        slice_sizes = array<i64: 1, 12>,
        indices_are_sorted = false
      }> : (tensor<732x12xf32>, tensor<38809x1xi32>, tensor<38809x12xf32>) -> tensor<38809x12xf32>
    return %1 : tensor<38809x12xf32>
  }

  func.func @gather_4(%operand: tensor<2048x1x200xf32>, %start_indices: tensor<1x2x1xi32>) -> tensor<1x2x1x200xf32> {
    %0 = ttir.empty() : tensor<1x2x1x200xf32>
    %1 = "ttir.gather"(%operand, %start_indices, %0) <{
        offset_dims = array<i64: 2,3>,
        collapsed_slice_dims = array<i64: 0>,
        operand_batching_dims = array<i64>,
        start_indices_batching_dims = array<i64>,
        start_index_map = array<i64: 0>,
        index_vector_dim = 2 : si64,
        slice_sizes = array<i64: 1, 1, 200>,
        indices_are_sorted = false
      }> : (tensor<2048x1x200xf32>, tensor<1x2x1xi32>, tensor<1x2x1x200xf32>) -> tensor<1x2x1x200xf32>
    return %1 : tensor<1x2x1x200xf32>
  }

  // Examples 5 to 10 test different rank combinations for input, start indices and output.
  func.func @gather_5(%operand: tensor<6xbf16>, %start_indices: tensor<1xi32>) -> (tensor<1xbf16> {jax.result_info = "result"}) {
    %0 = ttir.empty() : tensor<1xbf16>
    %1 = "ttir.gather"(%operand, %start_indices, %0) <{
        collapsed_slice_dims = array<i64>,
        index_vector_dim = 0 : si64,
        indices_are_sorted = false,
        offset_dims = array<i64: 0>,
        operand_batching_dims = array<i64>,
        slice_sizes = array<i64: 1>,
        start_index_map = array<i64: 0>,
        start_indices_batching_dims = array<i64>
      }> : (tensor<6xbf16>, tensor<1xi32>, tensor<1xbf16>) -> tensor<1xbf16>
    return %1 : tensor<1xbf16>
  }

  // In StableHLO result will be just tensor<bf16>
  func.func @gather_6(%operand: tensor<1xbf16>, %start_indices: tensor<1xi32>) -> (tensor<1xbf16> {jax.result_info = "result"}) {
    %0 = ttir.empty() : tensor<1xbf16>
    %1 = "ttir.gather"(%operand, %start_indices, %0) <{
        collapsed_slice_dims = array<i64: 0>,
        index_vector_dim = 0 : si64,
        indices_are_sorted = false,
        offset_dims = array<i64>,
        operand_batching_dims = array<i64>,
        slice_sizes = array<i64: 1>,
        start_index_map = array<i64: 0>,
        start_indices_batching_dims = array<i64>
      }> : (tensor<1xbf16>, tensor<1xi32>, tensor<1xbf16>) -> tensor<1xbf16>
    return %1 : tensor<1xbf16>
  }

  // In StableHLO result will be just tensor<bf16>
  func.func @gather_7(%operand: tensor<6xbf16>, %start_indices: tensor<1xi32>) -> (tensor<1xbf16> {jax.result_info = "result"}) {
    %0 = ttir.empty() : tensor<1xbf16>
    %1 = "ttir.gather"(%operand, %start_indices, %0) <{
        collapsed_slice_dims = array<i64: 0>,
        index_vector_dim = 0 : si64,
        indices_are_sorted = false,
        offset_dims = array<i64>,
        operand_batching_dims = array<i64>,
        slice_sizes = array<i64: 1>,
        start_index_map = array<i64: 0>,
        start_indices_batching_dims = array<i64>
      }> : (tensor<6xbf16>, tensor<1xi32>, tensor<1xbf16>) -> tensor<1xbf16>
    return %1 : tensor<1xbf16>
  }

  func.func @gather_8(%operand: tensor<6x4xbf16>, %start_indices: tensor<3x1xi32>) -> tensor<3x4xbf16> {
    %0 = ttir.empty() : tensor<3x4xbf16>
    %1 = "ttir.gather"(%operand, %start_indices, %0) <{
        collapsed_slice_dims = array<i64: 0>,
        index_vector_dim = 1 : si64,
        indices_are_sorted = false,
        offset_dims = array<i64: 1>,
        operand_batching_dims = array<i64>,
        slice_sizes = array<i64: 1, 4>,
        start_index_map = array<i64: 0>,
        start_indices_batching_dims = array<i64>
      }> : (tensor<6x4xbf16>, tensor<3x1xi32>, tensor<3x4xbf16>) -> tensor<3x4xbf16>
    return %1 : tensor<3x4xbf16>
  }

  func.func @gather_9(%operand: tensor<6x4xbf16>, %start_indices: tensor<3x1xi32>) -> tensor<3x1x4xbf16> {
    %0 = ttir.empty() : tensor<3x1x4xbf16>
    %1 = "ttir.gather"(%operand, %start_indices, %0) <{
        collapsed_slice_dims = array<i64: 0>,
        index_vector_dim = 2 : si64,
        indices_are_sorted = false,
        offset_dims = array<i64: 2>,
        operand_batching_dims = array<i64>,
        slice_sizes = array<i64: 1, 4>,
        start_index_map = array<i64: 0>,
        start_indices_batching_dims = array<i64>
      }> : (tensor<6x4xbf16>, tensor<3x1xi32>, tensor<3x1x4xbf16>) -> tensor<3x1x4xbf16>
    return %1 : tensor<3x1x4xbf16>
  }

  func.func @gather_10(%operand: tensor<6x4xbf16>, %start_indices: tensor<1x1xi32>) -> tensor<4x1xbf16> {
    %0 = ttir.empty() : tensor<4x1xbf16>
    %1 = "ttir.gather"(%operand, %start_indices, %0) <{
        collapsed_slice_dims = array<i64: 0>,
        index_vector_dim = 0 : si64,
        indices_are_sorted = false,
        offset_dims = array<i64: 0>,
        operand_batching_dims = array<i64>,
        slice_sizes = array<i64: 1, 4>,
        start_index_map = array<i64: 0>,
        start_indices_batching_dims = array<i64>
      }> : (tensor<6x4xbf16>, tensor<1x1xi32>, tensor<4x1xbf16>) -> tensor<4x1xbf16>
    return %1 : tensor<4x1xbf16>
  }
}
