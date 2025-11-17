// RUN: ttmlir-opt --ttir-to-ttnn-backend-pipeline="system-desc-path=%system_desc_path%" -o %t.mlir %s
// RUN: FileCheck %s --input-file=%t.mlir
// RUN: ttmlir-translate --ttnn-to-flatbuffer -o %t.ttnn %t.mlir
// RUN: ttmlir-opt --ttir-to-ttnn-backend-pipeline -o %t %s
// RUN: FileCheck %s --input-file=%t
module attributes {} {
  // Examples 0 to 5 are from models.
  func.func @gather_0(%operand: tensor<32000x1024xbf16>, %start_indices: tensor<1x32xi32>) -> tensor<1x32x1024xbf16> {
    // CHECK: "ttnn.embedding"
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

  func.func @gather_5(%operand: tensor<2x7x512xf32>, %start_indices: tensor<2x2xi32>) -> (tensor<2x512xf32> {jax.result_info = "result"}) {
    %0 = ttir.empty() : tensor<2x512xf32>
    %1 = "ttir.gather"(%operand, %start_indices, %0) <{
        collapsed_slice_dims = array<i64: 0, 1>,
        index_vector_dim = 1 : si64,
        indices_are_sorted = false,
        offset_dims = array<i64: 1>,
        operand_batching_dims = array<i64>,
        slice_sizes = array<i64: 1, 1, 512>,
        start_index_map = array<i64: 0, 1>,
        start_indices_batching_dims = array<i64>
      }> : (tensor<2x7x512xf32>, tensor<2x2xi32>, tensor<2x512xf32>) -> tensor<2x512xf32>
    return %1 : tensor<2x512xf32>
  }

  // Examples 6 to 8 test different rank combinations for input, start indices and output.
  func.func @gather_6(%operand: tensor<6xbf16>, %start_indices: tensor<1xi32>) -> (tensor<1xbf16> {jax.result_info = "result"}) {
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

  func.func @gather_7(%operand: tensor<6x4xbf16>, %start_indices: tensor<3x1xi32>) -> tensor<3x4xbf16> {
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

  func.func @gather_8(%operand: tensor<6x4xbf16>, %start_indices: tensor<3x1xi32>) -> tensor<3x1x4xbf16> {
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

  // In examples 9 to 14 more than one dim is being indexed
  func.func @gather_9(%operand: tensor<3x2x3xf32>, %start_indices: tensor<1x2xi32>) -> (tensor<1x3xf32> {jax.result_info = "result"}) {
    %0 = ttir.empty() : tensor<1x3xf32>
    %1 = "ttir.gather"(%operand, %start_indices, %0) <{
        collapsed_slice_dims = array<i64: 0, 1>,
        index_vector_dim = 1 : si64,
        indices_are_sorted = false,
        offset_dims = array<i64: 1>,
        operand_batching_dims = array<i64>,
        slice_sizes = array<i64: 1, 1, 3>,
        start_index_map = array<i64: 0, 1>,
        start_indices_batching_dims = array<i64>
      }> : (tensor<3x2x3xf32>, tensor<1x2xi32>, tensor<1x3xf32>) -> tensor<1x3xf32>
    return %1 : tensor<1x3xf32>
  }

  func.func @gather_10(%operand: tensor<7x8x2xf32>, %start_indices: tensor<2x2x2xi32>) -> (tensor<2x2x2xf32> {jax.result_info = "result"}) {
    %0 = ttir.empty() : tensor<2x2x2xf32>
    %1 = "ttir.gather"(%operand, %start_indices, %0) <{
        collapsed_slice_dims = array<i64: 0, 1>,
        index_vector_dim = 2 : si64,
        indices_are_sorted = false,
        offset_dims = array<i64: 2>,
        operand_batching_dims = array<i64>,
        slice_sizes = array<i64: 1, 1, 2>,
        start_index_map = array<i64: 0, 1>,
        start_indices_batching_dims = array<i64>
      }> : (tensor<7x8x2xf32>, tensor<2x2x2xi32>, tensor<2x2x2xf32>) -> tensor<2x2x2xf32>
    return %1 : tensor<2x2x2xf32>
  }

  func.func @gather_11(%operand: tensor<18x17x2xf32>, %start_indices: tensor<3x1x3x2xi32>) -> (tensor<3x1x3x2xf32> {jax.result_info = "result"}) {
    %0 = ttir.empty() : tensor<3x1x3x2xf32>
    %1 = "ttir.gather"(%operand, %start_indices, %0) <{
        collapsed_slice_dims = array<i64: 0, 1>,
        index_vector_dim = 3 : si64,
        indices_are_sorted = false,
        offset_dims = array<i64: 3>,
        operand_batching_dims = array<i64>,
        slice_sizes = array<i64: 1, 1, 2>,
        start_index_map = array<i64: 0, 1>,
        start_indices_batching_dims = array<i64>
      }> : (tensor<18x17x2xf32>, tensor<3x1x3x2xi32>, tensor<3x1x3x2xf32>) -> tensor<3x1x3x2xf32>
    return %1 : tensor<3x1x3x2xf32>
  }

  func.func @gather_12(%operand: tensor<4x5x2x2xf32>, %start_indices: tensor<2x1x1x2xi32>) -> (tensor<2x1x1x2x2xf32> {jax.result_info = "result"}) {
    %0 = ttir.empty() : tensor<2x1x1x2x2xf32>
    %1 = "ttir.gather"(%operand, %start_indices, %0) <{
      collapsed_slice_dims = array<i64: 0, 1>,
        index_vector_dim = 3 : si64,
        indices_are_sorted = false,
        offset_dims = array<i64: 3, 4>,
        operand_batching_dims = array<i64>,
        slice_sizes = array<i64: 1, 1, 2, 2>,
        start_index_map = array<i64: 0, 1>,
        start_indices_batching_dims = array<i64>
      }> : (tensor<4x5x2x2xf32>, tensor<2x1x1x2xi32>, tensor<2x1x1x2x2xf32>) -> tensor<2x1x1x2x2xf32>
    return %1 : tensor<2x1x1x2x2xf32>
  }

  func.func @gather_13(%operand: tensor<1x7x2xbf16>, %start_indices: tensor<1x2xi32>) -> (tensor<1x2xbf16> {jax.result_info = "result"}) {
    %0 = ttir.empty() : tensor<1x2xbf16>
    %1 = "ttir.gather"(%operand, %start_indices, %0) <{
        collapsed_slice_dims = array<i64: 0, 1>,
        index_vector_dim = 1 : si64,
        indices_are_sorted = false,
        offset_dims = array<i64: 1>,
        operand_batching_dims = array<i64>,
        slice_sizes = array<i64: 1, 1, 2>,
        start_index_map = array<i64: 0, 1>,
        start_indices_batching_dims = array<i64>
      }> : (tensor<1x7x2xbf16>, tensor<1x2xi32>, tensor<1x2xbf16>) -> tensor<1x2xbf16>
    return %1 : tensor<1x2xbf16>
  }

  func.func @gather_14(%operand: tensor<3x2x3x4xf32>, %start_indices: tensor<1x3xi32>) -> (tensor<1x4xf32> {jax.result_info = "result"}) {
    %0 = ttir.empty() : tensor<1x4xf32>
    %1 = "ttir.gather"(%operand, %start_indices, %0) <{
        collapsed_slice_dims = array<i64: 0, 1, 2>,
        index_vector_dim = 1 : si64,
        indices_are_sorted = false,
        offset_dims = array<i64: 1>,
        operand_batching_dims = array<i64>,
        slice_sizes = array<i64: 1, 1, 1, 4>,
        start_index_map = array<i64: 0, 1, 2>,
        start_indices_batching_dims = array<i64>
      }> : (tensor<3x2x3x4xf32>, tensor<1x3xi32>, tensor<1x4xf32>) -> tensor<1x4xf32>
    return %1 : tensor<1x4xf32>
  }

  // In examples 15 and 16 permute ops are needed
  func.func @gather_15(%operand: tensor<1x2x3x5xf32>, %start_indices: tensor<4x1xi32>) -> (tensor<1x2x3x4xf32> {jax.result_info = "result"}) {
    %0 = ttir.empty() : tensor<1x2x3x4xf32>
    %1 = "ttir.gather"(%operand, %start_indices, %0) <{
        collapsed_slice_dims = array<i64: 3>,
        index_vector_dim = 1 : si64,
        indices_are_sorted = false,
        offset_dims = array<i64: 0, 1, 2>,
        operand_batching_dims = array<i64>,
        slice_sizes = array<i64: 1, 2, 3, 1>,
        start_index_map = array<i64: 3>,
        start_indices_batching_dims = array<i64>
      }> : (tensor<1x2x3x5xf32>, tensor<4x1xi32>, tensor<1x2x3x4xf32>) -> tensor<1x2x3x4xf32>
    return %1 : tensor<1x2x3x4xf32>
  }

  func.func @gather_16(%operand: tensor<1x2x5x7xf32>, %start_indices: tensor<3x2xi32>) -> (tensor<1x2x3xf32> {jax.result_info = "result"}) {
    %0 = ttir.empty() : tensor<1x2x3xf32>
    %1 = "ttir.gather"(%operand, %start_indices, %0) <{
        collapsed_slice_dims = array<i64: 2, 3>,
        index_vector_dim = 1 : si64,
        indices_are_sorted = false,
        offset_dims = array<i64: 0, 1>,
        operand_batching_dims = array<i64>,
        slice_sizes = array<i64: 1, 2, 1, 1>,
        start_index_map = array<i64: 2, 3>,
        start_indices_batching_dims = array<i64>
      }> : (tensor<1x2x5x7xf32>, tensor<3x2xi32>, tensor<1x2x3xf32>) -> tensor<1x2x3xf32>
    return %1 : tensor<1x2x3xf32>
  }
}
