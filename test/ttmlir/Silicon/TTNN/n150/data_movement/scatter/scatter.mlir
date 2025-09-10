// RUN: ttmlir-opt --ttir-to-ttnn-backend-pipeline="system-desc-path=%system_desc_path%" -o %t.mlir %s
// RUN: FileCheck %s --input-file=%t.mlir
// RUN: ttmlir-translate --ttnn-to-flatbuffer -o %t.ttnn %t.mlir

module {
  func.func @scatter(%arg0: tensor<71x32xbf16>, %arg1: tensor<71x4x2xi64>, %arg2: tensor<71x4xbf16>) -> tensor<71x32xbf16> {
    %0 = ttir.empty() : tensor<71x32xbf16>
    %1 = "ttir.scatter"(%arg0, %arg1, %arg2, %0) <{index_vector_dim = 2 : i32, indices_are_sorted = false, input_batching_dims = array<i32>, inserted_window_dims = array<i32: 0, 1>, scatter_dims_to_operand_dims = array<i32: 0, 1>, scatter_indices_batching_dims = array<i32>, unique_indices = false, update_window_dims = array<i32>}> : (tensor<71x32xbf16>, tensor<71x4x2xi64>, tensor<71x4xbf16>, tensor<71x32xbf16>) -> tensor<71x32xbf16>
    // CHECK: ttnn.slice_static
    // CHECK: ttnn.reshape
    // CHECK: ttnn.scatter
    // CHECK-SAME: dim = 0 : i32
    return %1 : tensor<71x32xbf16>
  }

  func.func @scatter_1(%arg0: tensor<1x3x320x320xf32>, %arg1: tensor<1x1xi32>, %arg2: tensor<1x3x32x32xf32>) -> tensor<1x3x320x320xf32> {
    %0 = ttir.empty() : tensor<1x3x320x320xf32>
    %2 = "ttir.scatter"(%arg0, %arg1, %arg2, %0) <{index_vector_dim = 1 : i32, indices_are_sorted = false, input_batching_dims = array<i32>, inserted_window_dims = array<i32: 0>, scatter_dims_to_operand_dims = array<i32: 0>, scatter_indices_batching_dims = array<i32>, unique_indices = false, update_window_dims = array<i32: 1, 2, 3>}> : (tensor<1x3x320x320xf32>, tensor<1x1xi32>, tensor<1x3x32x32xf32>, tensor<1x3x320x320xf32>) -> tensor<1x3x320x320xf32>
    // CHECK: "ttnn.scatter"
    return %2 : tensor<1x3x320x320xf32>
  }

  // New test case from your request
  func.func @scatter_2(%arg0: tensor<1000x32xf32>, %arg1: tensor<10x1xi64>, %arg2: tensor<10x32xf32>) -> tensor<1000x32xf32> {
    %0 = ttir.empty() : tensor<1000x32xf32>
    %1 = "ttir.scatter"(%arg0, %arg1, %arg2, %0) <{index_vector_dim = 1 : i32, indices_are_sorted = false, input_batching_dims = array<i32>, inserted_window_dims = array<i32: 0>, scatter_dims_to_operand_dims = array<i32: 0>, scatter_indices_batching_dims = array<i32>, unique_indices = false, update_window_dims = array<i32: 1>}> : (tensor<1000x32xf32>, tensor<10x1xi64>, tensor<10x32xf32>, tensor<1000x32xf32>) -> tensor<1000x32xf32>
    return %1 : tensor<1000x32xf32>
  }

  // Test case with embeddings-like dimensions (similar to document patterns)
  func.func @scatter_embeddings(%arg0: tensor<2050x768xf32>, %arg1: tensor<1x5x1xi32>, %arg2: tensor<1x5x768xf32>) -> tensor<2050x768xf32> {
    %0 = ttir.empty() : tensor<2050x768xf32>
    %1 = "ttir.scatter"(%arg0, %arg1, %arg2, %0) <{index_vector_dim = 2 : i32, indices_are_sorted = false, input_batching_dims = array<i32>, inserted_window_dims = array<i32: 0>, scatter_dims_to_operand_dims = array<i32: 0>, scatter_indices_batching_dims = array<i32>, unique_indices = false, update_window_dims = array<i32: 2>}> : (tensor<2050x768xf32>, tensor<1x5x1xi32>, tensor<1x5x768xf32>, tensor<2050x768xf32>) -> tensor<2050x768xf32>
    return %1 : tensor<2050x768xf32>
  }

  // Test case with vocabulary-like dimensions
  func.func @scatter_vocab(%arg0: tensor<50272x768xf32>, %arg1: tensor<1x5x1xi32>, %arg2: tensor<1x5x768xf32>) -> tensor<50272x768xf32> {
    %0 = ttir.empty() : tensor<50272x768xf32>
    %1 = "ttir.scatter"(%arg0, %arg1, %arg2, %0) <{index_vector_dim = 2 : i32, indices_are_sorted = false, input_batching_dims = array<i32>, inserted_window_dims = array<i32: 0>, scatter_dims_to_operand_dims = array<i32: 0>, scatter_indices_batching_dims = array<i32>, unique_indices = false, update_window_dims = array<i32: 2>}> : (tensor<50272x768xf32>, tensor<1x5x1xi32>, tensor<1x5x768xf32>, tensor<50272x768xf32>) -> tensor<50272x768xf32>
    return %1 : tensor<50272x768xf32>
  }

  // Test case with bf16 and higher batch dimensions
  func.func @scatter_bf16_large(%arg0: tensor<256008x1024xbf16>, %arg1: tensor<1x8x1xi32>, %arg2: tensor<1x8x1024xbf16>) -> tensor<256008x1024xbf16> {
    %0 = ttir.empty() : tensor<256008x1024xbf16>
    %1 = "ttir.scatter"(%arg0, %arg1, %arg2, %0) <{index_vector_dim = 2 : i32, indices_are_sorted = false, input_batching_dims = array<i32>, inserted_window_dims = array<i32: 0>, scatter_dims_to_operand_dims = array<i32: 0>, scatter_indices_batching_dims = array<i32>, unique_indices = false, update_window_dims = array<i32: 2>}> : (tensor<256008x1024xbf16>, tensor<1x8x1xi32>, tensor<1x8x1024xbf16>, tensor<256008x1024xbf16>) -> tensor<256008x1024xbf16>
    return %1 : tensor<256008x1024xbf16>
  }

  // Test case with smaller embedding dimensions
  func.func @scatter_small_embed(%arg0: tensor<1026x768xbf16>, %arg1: tensor<1x13x1xi32>, %arg2: tensor<1x13x768xbf16>) -> tensor<1026x768xbf16> {
    %0 = ttir.empty() : tensor<1026x768xbf16>
    %1 = "ttir.scatter"(%arg0, %arg1, %arg2, %0) <{index_vector_dim = 2 : i32, indices_are_sorted = false, input_batching_dims = array<i32>, inserted_window_dims = array<i32: 0>, scatter_dims_to_operand_dims = array<i32: 0>, scatter_indices_batching_dims = array<i32>, unique_indices = false, update_window_dims = array<i32: 2>}> : (tensor<1026x768xbf16>, tensor<1x13x1xi32>, tensor<1x13x768xbf16>, tensor<1026x768xbf16>) -> tensor<1026x768xbf16>
    return %1 : tensor<1026x768xbf16>
  }

  // Test case with different vocabulary size
  func.func @scatter_alt_vocab(%arg0: tensor<50265x768xbf16>, %arg1: tensor<1x13x1xi32>, %arg2: tensor<1x13x768xbf16>) -> tensor<50265x768xbf16> {
    %0 = ttir.empty() : tensor<50265x768xbf16>
    %1 = "ttir.scatter"(%arg0, %arg1, %arg2, %0) <{index_vector_dim = 2 : i32, indices_are_sorted = false, input_batching_dims = array<i32>, inserted_window_dims = array<i32: 0>, scatter_dims_to_operand_dims = array<i32: 0>, scatter_indices_batching_dims = array<i32>, unique_indices = false, update_window_dims = array<i32: 2>}> : (tensor<50265x768xbf16>, tensor<1x13x1xi32>, tensor<1x13x768xbf16>, tensor<50265x768xbf16>) -> tensor<50265x768xbf16>
    return %1 : tensor<50265x768xbf16>
  }

  // Test case with 1024-dimensional features and 4 updates
  func.func @scatter_1024d(%arg0: tensor<250880x1024xf32>, %arg1: tensor<1x4x1xi32>, %arg2: tensor<1x4x1024xf32>) -> tensor<250880x1024xf32> {
    %0 = ttir.empty() : tensor<250880x1024xf32>
    %1 = "ttir.scatter"(%arg0, %arg1, %arg2, %0) <{index_vector_dim = 2 : i32, indices_are_sorted = false, input_batching_dims = array<i32>, inserted_window_dims = array<i32: 0>, scatter_dims_to_operand_dims = array<i32: 0>, scatter_indices_batching_dims = array<i32>, unique_indices = false, update_window_dims = array<i32: 2>}> : (tensor<250880x1024xf32>, tensor<1x4x1xi32>, tensor<1x4x1024xf32>, tensor<250880x1024xf32>) -> tensor<250880x1024xf32>
    return %1 : tensor<250880x1024xf32>
  }

  // Test case with small 512x768 matrix
  func.func @scatter_512_768(%arg0: tensor<512x768xf32>, %arg1: tensor<1x5x1xi32>, %arg2: tensor<1x5x768xf32>) -> tensor<512x768xf32> {
    %0 = ttir.empty() : tensor<512x768xf32>
    %1 = "ttir.scatter"(%arg0, %arg1, %arg2, %0) <{index_vector_dim = 2 : i32, indices_are_sorted = false, input_batching_dims = array<i32>, inserted_window_dims = array<i32: 0>, scatter_dims_to_operand_dims = array<i32: 0>, scatter_indices_batching_dims = array<i32>, unique_indices = false, update_window_dims = array<i32: 2>}> : (tensor<512x768xf32>, tensor<1x5x1xi32>, tensor<1x5x768xf32>, tensor<512x768xf32>) -> tensor<512x768xf32>
    return %1 : tensor<512x768xf32>
  }

  // Test case with BERT-like vocabulary size
  func.func @scatter_bert(%arg0: tensor<30522x768xf32>, %arg1: tensor<1x5x1xi32>, %arg2: tensor<1x5x768xf32>) -> tensor<30522x768xf32> {
    %0 = ttir.empty() : tensor<30522x768xf32>
    %1 = "ttir.scatter"(%arg0, %arg1, %arg2, %0) <{index_vector_dim = 2 : i32, indices_are_sorted = false, input_batching_dims = array<i32>, inserted_window_dims = array<i32: 0>, scatter_dims_to_operand_dims = array<i32: 0>, scatter_indices_batching_dims = array<i32>, unique_indices = false, update_window_dims = array<i32: 2>}> : (tensor<30522x768xf32>, tensor<1x5x1xi32>, tensor<1x5x768xf32>, tensor<30522x768xf32>) -> tensor<30522x768xf32>
    return %1 : tensor<30522x768xf32>
  }

  // Test case with 128-dimensional features
  func.func @scatter_128d(%arg0: tensor<512x128xf32>, %arg1: tensor<1x10x1xi32>, %arg2: tensor<1x10x128xf32>) -> tensor<512x128xf32> {
    %0 = ttir.empty() : tensor<512x128xf32>
    %1 = "ttir.scatter"(%arg0, %arg1, %arg2, %0) <{index_vector_dim = 2 : i32, indices_are_sorted = false, input_batching_dims = array<i32>, inserted_window_dims = array<i32: 0>, scatter_dims_to_operand_dims = array<i32: 0>, scatter_indices_batching_dims = array<i32>, unique_indices = false, update_window_dims = array<i32: 2>}> : (tensor<512x128xf32>, tensor<1x10x1xi32>, tensor<1x10x128xf32>, tensor<512x128xf32>) -> tensor<512x128xf32>
    return %1 : tensor<512x128xf32>
  }

  // Test case with tiny 2x128 matrix
  func.func @scatter_tiny(%arg0: tensor<2x128xf32>, %arg1: tensor<1x10x1xi32>, %arg2: tensor<1x10x128xf32>) -> tensor<2x128xf32> {
    %0 = ttir.empty() : tensor<2x128xf32>
    %1 = "ttir.scatter"(%arg0, %arg1, %arg2, %0) <{index_vector_dim = 2 : i32, indices_are_sorted = false, input_batching_dims = array<i32>, inserted_window_dims = array<i32: 0>, scatter_dims_to_operand_dims = array<i32: 0>, scatter_indices_batching_dims = array<i32>, unique_indices = false, update_window_dims = array<i32: 2>}> : (tensor<2x128xf32>, tensor<1x10x1xi32>, tensor<1x10x128xf32>, tensor<2x128xf32>) -> tensor<2x128xf32>
    return %1 : tensor<2x128xf32>
  }

  // Test case with large vocabulary
  func.func @scatter_large_vocab(%arg0: tensor<30000x128xf32>, %arg1: tensor<1x10x1xi32>, %arg2: tensor<1x10x128xf32>) -> tensor<30000x128xf32> {
    %0 = ttir.empty() : tensor<30000x128xf32>
    %1 = "ttir.scatter"(%arg0, %arg1, %arg2, %0) <{index_vector_dim = 2 : i32, indices_are_sorted = false, input_batching_dims = array<i32>, inserted_window_dims = array<i32: 0>, scatter_dims_to_operand_dims = array<i32: 0>, scatter_indices_batching_dims = array<i32>, unique_indices = false, update_window_dims = array<i32: 2>}> : (tensor<30000x128xf32>, tensor<1x10x1xi32>, tensor<1x10x128xf32>, tensor<30000x128xf32>) -> tensor<30000x128xf32>
    return %1 : tensor<30000x128xf32>
  }
}
