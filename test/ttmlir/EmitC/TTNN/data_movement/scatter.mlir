// RUN: ttmlir-opt --ttir-to-ttnn-backend-pipeline="system-desc-path=%system_desc_path%" -o %t.mlir %s
// RUN: ttmlir-translate --ttnn-to-flatbuffer -o %basename_t.ttnn %t.mlir
// RUN: ttmlir-opt --ttnn-backend-to-emitc-pipeline -o %t2.mlir %t.mlir
// RUN: ttmlir-translate --mlir-to-cpp -o %basename_t.cpp %t2.mlir
func.func @scatter(%arg0: tensor<71x32xbf16>, %arg1: tensor<71x4x2xi64>, %arg2: tensor<71x4xbf16>) -> tensor<71x32xbf16> {
  %0 = ttir.empty() : tensor<71x32xbf16>
  %1 = "ttir.scatter"(%arg0, %arg1, %arg2, %0) <{index_vector_dim = 2 : i32, indices_are_sorted = false, input_batching_dims = array<i32>, inserted_window_dims = array<i32: 0, 1>, scatter_dims_to_operand_dims = array<i32: 0, 1>, scatter_indices_batching_dims = array<i32>, unique_indices = false, update_window_dims = array<i32>}> : (tensor<71x32xbf16>, tensor<71x4x2xi64>, tensor<71x4xbf16>, tensor<71x32xbf16>) -> tensor<71x32xbf16>
  return %1 : tensor<71x32xbf16>
}

func.func @scatter1(%arg0: tensor<1x3x320x320xf32>, %arg1: tensor<1x1xi32>, %arg2: tensor<1x3x32x32xf32>) -> tensor<1x3x320x320xf32> {
  %0 = ttir.empty() : tensor<1x3x320x320xf32>
  %1 = "ttir.scatter"(%arg0, %arg1, %arg2, %0) <{index_vector_dim = 1 : i32, indices_are_sorted = false, input_batching_dims = array<i32>, inserted_window_dims = array<i32: 0>, scatter_dims_to_operand_dims = array<i32: 0>, scatter_indices_batching_dims = array<i32>, unique_indices = false, update_window_dims = array<i32: 1, 2, 3>}> : (tensor<1x3x320x320xf32>, tensor<1x1xi32>, tensor<1x3x32x32xf32>, tensor<1x3x320x320xf32>) -> tensor<1x3x320x320xf32>
  return %1 : tensor<1x3x320x320xf32>
}
