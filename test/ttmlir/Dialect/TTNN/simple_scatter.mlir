// RUN: ttmlir-opt --ttir-to-ttnn-backend-pipeline %s | FileCheck %s
module attributes {} {
  func.func @forward(%arg0: tensor<1x3x320x320xbf16>, %arg1: tensor<1x3x32x32xbf16>) -> tensor<1x3x320x320xbf16> {
    %0 = ttir.empty() : tensor<1x3x320x320xbf16>
    %1 = ttir.empty() : tensor<1x1xi32>
    %2 = "ttir.scatter"(%arg0, %1, %arg1, %0) <{index_vector_dim = 1 : i32, indices_are_sorted = false, input_batching_dims = array<i32>, inserted_window_dims = array<i32: 0>, scatter_dims_to_operand_dims = array<i32: 0>, scatter_indices_batching_dims = array<i32>, unique_indices = false, update_window_dims = array<i32: 1, 2, 3>}> : (tensor<1x3x320x320xbf16>, tensor<1x1xi32>, tensor<1x3x32x32xbf16>, tensor<1x3x320x320xbf16>) -> tensor<1x3x320x320xbf16>
    // CHECK: %{{[0-9]+}} = "ttnn.scatter"(%arg1, %arg0) : (tensor<1x3x32x32xbf16, {{.*}}>, tensor<1x3x320x320xbf16, {{.*}}>) -> tensor<1x3x320x320xbf16, {{.*}}>
    return %2 : tensor<1x3x320x320xbf16>
    // CHECK: return %{{[0-9]+}} : tensor<1x3x320x320xbf16, {{.*}}>
  }
}
