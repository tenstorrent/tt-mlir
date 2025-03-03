// RUN: ttmlir-opt --ttir-to-ttnn-backend-pipeline %s | FileCheck %s
module attributes {} {
  func.func @forward(%arg0: tensor<1x3x320x320xf32>, %arg1: tensor<1x3x32x32xf32>) -> tensor<1x3x320x320xf32> {
    %0 = tensor.empty() : tensor<1x3x320x320xf32>
    %1 = tensor.empty() : tensor<1x1xi32>
    %2 = "ttir.scatter"(%arg0, %1, %arg1, %0) <{index_vector_dim = 1 : i32, indices_are_sorted = false, input_batching_dims = array<i32>, inserted_window_dims = array<i32: 0>, scatter_dims_to_operand_dims = array<i32: 0>, scatter_indices_batching_dims = array<i32>, unique_indices = false, update_window_dims = array<i32: 1, 2, 3>}> ({
    ^bb0(%arg3: tensor<1xf32>, %arg4: tensor<1xf32>):
      "ttir.yield"(%arg4) : (tensor<1xf32>) -> ()
    }) : (tensor<1x3x320x320xf32>, tensor<1x1xi32>, tensor<1x3x32x32xf32>, tensor<1x3x320x320xf32>) -> tensor<1x3x320x320xf32>
    // CHECK: %{{[0-9]+}} = "ttnn.scatter"(%arg1, %arg0) : (tensor<1x3x32x32xf32, {{.*}}>, tensor<1x3x320x320xf32, {{.*}}>) -> tensor<1x3x320x320xf32, {{.*}}>
    return %2 : tensor<1x3x320x320xf32>
    // CHECK: return %{{[0-9]+}} : tensor<1x3x320x320xf32, {{.*}}>
  }
}
