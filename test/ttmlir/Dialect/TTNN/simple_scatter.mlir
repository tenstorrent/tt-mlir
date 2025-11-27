// RUN: ttmlir-opt --ttir-to-ttnn-backend-pipeline -o %t %s
// RUN: FileCheck %s --input-file=%t

module attributes {} {

  // Test 1. ScatterOp for SUM reduction type with fp32 tensors
  func.func @forward_fp32(%arg0: tensor<1x3x320x320xf32>, %arg1: tensor<1x3x32x32xf32>) -> tensor<1x3x320x320xf32> {
    %0 = ttir.empty() : tensor<1x3x320x320xf32>
    %1 = ttir.empty() : tensor<1x1xi32>
    %2 = "ttir.scatter"(%arg0, %1, %arg1, %0) <{index_vector_dim = 1 : i32, indices_are_sorted = false, input_batching_dims = array<i32>, inserted_window_dims = array<i32: 0>, scatter_dims_to_operand_dims = array<i32: 0>, scatter_indices_batching_dims = array<i32>, unique_indices = false, update_window_dims = array<i32: 1, 2, 3>, scatter_reduce_type = #ttcore.reduce_type<sum>}> : (tensor<1x3x320x320xf32>, tensor<1x1xi32>, tensor<1x3x32x32xf32>, tensor<1x3x320x320xf32>) -> tensor<1x3x320x320xf32>
    // CHECK: %{{[0-9]+}} = "ttnn.scatter"({{.*}}) <{dim = 0 : i32, scatter_reduce_type = #ttcore.reduce_type<sum>}> : (tensor<1x3x320x320xf32, {{.*}}>, tensor<1x3x32x32xsi32, {{.*}}>, tensor<1x3x32x32xf32, {{.*}}>) -> tensor<1x3x320x320xf32, {{.*}}>
    return %2 : tensor<1x3x320x320xf32>
    // CHECK: return %{{[0-9]+}} : tensor<1x3x320x320xf32, {{.*}}>
  }

  // Test 2: ScatterOp for SUM reduction type with bf16 tensors
  func.func @forward_bf16(%arg0: tensor<169x3xbf16>, %arg1: tensor<2401x3xbf16>) -> tensor<169x3xbf16> {
    %0 = ttir.empty() : tensor<169x3xbf16>
    %1 = ttir.empty() : tensor<2401x1xi64>
    %2 = "ttir.scatter"(%arg0, %1, %arg1, %0) <{ index_vector_dim = 1 : i32, indices_are_sorted = false, input_batching_dims = array<i32>, inserted_window_dims = array<i32: 0>, scatter_dims_to_operand_dims = array<i32: 0>, scatter_indices_batching_dims = array<i32>, unique_indices = false, update_window_dims = array<i32: 1>, scatter_reduce_type = #ttcore.reduce_type<sum> }> : (tensor<169x3xbf16>, tensor<2401x1xi64>, tensor<2401x3xbf16>, tensor<169x3xbf16>)  -> tensor<169x3xbf16>

    // CHECK-LABEL: func.func @forward_bf16(
    // CHECK: "ttnn.scatter"
    // CHECK: <{dim = 0 : i32, scatter_reduce_type = #ttcore.reduce_type<sum>}>
    // CHECK: : (tensor<169x3xbf16,
    // CHECK: return
    return %2 : tensor<169x3xbf16>
  }

  // gpt-oss - multi-dimensional scatter with scatter operation broken into multiple scatter operations each handling index_shape[dim] < 256
  func.func @scatter_1(%arg0: tensor<71x32xbf16>, %arg1: tensor<71x4x2xi64>, %arg2: tensor<71x4xbf16>) -> tensor<71x32xbf16> {
    %0 = ttir.empty() : tensor<71x32xbf16>
    %1 = "ttir.scatter"(%arg0, %arg1, %arg2, %0) <{index_vector_dim = 2 : i32, indices_are_sorted = false, input_batching_dims = array<i32>, inserted_window_dims = array<i32: 0, 1>, scatter_dims_to_operand_dims = array<i32: 0, 1>, scatter_indices_batching_dims = array<i32>, unique_indices = false, update_window_dims = array<i32>, scatter_reduce_type = #ttcore.reduce_type<invalid>}> : (tensor<71x32xbf16>, tensor<71x4x2xi64>, tensor<71x4xbf16>, tensor<71x32xbf16>) -> tensor<71x32xbf16>
    // CHECK: %{{[0-9]+}} = "ttnn.scatter"({{.*}}) <{dim = 0 : i32, scatter_reduce_type = #ttcore.reduce_type<invalid>}> : (tensor<2272xbf16, {{.*}}>, tensor<256xsi32, {{.*}}>, tensor<256xbf16, {{.*}}>) -> tensor<2272xbf16, {{.*}}>
    // CHECK: %{{[0-9]+}} = "ttnn.scatter"({{.*}}) <{dim = 0 : i32, scatter_reduce_type = #ttcore.reduce_type<invalid>}> : (tensor<2272xbf16, {{.*}}>, tensor<28xsi32, {{.*}}>, tensor<28xbf16, {{.*}}>) -> tensor<2272xbf16, {{.*}}>
    return %1 : tensor<71x32xbf16>
    // CHECK: return %{{[0-9]+}} : tensor<71x32xbf16, {{.*}}>
  }
}
