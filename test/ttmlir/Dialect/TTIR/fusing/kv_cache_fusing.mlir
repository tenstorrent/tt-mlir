// RUN: ttmlir-opt --stablehlo-to-ttir-pipeline -o %t %s
// RUN: FileCheck %s --input-file=%t

// Test case with a FillCache scatter op pattern
module @scatter_fill_cache{
  func.func @fill_cache(%arg0: tensor<1x8x64x128xbf16>, %arg1: tensor<15xi64>, %arg2: tensor<1x8x15x128xbf16>) -> tensor<1x8x64x128xbf16> {
    // CHECK-NOT: ttir.scatter
    // CHECK: %[[RET:[0-9]+]] = "ttir.fill_cache"(%arg0, %arg2) <{batch_offset = 0 : i32}> : (tensor<1x8x64x128xbf16>, tensor<1x8x15x128xbf16>) -> tensor<1x8x64x128xbf16>
    // CHECK: return %[[RET]] : tensor<1x8x64x128xbf16>
    %0 = "stablehlo.scatter"(%arg0, %arg1, %arg2) <{indices_are_sorted = false, scatter_dimension_numbers = #stablehlo.scatter<update_window_dims = [0, 1, 3], inserted_window_dims = [2], scatter_dims_to_operand_dims = [2], index_vector_dim = 1>, unique_indices = false}> ({
    ^bb0(%arg3: tensor<bf16>, %arg4: tensor<bf16>):
      stablehlo.return %arg4 : tensor<bf16>
    }) : (tensor<1x8x64x128xbf16>, tensor<15xi64>, tensor<1x8x15x128xbf16>) -> tensor<1x8x64x128xbf16>
    return %0 : tensor<1x8x64x128xbf16>
  }
}

// Test case with a UpdateCache scatter op pattern
module @scatter_update_cache{
  func.func @update_cache(%arg0: tensor<1x8x64x128xbf16>, %arg1: tensor<1xi64>, %arg2: tensor<1x8x1x128xbf16>) -> tensor<1x8x64x128xbf16> {
    // CHECK-NOT: ttir.scatter
    // CHECK: %[[RET:[0-9]+]] = "ttir.update_cache"(%arg0, %arg2, %arg1) <{batch_offset = 0 : i32}> : (tensor<1x8x64x128xbf16>, tensor<1x8x1x128xbf16>, tensor<1xi64>) -> tensor<1x8x64x128xbf16>
    // CHECK: return %[[RET]] : tensor<1x8x64x128xbf16>
    %0 = "stablehlo.scatter"(%arg0, %arg1, %arg2) <{indices_are_sorted = false, scatter_dimension_numbers = #stablehlo.scatter<update_window_dims = [0, 1, 3], inserted_window_dims = [2], scatter_dims_to_operand_dims = [2], index_vector_dim = 1>, unique_indices = false}> ({
    ^bb0(%arg3: tensor<bf16>, %arg4: tensor<bf16>):
      stablehlo.return %arg4 : tensor<bf16>
    }) : (tensor<1x8x64x128xbf16>, tensor<1xi64>, tensor<1x8x1x128xbf16>) -> tensor<1x8x64x128xbf16>
    return %0 : tensor<1x8x64x128xbf16>
  }
}

// Test case with a simple scatter op does not match fill/update cache pattern
module @scatter {
  func.func public @test_scatter(%arg0: tensor<1x3x320x320xf32>, %arg1: tensor<1x1xi64>, %arg2: tensor<1x3x32x32xf32>) -> tensor<1x3x320x320xf32> {
    // CHECK-NOT: ttir.fill_cache
    // CHECK-NOT: ttir.update_cache
    // CHECK: ttir.scatter_in_dim
    %0 = "stablehlo.scatter"(%arg0, %arg1, %arg2) <{indices_are_sorted = false, scatter_dimension_numbers = #stablehlo.scatter<update_window_dims = [1, 2, 3], inserted_window_dims = [0], scatter_dims_to_operand_dims = [0], index_vector_dim = 1>, unique_indices = false}> ({
    ^bb0(%arg3: tensor<f32>, %arg4: tensor<f32>):
      stablehlo.return %arg4 : tensor<f32>
    }) : (tensor<1x3x320x320xf32>, tensor<1x1xi64>, tensor<1x3x32x32xf32>) -> tensor<1x3x320x320xf32>
    return %0 : tensor<1x3x320x320xf32>
  }
}

// Test case with a scatter op that should be recognized as fill_cache
module {
  func.func @main(%arg0: tensor<1x8x64x128xbf16>, %arg1: tensor<14xi64>, %arg2: tensor<1x8x14x128xbf16>) -> tensor<1x8x64x128xbf16> {
    // CHECK-NOT: ttir.scatter
    // CHECK-NOT: ttir.update_cache
    // CHECK: %[[RET:[0-9]+]] = "ttir.fill_cache"(%arg0, %arg2) <{batch_offset = 0 : i32}> : (tensor<1x8x64x128xbf16>, tensor<1x8x14x128xbf16>) -> tensor<1x8x64x128xbf16>
    // CHECK: return %[[RET]] : tensor<1x8x64x128xbf16>
    %0 = "stablehlo.scatter"(%arg0, %arg1, %arg2) <{indices_are_sorted = false, scatter_dimension_numbers = #stablehlo.scatter<update_window_dims = [0, 1, 3], inserted_window_dims = [2], scatter_dims_to_operand_dims = [2], index_vector_dim = 1>, unique_indices = false}> ({
    ^bb0(%arg3: tensor<bf16>, %arg4: tensor<bf16>):
      stablehlo.return %arg4 : tensor<bf16>
    }) : (tensor<1x8x64x128xbf16>, tensor<14xi64>, tensor<1x8x14x128xbf16>) -> tensor<1x8x64x128xbf16>
    return %0 : tensor<1x8x64x128xbf16>
  }
}
