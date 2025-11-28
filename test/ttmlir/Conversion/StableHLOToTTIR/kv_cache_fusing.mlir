// RUN: ttmlir-opt --stablehlo-to-ttir-pipeline -o %t %s
// RUN: FileCheck %s --input-file=%t

// Test case with a FillCache scatter op pattern
module @scatter_fill_cache{
  func.func @fill_cache(%arg0: tensor<1x8x64x128xbf16>, %arg1: tensor<15xi64>, %arg2: tensor<1x8x15x128xbf16>) -> tensor<1x8x64x128xbf16> {
    // CHECK-NOT: ttir.scatter
    // CHECK-NOT: ttir.update_cache
    // CHECK: %[[RET:[0-9]+]] = "ttir.fill_cache"(%arg0, %arg2) <{batch_offset = 0 : i32}> : (tensor<1x8x64x128xbf16>, tensor<1x8x15x128xbf16>) -> tensor<1x8x64x128xbf16>
    // CHECK: return %[[RET]] : tensor<1x8x64x128xbf16>

    %0 = stablehlo.constant() <{value = dense<0> : tensor<1xi64>}> : () -> tensor<1xi64>
    %1 = stablehlo.constant() <{value = dense<1> : tensor<1xi64>}> : () -> tensor<1xi64>
    %2 = stablehlo.constant() <{value = dense<0> : tensor<8xi64>}> : () -> tensor<8xi64>
    %3 = stablehlo.constant() <{value = dense<1> : tensor<8xi64>}> : () -> tensor<8xi64>
    %4 = stablehlo.constant() <{value = dense<0> : tensor<128xi64>}> : () -> tensor<128xi64>
    %5 = stablehlo.constant() <{value = dense<1> : tensor<128xi64>}> : () -> tensor<128xi64>
    %6 = stablehlo.iota dim = 0 : tensor<128xi64>
    %8 = stablehlo.multiply %6, %5 : tensor<128xi64>
    %10 = stablehlo.add %8, %4 : tensor<128xi64>
    %11 = stablehlo.iota dim = 0 : tensor<8xi64>
    %13 = stablehlo.multiply %11, %3 : tensor<8xi64>
    %15 = stablehlo.add %13, %2 : tensor<8xi64>
    %16 = stablehlo.iota dim = 0 : tensor<1xi64>
    %18 = stablehlo.multiply %16, %1 : tensor<1xi64>
    %20 = stablehlo.add %18, %0 : tensor<1xi64>
    %22 = stablehlo.reshape %20 : (tensor<1xi64>) -> tensor<1x1x1x1xi64>
    %24 = stablehlo.broadcast_in_dim %22, dims = [0, 1, 2, 3] : (tensor<1x1x1x1xi64>) -> tensor<1x8x15x128xi64>
    %26 = stablehlo.reshape %24 : (tensor<1x8x15x128xi64>) -> tensor<1x8x15x128x1xi64>
    %28 = stablehlo.reshape %15 : (tensor<8xi64>) -> tensor<1x8x1x1xi64>
    %30 = stablehlo.broadcast_in_dim %28, dims = [0, 1, 2, 3] : (tensor<1x8x1x1xi64>) -> tensor<1x8x15x128xi64>
    %32 = stablehlo.reshape %30 : (tensor<1x8x15x128xi64>) -> tensor<1x8x15x128x1xi64>
    %34 = stablehlo.reshape %arg1 : (tensor<15xi64>) -> tensor<1x1x15x1xi64>
    %36 = stablehlo.broadcast_in_dim %34, dims = [0, 1, 2, 3] : (tensor<1x1x15x1xi64>) -> tensor<1x8x15x128xi64>
    %38 = stablehlo.reshape %36 : (tensor<1x8x15x128xi64>) -> tensor<1x8x15x128x1xi64>
    %40 = stablehlo.reshape %10 : (tensor<128xi64>) -> tensor<1x1x1x128xi64>
    %42 = stablehlo.broadcast_in_dim %40, dims = [0, 1, 2, 3] : (tensor<1x1x1x128xi64>) -> tensor<1x8x15x128xi64>
    %44 = stablehlo.reshape %42 : (tensor<1x8x15x128xi64>) -> tensor<1x8x15x128x1xi64>
    %46 = stablehlo.concatenate %26, %32, %38, %44, dim = 4 : (tensor<1x8x15x128x1xi64>, tensor<1x8x15x128x1xi64>, tensor<1x8x15x128x1xi64>, tensor<1x8x15x128x1xi64>) -> tensor<1x8x15x128x4xi64>
    %48 = "stablehlo.scatter"(%arg0, %46, %arg2) <{indices_are_sorted = false, scatter_dimension_numbers = #stablehlo.scatter<update_window_dims = [], inserted_window_dims = [0, 1, 2, 3], scatter_dims_to_operand_dims = [0, 1, 2, 3], index_vector_dim = 4>, unique_indices = false}> ({
    ^bb0(%arg3: tensor<bf16>, %arg4: tensor<bf16>):
      stablehlo.return %arg4 : tensor<bf16>
    }) : (tensor<1x8x64x128xbf16>, tensor<1x8x15x128x4xi64>, tensor<1x8x15x128xbf16>) -> tensor<1x8x64x128xbf16>
    return %48 : tensor<1x8x64x128xbf16>
  }
}

// Test case with a UpdateCache scatter op pattern
module @scatter_update_cache{
  func.func @update_cache(%arg0: tensor<1x8x64x128xbf16>, %arg1: tensor<1xi64>, %arg2: tensor<1x8x1x128xbf16>) -> tensor<1x8x64x128xbf16> {
    // CHECK-NOT: ttir.scatter
    // CHECK-NOT: ttir.fill_cache
    // CHECK: %[[RET:[0-9]+]] = "ttir.update_cache"(%arg0, %arg2, %arg1) <{batch_offset = 0 : i32}> : (tensor<1x8x64x128xbf16>, tensor<1x8x1x128xbf16>, tensor<1xi64>) -> tensor<1x8x64x128xbf16>
    // CHECK: return %[[RET]] : tensor<1x8x64x128xbf16>

    %0 = stablehlo.constant() <{value = dense<0> : tensor<1xi64>}> : () -> tensor<1xi64>
    %1 = stablehlo.constant() <{value = dense<1> : tensor<1xi64>}> : () -> tensor<1xi64>
    %2 = stablehlo.constant() <{value = dense<0> : tensor<8xi64>}> : () -> tensor<8xi64>
    %3 = stablehlo.constant() <{value = dense<1> : tensor<8xi64>}> : () -> tensor<8xi64>
    %4 = stablehlo.constant() <{value = dense<0> : tensor<128xi64>}> : () -> tensor<128xi64>
    %5 = stablehlo.constant() <{value = dense<1> : tensor<128xi64>}> : () -> tensor<128xi64>
    %6 = stablehlo.iota dim = 0 : tensor<128xi64>
    %8 = stablehlo.multiply %6, %5 : tensor<128xi64>
    %10 = stablehlo.add %8, %4 : tensor<128xi64>
    %11 = stablehlo.iota dim = 0 : tensor<8xi64>
    %13 = stablehlo.multiply %11, %3 : tensor<8xi64>
    %15 = stablehlo.add %13, %2 : tensor<8xi64>
    %16 = stablehlo.iota dim = 0 : tensor<1xi64>
    %18 = stablehlo.multiply %16, %1 : tensor<1xi64>
    %20 = stablehlo.add %18, %0 : tensor<1xi64>
    %22 = stablehlo.reshape %20 : (tensor<1xi64>) -> tensor<1x1x1x1xi64>
    %24 = stablehlo.broadcast_in_dim %22, dims = [0, 1, 2, 3] : (tensor<1x1x1x1xi64>) -> tensor<1x8x1x128xi64>
    %26 = stablehlo.reshape %24 : (tensor<1x8x1x128xi64>) -> tensor<1x8x1x128x1xi64>
    %28 = stablehlo.reshape %15 : (tensor<8xi64>) -> tensor<1x8x1x1xi64>
    %30 = stablehlo.broadcast_in_dim %28, dims = [0, 1, 2, 3] : (tensor<1x8x1x1xi64>) -> tensor<1x8x1x128xi64>
    %32 = stablehlo.reshape %30 : (tensor<1x8x1x128xi64>) -> tensor<1x8x1x128x1xi64>
    %34 = stablehlo.reshape %arg1 : (tensor<1xi64>) -> tensor<1x1x1x1xi64>
    %36 = stablehlo.broadcast_in_dim %34, dims = [0, 1, 2, 3] : (tensor<1x1x1x1xi64>) -> tensor<1x8x1x128xi64>
    %38 = stablehlo.reshape %36 : (tensor<1x8x1x128xi64>) -> tensor<1x8x1x128x1xi64>
    %40 = stablehlo.reshape %10 : (tensor<128xi64>) -> tensor<1x1x1x128xi64>
    %42 = stablehlo.broadcast_in_dim %40, dims = [0, 1, 2, 3] : (tensor<1x1x1x128xi64>) -> tensor<1x8x1x128xi64>
    %44 = stablehlo.reshape %42 : (tensor<1x8x1x128xi64>) -> tensor<1x8x1x128x1xi64>
    %46 = stablehlo.concatenate %26, %32, %38, %44, dim = 4 : (tensor<1x8x1x128x1xi64>, tensor<1x8x1x128x1xi64>, tensor<1x8x1x128x1xi64>, tensor<1x8x1x128x1xi64>) -> tensor<1x8x1x128x4xi64>
    %48 = "stablehlo.scatter"(%arg0, %46, %arg2) <{indices_are_sorted = false, scatter_dimension_numbers = #stablehlo.scatter<update_window_dims = [], inserted_window_dims = [0, 1, 2, 3], scatter_dims_to_operand_dims = [0, 1, 2, 3], index_vector_dim = 4>, unique_indices = false}> ({
    ^bb0(%arg3: tensor<bf16>, %arg4: tensor<bf16>):
      stablehlo.return %arg4 : tensor<bf16>
    }) : (tensor<1x8x64x128xbf16>, tensor<1x8x1x128x4xi64>, tensor<1x8x1x128xbf16>) -> tensor<1x8x64x128xbf16>
    return %48 : tensor<1x8x64x128xbf16>
  }
}

// Test case with a simple scatter op does not match fill/update cache pattern
module @scatter {
  func.func public @test_scatter(%arg0: tensor<1x3x320x320xf32>, %arg1: tensor<1x1xi64>, %arg2: tensor<1x3x32x32xf32>) -> tensor<1x3x320x320xf32> {
    // CHECK-NOT: ttir.fill_cache
    // CHECK-NOT: ttir.update_cache
    // CHECK: ttir.scatter
    %1 = "stablehlo.scatter"(%arg0, %arg1, %arg2) <{indices_are_sorted = false, scatter_dimension_numbers = #stablehlo.scatter<update_window_dims = [1, 2, 3], inserted_window_dims = [0], scatter_dims_to_operand_dims = [0], index_vector_dim = 1>, unique_indices = false}> ({
    ^bb0(%arg3: tensor<f32>, %arg4: tensor<f32>):
      stablehlo.return %arg4 : tensor<f32>
    }) : (tensor<1x3x320x320xf32>, tensor<1x1xi64>, tensor<1x3x32x32xf32>) -> tensor<1x3x320x320xf32>
    return %1 : tensor<1x3x320x320xf32>
  }
}

// Test case with a scatter op that has to track all the way up to a const arange with the wrong values. Should not fuse scatter into ttir.update_cache
module {
  func.func @main(%arg0: tensor<1x8x64x128xbf16>, %arg1: tensor<14xi64>, %arg2: tensor<1x8x14x128xbf16>) -> tensor<1x8x64x128xbf16> {
    // CHECK-NOT: stablehlo.scatter
    // CHECK-NOT: ttir.update_cache
    // CHECK: %[[RET:[0-9]+]] = "ttir.fill_cache"(%arg0, %arg2) <{batch_offset = 0 : i32}> : (tensor<1x8x64x128xbf16>, tensor<1x8x14x128xbf16>) -> tensor<1x8x64x128xbf16>
    // CHECK: return %[[RET]] : tensor<1x8x64x128xbf16>
    %1 = "stablehlo.scatter"(%arg0, %arg1, %arg2) <{indices_are_sorted = false, scatter_dimension_numbers = #stablehlo.scatter<update_window_dims = [0, 1, 3], inserted_window_dims = [2], scatter_dims_to_operand_dims = [2], index_vector_dim = 1>, unique_indices = false}> ({
    ^bb0(%arg3: tensor<bf16>, %arg4: tensor<bf16>):
      stablehlo.return %arg4 : tensor<bf16>
    }) : (tensor<1x8x64x128xbf16>, tensor<14xi64>, tensor<1x8x14x128xbf16>) -> tensor<1x8x64x128xbf16>
    return %1 : tensor<1x8x64x128xbf16>
  }
}
