// RUN: ttmlir-opt --ttir-implicit-broadcast-fold --ttir-fusing -o %t %s
// RUN: FileCheck %s --input-file=%t

// Test case with a FillCache scatter op pattern
module @scatter_fill_cache{
  func.func @fill_cache(%arg0: tensor<1x8x64x128xbf16>, %arg1: tensor<15xi64>, %arg2: tensor<1x8x15x128xbf16>) -> tensor<1x8x64x128xbf16> {
    // CHECK-NOT: ttir.multiply
    // CHECK-NOT: ttir.add
    // CHECK-NOT: ttir.broadcast
    // CHECK-NOT: ttir.reshape
    // CHECK-NOT: ttir.arange
    // CHECK-NOT: ttir.concat
    // CHECK-NOT: ttir.scatter
    // CHECK-NOT: ttir.update_cache
    // CHECK: %[[RET:[0-9]+]] = "ttir.fill_cache"(%arg0, %arg2) <{batch_offset = 0 : i32}> : (tensor<1x8x64x128xbf16>, tensor<1x8x15x128xbf16>) -> tensor<1x8x64x128xbf16>
    // CHECK: return %[[RET]] : tensor<1x8x64x128xbf16>

    %0 = "ttir.constant"() <{value = dense<0> : tensor<1xi64>}> : () -> tensor<1xi64>
    %1 = "ttir.constant"() <{value = dense<1> : tensor<1xi64>}> : () -> tensor<1xi64>
    %2 = "ttir.constant"() <{value = dense<0> : tensor<8xi64>}> : () -> tensor<8xi64>
    %3 = "ttir.constant"() <{value = dense<1> : tensor<8xi64>}> : () -> tensor<8xi64>
    %4 = "ttir.constant"() <{value = dense<0> : tensor<128xi64>}> : () -> tensor<128xi64>
    %5 = "ttir.constant"() <{value = dense<1> : tensor<128xi64>}> : () -> tensor<128xi64>
    %6 = "ttir.arange"() <{arange_dimension = 0 : i64, end = 128 : si64, start = 0 : si64, step = 1 : si64}> : () -> tensor<128xi64>
    %7 = ttir.empty() : tensor<128xi64>
    %8 = "ttir.multiply"(%6, %5, %7) : (tensor<128xi64>, tensor<128xi64>, tensor<128xi64>) -> tensor<128xi64>
    %9 = ttir.empty() : tensor<128xi64>
    %10 = "ttir.add"(%8, %4, %9) : (tensor<128xi64>, tensor<128xi64>, tensor<128xi64>) -> tensor<128xi64>
    %11 = "ttir.arange"() <{arange_dimension = 0 : i64, end = 8 : si64, start = 0 : si64, step = 1 : si64}> : () -> tensor<8xi64>
    %12 = ttir.empty() : tensor<8xi64>
    %13 = "ttir.multiply"(%11, %3, %12) : (tensor<8xi64>, tensor<8xi64>, tensor<8xi64>) -> tensor<8xi64>
    %14 = ttir.empty() : tensor<8xi64>
    %15 = "ttir.add"(%13, %2, %14) : (tensor<8xi64>, tensor<8xi64>, tensor<8xi64>) -> tensor<8xi64>
    %16 = "ttir.arange"() <{arange_dimension = 0 : i64, end = 1 : si64, start = 0 : si64, step = 1 : si64}> : () -> tensor<1xi64>
    %17 = ttir.empty() : tensor<1xi64>
    %18 = "ttir.multiply"(%16, %1, %17) : (tensor<1xi64>, tensor<1xi64>, tensor<1xi64>) -> tensor<1xi64>
    %19 = ttir.empty() : tensor<1xi64>
    %20 = "ttir.add"(%18, %0, %19) : (tensor<1xi64>, tensor<1xi64>, tensor<1xi64>) -> tensor<1xi64>
    %21 = ttir.empty() : tensor<1x1x1x1xi64>
    %22 = "ttir.reshape"(%20, %21) <{shape = [1 : i32, 1 : i32, 1 : i32, 1 : i32]}> : (tensor<1xi64>, tensor<1x1x1x1xi64>) -> tensor<1x1x1x1xi64>
    %23 = ttir.empty() : tensor<1x8x15x128xi64>
    %24 = "ttir.broadcast"(%22, %23) <{broadcast_dimensions = array<i64: 1, 8, 15, 128>}> : (tensor<1x1x1x1xi64>, tensor<1x8x15x128xi64>) -> tensor<1x8x15x128xi64>
    %25 = ttir.empty() : tensor<1x8x15x128x1xi64>
    %26 = "ttir.reshape"(%24, %25) <{shape = [1 : i32, 8 : i32, 15 : i32, 128 : i32, 1 : i32]}> : (tensor<1x8x15x128xi64>, tensor<1x8x15x128x1xi64>) -> tensor<1x8x15x128x1xi64>
    %27 = ttir.empty() : tensor<1x8x1x1xi64>
    %28 = "ttir.reshape"(%15, %27) <{shape = [1 : i32, 8 : i32, 1 : i32, 1 : i32]}> : (tensor<8xi64>, tensor<1x8x1x1xi64>) -> tensor<1x8x1x1xi64>
    %29 = ttir.empty() : tensor<1x8x15x128xi64>
    %30 = "ttir.broadcast"(%28, %29) <{broadcast_dimensions = array<i64: 1, 1, 15, 128>}> : (tensor<1x8x1x1xi64>, tensor<1x8x15x128xi64>) -> tensor<1x8x15x128xi64>
    %31 = ttir.empty() : tensor<1x8x15x128x1xi64>
    %32 = "ttir.reshape"(%30, %31) <{shape = [1 : i32, 8 : i32, 15 : i32, 128 : i32, 1 : i32]}> : (tensor<1x8x15x128xi64>, tensor<1x8x15x128x1xi64>) -> tensor<1x8x15x128x1xi64>
    %33 = ttir.empty() : tensor<1x1x15x1xi64>
    %34 = "ttir.reshape"(%arg1, %33) <{shape = [1 : i32, 1 : i32, 15 : i32, 1 : i32]}> : (tensor<15xi64>, tensor<1x1x15x1xi64>) -> tensor<1x1x15x1xi64>
    %35 = ttir.empty() : tensor<1x8x15x128xi64>
    %36 = "ttir.broadcast"(%34, %35) <{broadcast_dimensions = array<i64: 1, 8, 1, 128>}> : (tensor<1x1x15x1xi64>, tensor<1x8x15x128xi64>) -> tensor<1x8x15x128xi64>
    %37 = ttir.empty() : tensor<1x8x15x128x1xi64>
    %38 = "ttir.reshape"(%36, %37) <{shape = [1 : i32, 8 : i32, 15 : i32, 128 : i32, 1 : i32]}> : (tensor<1x8x15x128xi64>, tensor<1x8x15x128x1xi64>) -> tensor<1x8x15x128x1xi64>
    %39 = ttir.empty() : tensor<1x1x1x128xi64>
    %40 = "ttir.reshape"(%10, %39) <{shape = [1 : i32, 1 : i32, 1 : i32, 128 : i32]}> : (tensor<128xi64>, tensor<1x1x1x128xi64>) -> tensor<1x1x1x128xi64>
    %41 = ttir.empty() : tensor<1x8x15x128xi64>
    %42 = "ttir.broadcast"(%40, %41) <{broadcast_dimensions = array<i64: 1, 8, 15, 1>}> : (tensor<1x1x1x128xi64>, tensor<1x8x15x128xi64>) -> tensor<1x8x15x128xi64>
    %43 = ttir.empty() : tensor<1x8x15x128x1xi64>
    %44 = "ttir.reshape"(%42, %43) <{shape = [1 : i32, 8 : i32, 15 : i32, 128 : i32, 1 : i32]}> : (tensor<1x8x15x128xi64>, tensor<1x8x15x128x1xi64>) -> tensor<1x8x15x128x1xi64>
    %45 = ttir.empty() : tensor<1x8x15x128x4xi64>
    %46 = "ttir.concat"(%26, %32, %38, %44, %45) <{dim = 4 : si32}> : (tensor<1x8x15x128x1xi64>, tensor<1x8x15x128x1xi64>, tensor<1x8x15x128x1xi64>, tensor<1x8x15x128x1xi64>, tensor<1x8x15x128x4xi64>) -> tensor<1x8x15x128x4xi64>
    %47 = ttir.empty() : tensor<1x8x64x128xbf16>
    %48 = "ttir.scatter"(%arg0, %46, %arg2, %47) <{index_vector_dim = 4 : i32, indices_are_sorted = false, input_batching_dims = array<i32>, inserted_window_dims = array<i32: 0, 1, 2, 3>, scatter_dims_to_operand_dims = array<i32: 0, 1, 2, 3>, scatter_indices_batching_dims = array<i32>, unique_indices = false, update_window_dims = array<i32>}> : (tensor<1x8x64x128xbf16>, tensor<1x8x15x128x4xi64>, tensor<1x8x15x128xbf16>, tensor<1x8x64x128xbf16>) -> tensor<1x8x64x128xbf16>
    return %48 : tensor<1x8x64x128xbf16>
  }
}

// Test case with a UpdateCache scatter op pattern
module @scatter_update_cache{
  func.func @update_cache(%arg0: tensor<1x8x64x128xbf16>, %arg1: tensor<1xi64>, %arg2: tensor<1x8x1x128xbf16>) -> tensor<1x8x64x128xbf16> {
    // CHECK-NOT: ttir.multiply
    // CHECK-NOT: ttir.add
    // CHECK-NOT: ttir.broadcast
    // CHECK-NOT: ttir.reshape
    // CHECK-NOT: ttir.arange
    // CHECK-NOT: ttir.concat
    // CHECK-NOT: ttir.scatter
    // CHECK-NOT: ttir.fill_cache
    // CHECK: %[[RET:[0-9]+]] = "ttir.update_cache"(%arg0, %arg2, %arg1) <{batch_offset = 0 : i32}> : (tensor<1x8x64x128xbf16>, tensor<1x8x1x128xbf16>, tensor<1xi64>) -> tensor<1x8x64x128xbf16>
    // CHECK: return %[[RET]] : tensor<1x8x64x128xbf16>

    %0 = "ttir.constant"() <{value = dense<0> : tensor<1xi64>}> : () -> tensor<1xi64>
    %1 = "ttir.constant"() <{value = dense<1> : tensor<1xi64>}> : () -> tensor<1xi64>
    %2 = "ttir.constant"() <{value = dense<0> : tensor<8xi64>}> : () -> tensor<8xi64>
    %3 = "ttir.constant"() <{value = dense<1> : tensor<8xi64>}> : () -> tensor<8xi64>
    %4 = "ttir.constant"() <{value = dense<0> : tensor<128xi64>}> : () -> tensor<128xi64>
    %5 = "ttir.constant"() <{value = dense<1> : tensor<128xi64>}> : () -> tensor<128xi64>
    %6 = "ttir.arange"() <{arange_dimension = 0 : i64, end = 128 : si64, start = 0 : si64, step = 1 : si64}> : () -> tensor<128xi64>
    %7 = ttir.empty() : tensor<128xi64>
    %8 = "ttir.multiply"(%6, %5, %7) : (tensor<128xi64>, tensor<128xi64>, tensor<128xi64>) -> tensor<128xi64>
    %9 = ttir.empty() : tensor<128xi64>
    %10 = "ttir.add"(%8, %4, %9) : (tensor<128xi64>, tensor<128xi64>, tensor<128xi64>) -> tensor<128xi64>
    %11 = "ttir.arange"() <{arange_dimension = 0 : i64, end = 8 : si64, start = 0 : si64, step = 1 : si64}> : () -> tensor<8xi64>
    %12 = ttir.empty() : tensor<8xi64>
    %13 = "ttir.multiply"(%11, %3, %12) : (tensor<8xi64>, tensor<8xi64>, tensor<8xi64>) -> tensor<8xi64>
    %14 = ttir.empty() : tensor<8xi64>
    %15 = "ttir.add"(%13, %2, %14) : (tensor<8xi64>, tensor<8xi64>, tensor<8xi64>) -> tensor<8xi64>
    %16 = "ttir.arange"() <{arange_dimension = 0 : i64, end = 1 : si64, start = 0 : si64, step = 1 : si64}> : () -> tensor<1xi64>
    %17 = ttir.empty() : tensor<1xi64>
    %18 = "ttir.multiply"(%16, %1, %17) : (tensor<1xi64>, tensor<1xi64>, tensor<1xi64>) -> tensor<1xi64>
    %19 = ttir.empty() : tensor<1xi64>
    %20 = "ttir.add"(%18, %0, %19) : (tensor<1xi64>, tensor<1xi64>, tensor<1xi64>) -> tensor<1xi64>
    %21 = ttir.empty() : tensor<1x1x1x1xi64>
    %22 = "ttir.reshape"(%20, %21) <{shape = [1 : i32, 1 : i32, 1 : i32, 1 : i32]}> : (tensor<1xi64>, tensor<1x1x1x1xi64>) -> tensor<1x1x1x1xi64>
    %23 = ttir.empty() : tensor<1x8x1x128xi64>
    %24 = "ttir.broadcast"(%22, %23) <{broadcast_dimensions = array<i64: 1, 8, 1, 128>}> : (tensor<1x1x1x1xi64>, tensor<1x8x1x128xi64>) -> tensor<1x8x1x128xi64>
    %25 = ttir.empty() : tensor<1x8x1x128x1xi64>
    %26 = "ttir.reshape"(%24, %25) <{shape = [1 : i32, 8 : i32, 1 : i32, 128 : i32, 1 : i32]}> : (tensor<1x8x1x128xi64>, tensor<1x8x1x128x1xi64>) -> tensor<1x8x1x128x1xi64>
    %27 = ttir.empty() : tensor<1x8x1x1xi64>
    %28 = "ttir.reshape"(%15, %27) <{shape = [1 : i32, 8 : i32, 1 : i32, 1 : i32]}> : (tensor<8xi64>, tensor<1x8x1x1xi64>) -> tensor<1x8x1x1xi64>
    %29 = ttir.empty() : tensor<1x8x1x128xi64>
    %30 = "ttir.broadcast"(%28, %29) <{broadcast_dimensions = array<i64: 1, 1, 1, 128>}> : (tensor<1x8x1x1xi64>, tensor<1x8x1x128xi64>) -> tensor<1x8x1x128xi64>
    %31 = ttir.empty() : tensor<1x8x1x128x1xi64>
    %32 = "ttir.reshape"(%30, %31) <{shape = [1 : i32, 8 : i32, 1 : i32, 128 : i32, 1 : i32]}> : (tensor<1x8x1x128xi64>, tensor<1x8x1x128x1xi64>) -> tensor<1x8x1x128x1xi64>
    %33 = ttir.empty() : tensor<1x1x1x1xi64>
    %34 = "ttir.reshape"(%arg1, %33) <{shape = [1 : i32, 1 : i32, 1 : i32, 1 : i32]}> : (tensor<1xi64>, tensor<1x1x1x1xi64>) -> tensor<1x1x1x1xi64>
    %35 = ttir.empty() : tensor<1x8x1x128xi64>
    %36 = "ttir.broadcast"(%34, %35) <{broadcast_dimensions = array<i64: 1, 8, 1, 128>}> : (tensor<1x1x1x1xi64>, tensor<1x8x1x128xi64>) -> tensor<1x8x1x128xi64>
    %37 = ttir.empty() : tensor<1x8x1x128x1xi64>
    %38 = "ttir.reshape"(%36, %37) <{shape = [1 : i32, 8 : i32, 1 : i32, 128 : i32, 1 : i32]}> : (tensor<1x8x1x128xi64>, tensor<1x8x1x128x1xi64>) -> tensor<1x8x1x128x1xi64>
    %39 = ttir.empty() : tensor<1x1x1x128xi64>
    %40 = "ttir.reshape"(%10, %39) <{shape = [1 : i32, 1 : i32, 1 : i32, 128 : i32]}> : (tensor<128xi64>, tensor<1x1x1x128xi64>) -> tensor<1x1x1x128xi64>
    %41 = ttir.empty() : tensor<1x8x1x128xi64>
    %42 = "ttir.broadcast"(%40, %41) <{broadcast_dimensions = array<i64: 1, 8, 1, 1>}> : (tensor<1x1x1x128xi64>, tensor<1x8x1x128xi64>) -> tensor<1x8x1x128xi64>
    %43 = ttir.empty() : tensor<1x8x1x128x1xi64>
    %44 = "ttir.reshape"(%42, %43) <{shape = [1 : i32, 8 : i32, 1 : i32, 128 : i32, 1 : i32]}> : (tensor<1x8x1x128xi64>, tensor<1x8x1x128x1xi64>) -> tensor<1x8x1x128x1xi64>
    %45 = ttir.empty() : tensor<1x8x1x128x4xi64>
    %46 = "ttir.concat"(%26, %32, %38, %44, %45) <{dim = 4 : si32}> : (tensor<1x8x1x128x1xi64>, tensor<1x8x1x128x1xi64>, tensor<1x8x1x128x1xi64>, tensor<1x8x1x128x1xi64>, tensor<1x8x1x128x4xi64>) -> tensor<1x8x1x128x4xi64>
    %47 = ttir.empty() : tensor<1x8x64x128xbf16>
    %48 = "ttir.scatter"(%arg0, %46, %arg2, %47) <{index_vector_dim = 4 : i32, indices_are_sorted = false, input_batching_dims = array<i32>, inserted_window_dims = array<i32: 0, 1, 2, 3>, scatter_dims_to_operand_dims = array<i32: 0, 1, 2, 3>, scatter_indices_batching_dims = array<i32>, unique_indices = false, update_window_dims = array<i32>}> : (tensor<1x8x64x128xbf16>, tensor<1x8x1x128x4xi64>, tensor<1x8x1x128xbf16>, tensor<1x8x64x128xbf16>) -> tensor<1x8x64x128xbf16>
    return %48 : tensor<1x8x64x128xbf16>
  }
}

// Test case with a simple scatter op does not match fill/update cache pattern
module @scatter {
  func.func public @test_scatter(%arg0: tensor<1x3x320x320xf32>, %arg1: tensor<1x1xi64>, %arg2: tensor<1x3x32x32xf32>) -> tensor<1x3x320x320xf32> {
    // CHECK-NOT: ttir.fill_cache
    // CHECK-NOT: ttir.update_cache
    // CHECK: ttir.scatter
    %0 = ttir.empty() : tensor<1x3x320x320xf32>
    %1 = "ttir.scatter"(%arg0, %arg1, %arg2, %0) <{index_vector_dim = 1 : i32, indices_are_sorted = false, input_batching_dims = array<i32>, inserted_window_dims = array<i32: 0>, scatter_dims_to_operand_dims = array<i32: 0>, scatter_indices_batching_dims = array<i32>, unique_indices = false, update_window_dims = array<i32: 1, 2, 3>}> : (tensor<1x3x320x320xf32>, tensor<1x1xi64>, tensor<1x3x32x32xf32>, tensor<1x3x320x320xf32>) -> tensor<1x3x320x320xf32>
    return %1 : tensor<1x3x320x320xf32>
  }
}

// Test case with a scatter op that has to track all the way up to a const arange with the wrong values. Should not fuse scatter into ttir.update_cache
module {
  func.func @main(%arg0: tensor<1x8x64x128xbf16>, %arg1: tensor<14xi64>, %arg2: tensor<1x8x14x128xbf16>) -> tensor<1x8x64x128xbf16> {
    // CHECK-NOT: ttir.scatter
    // CHECK-NOT: ttir.update_cache
    // CHECK: %[[RET:[0-9]+]] = "ttir.fill_cache"(%arg0, %arg2) <{batch_offset = 0 : i32}> : (tensor<1x8x64x128xbf16>, tensor<1x8x14x128xbf16>) -> tensor<1x8x64x128xbf16>
    // CHECK: return %[[RET]] : tensor<1x8x64x128xbf16>
    %1 = "ttir.mesh_shard"(%arg1) <{shard_dims = array<i64: -1>, shard_direction = #ttcore.shard_direction<full_to_shard>, shard_shape = array<i64: 1>, shard_type = #ttcore.shard_type<identity>}> : (tensor<14xi64>) -> tensor<14xi64>
    %2 = ttir.empty() : tensor<1x8x64x128xbf16>
    %3 = "ttir.scatter"(%arg0, %1, %arg2, %2) <{index_vector_dim = 1 : i32, indices_are_sorted = false, input_batching_dims = array<i32>, inserted_window_dims = array<i32: 2>, scatter_dims_to_operand_dims = array<i32: 2>, scatter_indices_batching_dims = array<i32>, unique_indices = false, update_window_dims = array<i32: 0, 1, 3>}> : (tensor<1x8x64x128xbf16>, tensor<14xi64>, tensor<1x8x14x128xbf16>, tensor<1x8x64x128xbf16>) -> tensor<1x8x64x128xbf16>
    return %3 : tensor<1x8x64x128xbf16>
  }
}
