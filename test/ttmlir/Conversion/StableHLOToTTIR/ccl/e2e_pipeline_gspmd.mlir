// REQUIRES: stablehlo
// RUN: ttmlir-opt -split-input-file --stablehlo-to-ttir-pipeline -o %t %s
// RUN: FileCheck %s --input-file=%t
// UNSUPPORTED: true

module @jit_loss_pp attributes {mhlo.num_partitions = 8 : i32, mhlo.num_replicas = 1 : i32} {
  func.func public @main(%arg0: tensor<784x128xf32> {mhlo.sharding = "{replicated}"}, %arg1: tensor<128xf32> {mhlo.sharding = "{replicated}"}, %arg2: tensor<4x128x128xf32> {mhlo.sharding = "{devices=[2,1,1,4]<=[8] last_tile_dim_replicate}"}, %arg3: tensor<4x128xf32> {mhlo.sharding = "{devices=[2,1,4]<=[8] last_tile_dim_replicate}"}, %arg4: tensor<128x8xf32> {mhlo.sharding = "{replicated}"}, %arg5: tensor<8xf32> {mhlo.sharding = "{replicated}"}, %arg6: tensor<32x784xf32> {mhlo.sharding = "{devices=[2,1,4]<=[8] last_tile_dim_replicate}"}, %arg7: tensor<32x8xf32> {mhlo.sharding = "{devices=[2,1,4]<=[8] last_tile_dim_replicate}"}) -> (tensor<f32> {jax.result_info = ""}) {
    %0 = stablehlo.custom_call @Sharding(%arg0) {backend_config = "", mhlo.sharding = "{replicated}"} : (tensor<784x128xf32>) -> tensor<784x128xf32>
    %1 = stablehlo.custom_call @SPMDFullToShardShape(%0) {backend_config = "", mhlo.sharding = "{manual}"} : (tensor<784x128xf32>) -> tensor<784x128xf32>
    %2 = stablehlo.custom_call @Sharding(%arg1) {backend_config = "", mhlo.sharding = "{replicated}"} : (tensor<128xf32>) -> tensor<128xf32>
    %3 = stablehlo.custom_call @SPMDFullToShardShape(%2) {backend_config = "", mhlo.sharding = "{manual}"} : (tensor<128xf32>) -> tensor<128xf32>
    %4 = stablehlo.custom_call @Sharding(%arg2) {backend_config = "", mhlo.sharding = "{devices=[2,1,1,4]<=[8] last_tile_dim_replicate}"} : (tensor<4x128x128xf32>) -> tensor<4x128x128xf32>
    %5 = stablehlo.custom_call @SPMDFullToShardShape(%4) {backend_config = "", mhlo.sharding = "{manual}"} : (tensor<4x128x128xf32>) -> tensor<2x128x128xf32>
    %6 = stablehlo.custom_call @Sharding(%arg3) {backend_config = "", mhlo.sharding = "{devices=[2,1,4]<=[8] last_tile_dim_replicate}"} : (tensor<4x128xf32>) -> tensor<4x128xf32>
    %7 = stablehlo.custom_call @SPMDFullToShardShape(%6) {backend_config = "", mhlo.sharding = "{manual}"} : (tensor<4x128xf32>) -> tensor<2x128xf32>
    %8 = stablehlo.custom_call @Sharding(%arg4) {backend_config = "", mhlo.sharding = "{replicated}"} : (tensor<128x8xf32>) -> tensor<128x8xf32>
    %9 = stablehlo.custom_call @SPMDFullToShardShape(%8) {backend_config = "", mhlo.sharding = "{manual}"} : (tensor<128x8xf32>) -> tensor<128x8xf32>
    %10 = stablehlo.custom_call @Sharding(%arg5) {backend_config = "", mhlo.sharding = "{replicated}"} : (tensor<8xf32>) -> tensor<8xf32>
    %11 = stablehlo.custom_call @SPMDFullToShardShape(%10) {backend_config = "", mhlo.sharding = "{manual}"} : (tensor<8xf32>) -> tensor<8xf32>
    %12 = stablehlo.custom_call @Sharding(%arg6) {backend_config = "", mhlo.sharding = "{devices=[2,1,4]<=[8] last_tile_dim_replicate}"} : (tensor<32x784xf32>) -> tensor<32x784xf32>
    %13 = stablehlo.custom_call @SPMDFullToShardShape(%12) {backend_config = "", mhlo.sharding = "{manual}"} : (tensor<32x784xf32>) -> tensor<16x784xf32>
    %14 = stablehlo.custom_call @Sharding(%arg7) {backend_config = "", mhlo.sharding = "{devices=[2,1,4]<=[8] last_tile_dim_replicate}"} : (tensor<32x8xf32>) -> tensor<32x8xf32>
    %15 = stablehlo.custom_call @SPMDFullToShardShape(%14) {backend_config = "", mhlo.sharding = "{manual}"} : (tensor<32x8xf32>) -> tensor<16x8xf32>
    %16 = call @shmap_body(%1, %3, %5, %7, %9, %11, %13, %15) : (tensor<784x128xf32>, tensor<128xf32>, tensor<2x128x128xf32>, tensor<2x128xf32>, tensor<128x8xf32>, tensor<8xf32>, tensor<16x784xf32>, tensor<16x8xf32>) -> tensor<f32>
    %17 = stablehlo.custom_call @Sharding(%16) {backend_config = "", mhlo.sharding = "{manual}"} : (tensor<f32>) -> tensor<f32>
    %18 = stablehlo.custom_call @SPMDShardToFullShape(%17) {backend_config = "", mhlo.sharding = "{replicated}"} : (tensor<f32>) -> tensor<f32>
    return %18 : tensor<f32>
  }
  func.func private @shmap_body(%arg0: tensor<784x128xf32>, %arg1: tensor<128xf32>, %arg2: tensor<2x128x128xf32>, %arg3: tensor<2x128xf32>, %arg4: tensor<128x8xf32>, %arg5: tensor<8xf32>, %arg6: tensor<16x784xf32>, %arg7: tensor<16x8xf32>) -> (tensor<f32> {jax.result_info = "[]"}) {
    %0 = stablehlo.reshape %arg6 : (tensor<16x784xf32>) -> tensor<2x8x784xf32>
    %1 = stablehlo.dot_general %0, %arg0, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<2x8x784xf32>, tensor<784x128xf32>) -> tensor<2x8x128xf32>
    %2 = stablehlo.broadcast_in_dim %arg1, dims = [2] : (tensor<128xf32>) -> tensor<1x1x128xf32>
    %3 = stablehlo.broadcast_in_dim %2, dims = [0, 1, 2] : (tensor<1x1x128xf32>) -> tensor<2x8x128xf32>
    %4 = stablehlo.add %1, %3 : tensor<2x8x128xf32>
    %5 = call @relu(%4) : (tensor<2x8x128xf32>) -> tensor<2x8x128xf32>
    %c = stablehlo.constant dense<4> : tensor<ui32>
    %c_0 = stablehlo.constant dense<2> : tensor<ui32>
    %6 = stablehlo.partition_id : tensor<ui32>
    %7 = stablehlo.divide %6, %c : tensor<ui32>
    %8 = stablehlo.remainder %7, %c_0 : tensor<ui32>
    %9 = stablehlo.convert %8 : (tensor<ui32>) -> tensor<i32>
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %10 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<2x8x128xf32>
    %cst_1 = stablehlo.constant dense<0x7FC00000> : tensor<f32>
    %11 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<2x8x128xf32>
    %12 = stablehlo.multiply %10, %11 : tensor<2x8x128xf32>
    %13 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<2x8x128xf32>
    %14 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<2x8x128xf32>
    %15 = stablehlo.multiply %13, %14 : tensor<2x8x128xf32>
    %c_2 = stablehlo.constant dense<0> : tensor<i32>
    %16 = stablehlo.compare  EQ, %9, %c_2,  SIGNED : (tensor<i32>, tensor<i32>) -> tensor<i1>
    %17 = stablehlo.slice %5 [0:1, 0:8, 0:128] : (tensor<2x8x128xf32>) -> tensor<1x8x128xf32>
    %18 = stablehlo.reshape %17 : (tensor<1x8x128xf32>) -> tensor<8x128xf32>
    %19 = stablehlo.slice %15 [0:1, 0:8, 0:128] : (tensor<2x8x128xf32>) -> tensor<1x8x128xf32>
    %20 = stablehlo.reshape %19 : (tensor<1x8x128xf32>) -> tensor<8x128xf32>
    %21 = call @_where(%16, %18, %20) : (tensor<i1>, tensor<8x128xf32>, tensor<8x128xf32>) -> tensor<8x128xf32>
    %22 = stablehlo.broadcast_in_dim %c_2, dims = [] : (tensor<i32>) -> tensor<1xi32>
    %23 = "stablehlo.scatter"(%15, %22, %21) <{indices_are_sorted = true, scatter_dimension_numbers = #stablehlo.scatter<update_window_dims = [0, 1], inserted_window_dims = [0], scatter_dims_to_operand_dims = [0]>, unique_indices = true}> ({
    ^bb0(%arg8: tensor<f32>, %arg9: tensor<f32>):
      stablehlo.return %arg9 : tensor<f32>
    }) : (tensor<2x8x128xf32>, tensor<1xi32>, tensor<8x128xf32>) -> tensor<2x8x128xf32>
    %24 = stablehlo.dot_general %23, %arg2, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<2x8x128xf32>, tensor<2x128x128xf32>) -> tensor<2x8x128xf32>
    %25 = stablehlo.broadcast_in_dim %arg3, dims = [0, 2] : (tensor<2x128xf32>) -> tensor<2x1x128xf32>
    %26 = stablehlo.broadcast_in_dim %25, dims = [0, 1, 2] : (tensor<2x1x128xf32>) -> tensor<2x8x128xf32>
    %27 = stablehlo.add %24, %26 : tensor<2x8x128xf32>
    %28 = call @relu_0(%27) : (tensor<2x8x128xf32>) -> tensor<2x8x128xf32>
    %c_3 = stablehlo.constant dense<1> : tensor<i32>
    %29 = stablehlo.compare  EQ, %9, %c_3,  SIGNED : (tensor<i32>, tensor<i32>) -> tensor<i1>
    %c_4 = stablehlo.constant dense<-1> : tensor<i32>
    %c_5 = stablehlo.constant dense<0> : tensor<i32>
    %30 = stablehlo.compare  LT, %c_4, %c_5,  SIGNED : (tensor<i32>, tensor<i32>) -> tensor<i1>
    %c_6 = stablehlo.constant dense<-1> : tensor<i32>
    %c_7 = stablehlo.constant dense<2> : tensor<i32>
    %31 = stablehlo.add %c_6, %c_7 : tensor<i32>
    %32 = stablehlo.select %30, %31, %c_4 : tensor<i1>, tensor<i32>
    %33 = stablehlo.compare  LT, %c_5, %c_5,  SIGNED : (tensor<i32>, tensor<i32>) -> tensor<i1>
    %c_8 = stablehlo.constant dense<8> : tensor<i32>
    %34 = stablehlo.add %c_2, %c_8 : tensor<i32>
    %35 = stablehlo.select %33, %34, %c_5 : tensor<i1>, tensor<i32>
    %36 = stablehlo.compare  LT, %c_5, %c_5,  SIGNED : (tensor<i32>, tensor<i32>) -> tensor<i1>
    %c_9 = stablehlo.constant dense<128> : tensor<i32>
    %37 = stablehlo.add %c_2, %c_9 : tensor<i32>
    %38 = stablehlo.select %36, %37, %c_5 : tensor<i1>, tensor<i32>
    %39 = stablehlo.dynamic_slice %28, %32, %35, %38, sizes = [1, 8, 128] : (tensor<2x8x128xf32>, tensor<i32>, tensor<i32>, tensor<i32>) -> tensor<1x8x128xf32>
    %40 = stablehlo.reshape %39 : (tensor<1x8x128xf32>) -> tensor<8x128xf32>
    %41 = stablehlo.slice %12 [1:2, 0:8, 0:128] : (tensor<2x8x128xf32>) -> tensor<1x8x128xf32>
    %42 = stablehlo.reshape %41 : (tensor<1x8x128xf32>) -> tensor<8x128xf32>
    %43 = call @_where_1(%29, %40, %42) : (tensor<i1>, tensor<8x128xf32>, tensor<8x128xf32>) -> tensor<8x128xf32>
    %44 = stablehlo.broadcast_in_dim %c_3, dims = [] : (tensor<i32>) -> tensor<1xi32>
    %45 = "stablehlo.scatter"(%12, %44, %43) <{indices_are_sorted = true, scatter_dimension_numbers = #stablehlo.scatter<update_window_dims = [0, 1], inserted_window_dims = [0], scatter_dims_to_operand_dims = [0]>, unique_indices = true}> ({
    ^bb0(%arg8: tensor<f32>, %arg9: tensor<f32>):
      stablehlo.return %arg9 : tensor<f32>
    }) : (tensor<2x8x128xf32>, tensor<1xi32>, tensor<8x128xf32>) -> tensor<2x8x128xf32>
    %46 = call @_roll_static(%28) : (tensor<2x8x128xf32>) -> tensor<2x8x128xf32>
    %47 = stablehlo.compare  LT, %c_4, %c_5,  SIGNED : (tensor<i32>, tensor<i32>) -> tensor<i1>
    %48 = stablehlo.add %c_6, %c_7 : tensor<i32>
    %49 = stablehlo.select %47, %48, %c_4 : tensor<i1>, tensor<i32>
    %50 = stablehlo.compare  LT, %c_5, %c_5,  SIGNED : (tensor<i32>, tensor<i32>) -> tensor<i1>
    %51 = stablehlo.add %c_2, %c_8 : tensor<i32>
    %52 = stablehlo.select %50, %51, %c_5 : tensor<i1>, tensor<i32>
    %53 = stablehlo.compare  LT, %c_5, %c_5,  SIGNED : (tensor<i32>, tensor<i32>) -> tensor<i1>
    %54 = stablehlo.add %c_2, %c_9 : tensor<i32>
    %55 = stablehlo.select %53, %54, %c_5 : tensor<i1>, tensor<i32>
    %56 = stablehlo.dynamic_slice %28, %49, %52, %55, sizes = [1, 8, 128] : (tensor<2x8x128xf32>, tensor<i32>, tensor<i32>, tensor<i32>) -> tensor<1x8x128xf32>
    %57 = stablehlo.reshape %56 : (tensor<1x8x128xf32>) -> tensor<8x128xf32>
    %58 = "stablehlo.collective_permute"(%57) <{channel_handle = #stablehlo.channel_handle<handle = 1, type = 1>, source_target_pairs = dense<[[0, 4], [4, 0], [1, 5], [5, 1], [2, 6], [6, 2], [3, 7], [7, 3]]> : tensor<8x2xi64>}> : (tensor<8x128xf32>) -> tensor<8x128xf32>
    %59 = stablehlo.broadcast_in_dim %c_2, dims = [] : (tensor<i32>) -> tensor<1xi32>
    %60 = "stablehlo.scatter"(%46, %59, %58) <{indices_are_sorted = true, scatter_dimension_numbers = #stablehlo.scatter<update_window_dims = [0, 1], inserted_window_dims = [0], scatter_dims_to_operand_dims = [0]>, unique_indices = true}> ({
    ^bb0(%arg8: tensor<f32>, %arg9: tensor<f32>):
      stablehlo.return %arg9 : tensor<f32>
    }) : (tensor<2x8x128xf32>, tensor<1xi32>, tensor<8x128xf32>) -> tensor<2x8x128xf32>
    %61 = "stablehlo.collective_permute"(%45) <{channel_handle = #stablehlo.channel_handle<handle = 2, type = 1>, source_target_pairs = dense<[[0, 4], [4, 0], [1, 5], [5, 1], [2, 6], [6, 2], [3, 7], [7, 3]]> : tensor<8x2xi64>}> : (tensor<2x8x128xf32>) -> tensor<2x8x128xf32>
    %62 = stablehlo.compare  EQ, %9, %c_2,  SIGNED : (tensor<i32>, tensor<i32>) -> tensor<i1>
    %63 = stablehlo.slice %5 [1:2, 0:8, 0:128] : (tensor<2x8x128xf32>) -> tensor<1x8x128xf32>
    %64 = stablehlo.reshape %63 : (tensor<1x8x128xf32>) -> tensor<8x128xf32>
    %65 = stablehlo.slice %60 [0:1, 0:8, 0:128] : (tensor<2x8x128xf32>) -> tensor<1x8x128xf32>
    %66 = stablehlo.reshape %65 : (tensor<1x8x128xf32>) -> tensor<8x128xf32>
    %67 = call @_where_2(%62, %64, %66) : (tensor<i1>, tensor<8x128xf32>, tensor<8x128xf32>) -> tensor<8x128xf32>
    %68 = stablehlo.broadcast_in_dim %c_2, dims = [] : (tensor<i32>) -> tensor<1xi32>
    %69 = "stablehlo.scatter"(%60, %68, %67) <{indices_are_sorted = true, scatter_dimension_numbers = #stablehlo.scatter<update_window_dims = [0, 1], inserted_window_dims = [0], scatter_dims_to_operand_dims = [0]>, unique_indices = true}> ({
    ^bb0(%arg8: tensor<f32>, %arg9: tensor<f32>):
      stablehlo.return %arg9 : tensor<f32>
    }) : (tensor<2x8x128xf32>, tensor<1xi32>, tensor<8x128xf32>) -> tensor<2x8x128xf32>
    %70 = stablehlo.dot_general %69, %arg2, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<2x8x128xf32>, tensor<2x128x128xf32>) -> tensor<2x8x128xf32>
    %71 = stablehlo.broadcast_in_dim %arg3, dims = [0, 2] : (tensor<2x128xf32>) -> tensor<2x1x128xf32>
    %72 = stablehlo.broadcast_in_dim %71, dims = [0, 1, 2] : (tensor<2x1x128xf32>) -> tensor<2x8x128xf32>
    %73 = stablehlo.add %70, %72 : tensor<2x8x128xf32>
    %74 = call @relu_3(%73) : (tensor<2x8x128xf32>) -> tensor<2x8x128xf32>
    %75 = stablehlo.compare  EQ, %9, %c_3,  SIGNED : (tensor<i32>, tensor<i32>) -> tensor<i1>
    %76 = stablehlo.compare  LT, %c_4, %c_5,  SIGNED : (tensor<i32>, tensor<i32>) -> tensor<i1>
    %77 = stablehlo.add %c_6, %c_7 : tensor<i32>
    %78 = stablehlo.select %76, %77, %c_4 : tensor<i1>, tensor<i32>
    %79 = stablehlo.compare  LT, %c_5, %c_5,  SIGNED : (tensor<i32>, tensor<i32>) -> tensor<i1>
    %80 = stablehlo.add %c_2, %c_8 : tensor<i32>
    %81 = stablehlo.select %79, %80, %c_5 : tensor<i1>, tensor<i32>
    %82 = stablehlo.compare  LT, %c_5, %c_5,  SIGNED : (tensor<i32>, tensor<i32>) -> tensor<i1>
    %83 = stablehlo.add %c_2, %c_9 : tensor<i32>
    %84 = stablehlo.select %82, %83, %c_5 : tensor<i1>, tensor<i32>
    %85 = stablehlo.dynamic_slice %74, %78, %81, %84, sizes = [1, 8, 128] : (tensor<2x8x128xf32>, tensor<i32>, tensor<i32>, tensor<i32>) -> tensor<1x8x128xf32>
    %86 = stablehlo.reshape %85 : (tensor<1x8x128xf32>) -> tensor<8x128xf32>
    %87 = stablehlo.slice %61 [0:1, 0:8, 0:128] : (tensor<2x8x128xf32>) -> tensor<1x8x128xf32>
    %88 = stablehlo.reshape %87 : (tensor<1x8x128xf32>) -> tensor<8x128xf32>
    %89 = call @_where_4(%75, %86, %88) : (tensor<i1>, tensor<8x128xf32>, tensor<8x128xf32>) -> tensor<8x128xf32>
    %90 = stablehlo.broadcast_in_dim %c_2, dims = [] : (tensor<i32>) -> tensor<1xi32>
    %91 = "stablehlo.scatter"(%61, %90, %89) <{indices_are_sorted = true, scatter_dimension_numbers = #stablehlo.scatter<update_window_dims = [0, 1], inserted_window_dims = [0], scatter_dims_to_operand_dims = [0]>, unique_indices = true}> ({
    ^bb0(%arg8: tensor<f32>, %arg9: tensor<f32>):
      stablehlo.return %arg9 : tensor<f32>
    }) : (tensor<2x8x128xf32>, tensor<1xi32>, tensor<8x128xf32>) -> tensor<2x8x128xf32>
    %92 = call @_roll_static_5(%74) : (tensor<2x8x128xf32>) -> tensor<2x8x128xf32>
    %93 = stablehlo.compare  LT, %c_4, %c_5,  SIGNED : (tensor<i32>, tensor<i32>) -> tensor<i1>
    %94 = stablehlo.add %c_6, %c_7 : tensor<i32>
    %95 = stablehlo.select %93, %94, %c_4 : tensor<i1>, tensor<i32>
    %96 = stablehlo.compare  LT, %c_5, %c_5,  SIGNED : (tensor<i32>, tensor<i32>) -> tensor<i1>
    %97 = stablehlo.add %c_2, %c_8 : tensor<i32>
    %98 = stablehlo.select %96, %97, %c_5 : tensor<i1>, tensor<i32>
    %99 = stablehlo.compare  LT, %c_5, %c_5,  SIGNED : (tensor<i32>, tensor<i32>) -> tensor<i1>
    %100 = stablehlo.add %c_2, %c_9 : tensor<i32>
    %101 = stablehlo.select %99, %100, %c_5 : tensor<i1>, tensor<i32>
    %102 = stablehlo.dynamic_slice %74, %95, %98, %101, sizes = [1, 8, 128] : (tensor<2x8x128xf32>, tensor<i32>, tensor<i32>, tensor<i32>) -> tensor<1x8x128xf32>
    %103 = stablehlo.reshape %102 : (tensor<1x8x128xf32>) -> tensor<8x128xf32>
    %104 = "stablehlo.collective_permute"(%103) <{channel_handle = #stablehlo.channel_handle<handle = 3, type = 1>, source_target_pairs = dense<[[0, 4], [4, 0], [1, 5], [5, 1], [2, 6], [6, 2], [3, 7], [7, 3]]> : tensor<8x2xi64>}> : (tensor<8x128xf32>) -> tensor<8x128xf32>
    %105 = stablehlo.broadcast_in_dim %c_2, dims = [] : (tensor<i32>) -> tensor<1xi32>
    %106 = "stablehlo.scatter"(%92, %105, %104) <{indices_are_sorted = true, scatter_dimension_numbers = #stablehlo.scatter<update_window_dims = [0, 1], inserted_window_dims = [0], scatter_dims_to_operand_dims = [0]>, unique_indices = true}> ({
    ^bb0(%arg8: tensor<f32>, %arg9: tensor<f32>):
      stablehlo.return %arg9 : tensor<f32>
    }) : (tensor<2x8x128xf32>, tensor<1xi32>, tensor<8x128xf32>) -> tensor<2x8x128xf32>
    %107 = "stablehlo.collective_permute"(%5) <{channel_handle = #stablehlo.channel_handle<handle = 4, type = 1>, source_target_pairs = dense<[[0, 4], [4, 0], [1, 5], [5, 1], [2, 6], [6, 2], [3, 7], [7, 3]]> : tensor<8x2xi64>}> : (tensor<2x8x128xf32>) -> tensor<2x8x128xf32>
    %108 = stablehlo.compare  EQ, %9, %c_2,  SIGNED : (tensor<i32>, tensor<i32>) -> tensor<i1>
    %109 = stablehlo.slice %107 [0:1, 0:8, 0:128] : (tensor<2x8x128xf32>) -> tensor<1x8x128xf32>
    %110 = stablehlo.reshape %109 : (tensor<1x8x128xf32>) -> tensor<8x128xf32>
    %111 = stablehlo.slice %106 [0:1, 0:8, 0:128] : (tensor<2x8x128xf32>) -> tensor<1x8x128xf32>
    %112 = stablehlo.reshape %111 : (tensor<1x8x128xf32>) -> tensor<8x128xf32>
    %113 = call @_where_6(%108, %110, %112) : (tensor<i1>, tensor<8x128xf32>, tensor<8x128xf32>) -> tensor<8x128xf32>
    %114 = stablehlo.broadcast_in_dim %c_2, dims = [] : (tensor<i32>) -> tensor<1xi32>
    %115 = "stablehlo.scatter"(%106, %114, %113) <{indices_are_sorted = true, scatter_dimension_numbers = #stablehlo.scatter<update_window_dims = [0, 1], inserted_window_dims = [0], scatter_dims_to_operand_dims = [0]>, unique_indices = true}> ({
    ^bb0(%arg8: tensor<f32>, %arg9: tensor<f32>):
      stablehlo.return %arg9 : tensor<f32>
    }) : (tensor<2x8x128xf32>, tensor<1xi32>, tensor<8x128xf32>) -> tensor<2x8x128xf32>
    %116 = stablehlo.dot_general %115, %arg2, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<2x8x128xf32>, tensor<2x128x128xf32>) -> tensor<2x8x128xf32>
    %117 = stablehlo.broadcast_in_dim %arg3, dims = [0, 2] : (tensor<2x128xf32>) -> tensor<2x1x128xf32>
    %118 = stablehlo.broadcast_in_dim %117, dims = [0, 1, 2] : (tensor<2x1x128xf32>) -> tensor<2x8x128xf32>
    %119 = stablehlo.add %116, %118 : tensor<2x8x128xf32>
    %120 = call @relu_7(%119) : (tensor<2x8x128xf32>) -> tensor<2x8x128xf32>
    %121 = stablehlo.compare  EQ, %9, %c_3,  SIGNED : (tensor<i32>, tensor<i32>) -> tensor<i1>
    %122 = stablehlo.compare  LT, %c_4, %c_5,  SIGNED : (tensor<i32>, tensor<i32>) -> tensor<i1>
    %123 = stablehlo.add %c_6, %c_7 : tensor<i32>
    %124 = stablehlo.select %122, %123, %c_4 : tensor<i1>, tensor<i32>
    %125 = stablehlo.compare  LT, %c_5, %c_5,  SIGNED : (tensor<i32>, tensor<i32>) -> tensor<i1>
    %126 = stablehlo.add %c_2, %c_8 : tensor<i32>
    %127 = stablehlo.select %125, %126, %c_5 : tensor<i1>, tensor<i32>
    %128 = stablehlo.compare  LT, %c_5, %c_5,  SIGNED : (tensor<i32>, tensor<i32>) -> tensor<i1>
    %129 = stablehlo.add %c_2, %c_9 : tensor<i32>
    %130 = stablehlo.select %128, %129, %c_5 : tensor<i1>, tensor<i32>
    %131 = stablehlo.dynamic_slice %120, %124, %127, %130, sizes = [1, 8, 128] : (tensor<2x8x128xf32>, tensor<i32>, tensor<i32>, tensor<i32>) -> tensor<1x8x128xf32>
    %132 = stablehlo.reshape %131 : (tensor<1x8x128xf32>) -> tensor<8x128xf32>
    %133 = stablehlo.slice %91 [1:2, 0:8, 0:128] : (tensor<2x8x128xf32>) -> tensor<1x8x128xf32>
    %134 = stablehlo.reshape %133 : (tensor<1x8x128xf32>) -> tensor<8x128xf32>
    %135 = call @_where_8(%121, %132, %134) : (tensor<i1>, tensor<8x128xf32>, tensor<8x128xf32>) -> tensor<8x128xf32>
    %136 = stablehlo.broadcast_in_dim %c_3, dims = [] : (tensor<i32>) -> tensor<1xi32>
    %137 = "stablehlo.scatter"(%91, %136, %135) <{indices_are_sorted = true, scatter_dimension_numbers = #stablehlo.scatter<update_window_dims = [0, 1], inserted_window_dims = [0], scatter_dims_to_operand_dims = [0]>, unique_indices = true}> ({
    ^bb0(%arg8: tensor<f32>, %arg9: tensor<f32>):
      stablehlo.return %arg9 : tensor<f32>
    }) : (tensor<2x8x128xf32>, tensor<1xi32>, tensor<8x128xf32>) -> tensor<2x8x128xf32>
    %138 = call @_roll_static_9(%120) : (tensor<2x8x128xf32>) -> tensor<2x8x128xf32>
    %139 = stablehlo.compare  LT, %c_4, %c_5,  SIGNED : (tensor<i32>, tensor<i32>) -> tensor<i1>
    %140 = stablehlo.add %c_6, %c_7 : tensor<i32>
    %141 = stablehlo.select %139, %140, %c_4 : tensor<i1>, tensor<i32>
    %142 = stablehlo.compare  LT, %c_5, %c_5,  SIGNED : (tensor<i32>, tensor<i32>) -> tensor<i1>
    %143 = stablehlo.add %c_2, %c_8 : tensor<i32>
    %144 = stablehlo.select %142, %143, %c_5 : tensor<i1>, tensor<i32>
    %145 = stablehlo.compare  LT, %c_5, %c_5,  SIGNED : (tensor<i32>, tensor<i32>) -> tensor<i1>
    %146 = stablehlo.add %c_2, %c_9 : tensor<i32>
    %147 = stablehlo.select %145, %146, %c_5 : tensor<i1>, tensor<i32>
    %148 = stablehlo.dynamic_slice %120, %141, %144, %147, sizes = [1, 8, 128] : (tensor<2x8x128xf32>, tensor<i32>, tensor<i32>, tensor<i32>) -> tensor<1x8x128xf32>
    %149 = stablehlo.reshape %148 : (tensor<1x8x128xf32>) -> tensor<8x128xf32>
    %150 = "stablehlo.collective_permute"(%149) <{channel_handle = #stablehlo.channel_handle<handle = 5, type = 1>, source_target_pairs = dense<[[0, 4], [4, 0], [1, 5], [5, 1], [2, 6], [6, 2], [3, 7], [7, 3]]> : tensor<8x2xi64>}> : (tensor<8x128xf32>) -> tensor<8x128xf32>
    %151 = stablehlo.broadcast_in_dim %c_2, dims = [] : (tensor<i32>) -> tensor<1xi32>
    %152 = "stablehlo.scatter"(%138, %151, %150) <{indices_are_sorted = true, scatter_dimension_numbers = #stablehlo.scatter<update_window_dims = [0, 1], inserted_window_dims = [0], scatter_dims_to_operand_dims = [0]>, unique_indices = true}> ({
    ^bb0(%arg8: tensor<f32>, %arg9: tensor<f32>):
      stablehlo.return %arg9 : tensor<f32>
    }) : (tensor<2x8x128xf32>, tensor<1xi32>, tensor<8x128xf32>) -> tensor<2x8x128xf32>
    %153 = "stablehlo.collective_permute"(%137) <{channel_handle = #stablehlo.channel_handle<handle = 6, type = 1>, source_target_pairs = dense<[[0, 4], [4, 0], [1, 5], [5, 1], [2, 6], [6, 2], [3, 7], [7, 3]]> : tensor<8x2xi64>}> : (tensor<2x8x128xf32>) -> tensor<2x8x128xf32>
    %154 = stablehlo.compare  EQ, %9, %c_2,  SIGNED : (tensor<i32>, tensor<i32>) -> tensor<i1>
    %155 = stablehlo.slice %107 [1:2, 0:8, 0:128] : (tensor<2x8x128xf32>) -> tensor<1x8x128xf32>
    %156 = stablehlo.reshape %155 : (tensor<1x8x128xf32>) -> tensor<8x128xf32>
    %157 = stablehlo.slice %152 [0:1, 0:8, 0:128] : (tensor<2x8x128xf32>) -> tensor<1x8x128xf32>
    %158 = stablehlo.reshape %157 : (tensor<1x8x128xf32>) -> tensor<8x128xf32>
    %159 = call @_where_10(%154, %156, %158) : (tensor<i1>, tensor<8x128xf32>, tensor<8x128xf32>) -> tensor<8x128xf32>
    %160 = stablehlo.broadcast_in_dim %c_2, dims = [] : (tensor<i32>) -> tensor<1xi32>
    %161 = "stablehlo.scatter"(%152, %160, %159) <{indices_are_sorted = true, scatter_dimension_numbers = #stablehlo.scatter<update_window_dims = [0, 1], inserted_window_dims = [0], scatter_dims_to_operand_dims = [0]>, unique_indices = true}> ({
    ^bb0(%arg8: tensor<f32>, %arg9: tensor<f32>):
      stablehlo.return %arg9 : tensor<f32>
    }) : (tensor<2x8x128xf32>, tensor<1xi32>, tensor<8x128xf32>) -> tensor<2x8x128xf32>
    %162 = stablehlo.dot_general %161, %arg2, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<2x8x128xf32>, tensor<2x128x128xf32>) -> tensor<2x8x128xf32>
    %163 = stablehlo.broadcast_in_dim %arg3, dims = [0, 2] : (tensor<2x128xf32>) -> tensor<2x1x128xf32>
    %164 = stablehlo.broadcast_in_dim %163, dims = [0, 1, 2] : (tensor<2x1x128xf32>) -> tensor<2x8x128xf32>
    %165 = stablehlo.add %162, %164 : tensor<2x8x128xf32>
    %166 = call @relu_11(%165) : (tensor<2x8x128xf32>) -> tensor<2x8x128xf32>
    %167 = stablehlo.compare  EQ, %9, %c_3,  SIGNED : (tensor<i32>, tensor<i32>) -> tensor<i1>
    %168 = stablehlo.compare  LT, %c_4, %c_5,  SIGNED : (tensor<i32>, tensor<i32>) -> tensor<i1>
    %169 = stablehlo.add %c_6, %c_7 : tensor<i32>
    %170 = stablehlo.select %168, %169, %c_4 : tensor<i1>, tensor<i32>
    %171 = stablehlo.compare  LT, %c_5, %c_5,  SIGNED : (tensor<i32>, tensor<i32>) -> tensor<i1>
    %172 = stablehlo.add %c_2, %c_8 : tensor<i32>
    %173 = stablehlo.select %171, %172, %c_5 : tensor<i1>, tensor<i32>
    %174 = stablehlo.compare  LT, %c_5, %c_5,  SIGNED : (tensor<i32>, tensor<i32>) -> tensor<i1>
    %175 = stablehlo.add %c_2, %c_9 : tensor<i32>
    %176 = stablehlo.select %174, %175, %c_5 : tensor<i1>, tensor<i32>
    %177 = stablehlo.dynamic_slice %166, %170, %173, %176, sizes = [1, 8, 128] : (tensor<2x8x128xf32>, tensor<i32>, tensor<i32>, tensor<i32>) -> tensor<1x8x128xf32>
    %178 = stablehlo.reshape %177 : (tensor<1x8x128xf32>) -> tensor<8x128xf32>
    %179 = stablehlo.slice %153 [0:1, 0:8, 0:128] : (tensor<2x8x128xf32>) -> tensor<1x8x128xf32>
    %180 = stablehlo.reshape %179 : (tensor<1x8x128xf32>) -> tensor<8x128xf32>
    %181 = call @_where_12(%167, %178, %180) : (tensor<i1>, tensor<8x128xf32>, tensor<8x128xf32>) -> tensor<8x128xf32>
    %182 = stablehlo.broadcast_in_dim %c_2, dims = [] : (tensor<i32>) -> tensor<1xi32>
    %183 = "stablehlo.scatter"(%153, %182, %181) <{indices_are_sorted = true, scatter_dimension_numbers = #stablehlo.scatter<update_window_dims = [0, 1], inserted_window_dims = [0], scatter_dims_to_operand_dims = [0]>, unique_indices = true}> ({
    ^bb0(%arg8: tensor<f32>, %arg9: tensor<f32>):
      stablehlo.return %arg9 : tensor<f32>
    }) : (tensor<2x8x128xf32>, tensor<1xi32>, tensor<8x128xf32>) -> tensor<2x8x128xf32>
    %184 = call @_roll_static_13(%166) : (tensor<2x8x128xf32>) -> tensor<2x8x128xf32>
    %185 = stablehlo.compare  LT, %c_4, %c_5,  SIGNED : (tensor<i32>, tensor<i32>) -> tensor<i1>
    %186 = stablehlo.add %c_6, %c_7 : tensor<i32>
    %187 = stablehlo.select %185, %186, %c_4 : tensor<i1>, tensor<i32>
    %188 = stablehlo.compare  LT, %c_5, %c_5,  SIGNED : (tensor<i32>, tensor<i32>) -> tensor<i1>
    %189 = stablehlo.add %c_2, %c_8 : tensor<i32>
    %190 = stablehlo.select %188, %189, %c_5 : tensor<i1>, tensor<i32>
    %191 = stablehlo.compare  LT, %c_5, %c_5,  SIGNED : (tensor<i32>, tensor<i32>) -> tensor<i1>
    %192 = stablehlo.add %c_2, %c_9 : tensor<i32>
    %193 = stablehlo.select %191, %192, %c_5 : tensor<i1>, tensor<i32>
    %194 = stablehlo.dynamic_slice %166, %187, %190, %193, sizes = [1, 8, 128] : (tensor<2x8x128xf32>, tensor<i32>, tensor<i32>, tensor<i32>) -> tensor<1x8x128xf32>
    %195 = stablehlo.reshape %194 : (tensor<1x8x128xf32>) -> tensor<8x128xf32>
    %196 = "stablehlo.collective_permute"(%195) <{channel_handle = #stablehlo.channel_handle<handle = 7, type = 1>, source_target_pairs = dense<[[0, 4], [4, 0], [1, 5], [5, 1], [2, 6], [6, 2], [3, 7], [7, 3]]> : tensor<8x2xi64>}> : (tensor<8x128xf32>) -> tensor<8x128xf32>
    %197 = stablehlo.broadcast_in_dim %c_2, dims = [] : (tensor<i32>) -> tensor<1xi32>
    %198 = "stablehlo.scatter"(%184, %197, %196) <{indices_are_sorted = true, scatter_dimension_numbers = #stablehlo.scatter<update_window_dims = [0, 1], inserted_window_dims = [0], scatter_dims_to_operand_dims = [0]>, unique_indices = true}> ({
    ^bb0(%arg8: tensor<f32>, %arg9: tensor<f32>):
      stablehlo.return %arg9 : tensor<f32>
    }) : (tensor<2x8x128xf32>, tensor<1xi32>, tensor<8x128xf32>) -> tensor<2x8x128xf32>
    %199 = "stablehlo.collective_permute"(%107) <{channel_handle = #stablehlo.channel_handle<handle = 8, type = 1>, source_target_pairs = dense<[[0, 4], [4, 0], [1, 5], [5, 1], [2, 6], [6, 2], [3, 7], [7, 3]]> : tensor<8x2xi64>}> : (tensor<2x8x128xf32>) -> tensor<2x8x128xf32>
    %200 = stablehlo.compare  EQ, %9, %c_2,  SIGNED : (tensor<i32>, tensor<i32>) -> tensor<i1>
    %201 = stablehlo.slice %199 [0:1, 0:8, 0:128] : (tensor<2x8x128xf32>) -> tensor<1x8x128xf32>
    %202 = stablehlo.reshape %201 : (tensor<1x8x128xf32>) -> tensor<8x128xf32>
    %203 = stablehlo.slice %198 [0:1, 0:8, 0:128] : (tensor<2x8x128xf32>) -> tensor<1x8x128xf32>
    %204 = stablehlo.reshape %203 : (tensor<1x8x128xf32>) -> tensor<8x128xf32>
    %205 = call @_where_14(%200, %202, %204) : (tensor<i1>, tensor<8x128xf32>, tensor<8x128xf32>) -> tensor<8x128xf32>
    %206 = stablehlo.broadcast_in_dim %c_2, dims = [] : (tensor<i32>) -> tensor<1xi32>
    %207 = "stablehlo.scatter"(%198, %206, %205) <{indices_are_sorted = true, scatter_dimension_numbers = #stablehlo.scatter<update_window_dims = [0, 1], inserted_window_dims = [0], scatter_dims_to_operand_dims = [0]>, unique_indices = true}> ({
    ^bb0(%arg8: tensor<f32>, %arg9: tensor<f32>):
      stablehlo.return %arg9 : tensor<f32>
    }) : (tensor<2x8x128xf32>, tensor<1xi32>, tensor<8x128xf32>) -> tensor<2x8x128xf32>
    %208 = stablehlo.dot_general %207, %arg2, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<2x8x128xf32>, tensor<2x128x128xf32>) -> tensor<2x8x128xf32>
    %209 = stablehlo.broadcast_in_dim %arg3, dims = [0, 2] : (tensor<2x128xf32>) -> tensor<2x1x128xf32>
    %210 = stablehlo.broadcast_in_dim %209, dims = [0, 1, 2] : (tensor<2x1x128xf32>) -> tensor<2x8x128xf32>
    %211 = stablehlo.add %208, %210 : tensor<2x8x128xf32>
    %212 = call @relu_15(%211) : (tensor<2x8x128xf32>) -> tensor<2x8x128xf32>
    %213 = stablehlo.compare  EQ, %9, %c_3,  SIGNED : (tensor<i32>, tensor<i32>) -> tensor<i1>
    %214 = stablehlo.compare  LT, %c_4, %c_5,  SIGNED : (tensor<i32>, tensor<i32>) -> tensor<i1>
    %215 = stablehlo.add %c_6, %c_7 : tensor<i32>
    %216 = stablehlo.select %214, %215, %c_4 : tensor<i1>, tensor<i32>
    %217 = stablehlo.compare  LT, %c_5, %c_5,  SIGNED : (tensor<i32>, tensor<i32>) -> tensor<i1>
    %218 = stablehlo.add %c_2, %c_8 : tensor<i32>
    %219 = stablehlo.select %217, %218, %c_5 : tensor<i1>, tensor<i32>
    %220 = stablehlo.compare  LT, %c_5, %c_5,  SIGNED : (tensor<i32>, tensor<i32>) -> tensor<i1>
    %221 = stablehlo.add %c_2, %c_9 : tensor<i32>
    %222 = stablehlo.select %220, %221, %c_5 : tensor<i1>, tensor<i32>
    %223 = stablehlo.dynamic_slice %212, %216, %219, %222, sizes = [1, 8, 128] : (tensor<2x8x128xf32>, tensor<i32>, tensor<i32>, tensor<i32>) -> tensor<1x8x128xf32>
    %224 = stablehlo.reshape %223 : (tensor<1x8x128xf32>) -> tensor<8x128xf32>
    %225 = stablehlo.slice %183 [1:2, 0:8, 0:128] : (tensor<2x8x128xf32>) -> tensor<1x8x128xf32>
    %226 = stablehlo.reshape %225 : (tensor<1x8x128xf32>) -> tensor<8x128xf32>
    %227 = call @_where_16(%213, %224, %226) : (tensor<i1>, tensor<8x128xf32>, tensor<8x128xf32>) -> tensor<8x128xf32>
    %228 = stablehlo.broadcast_in_dim %c_3, dims = [] : (tensor<i32>) -> tensor<1xi32>
    %229 = "stablehlo.scatter"(%183, %228, %227) <{indices_are_sorted = true, scatter_dimension_numbers = #stablehlo.scatter<update_window_dims = [0, 1], inserted_window_dims = [0], scatter_dims_to_operand_dims = [0]>, unique_indices = true}> ({
    ^bb0(%arg8: tensor<f32>, %arg9: tensor<f32>):
      stablehlo.return %arg9 : tensor<f32>
    }) : (tensor<2x8x128xf32>, tensor<1xi32>, tensor<8x128xf32>) -> tensor<2x8x128xf32>
    %230 = call @_roll_static_17(%212) : (tensor<2x8x128xf32>) -> tensor<2x8x128xf32>
    %231 = stablehlo.compare  LT, %c_4, %c_5,  SIGNED : (tensor<i32>, tensor<i32>) -> tensor<i1>
    %232 = stablehlo.add %c_6, %c_7 : tensor<i32>
    %233 = stablehlo.select %231, %232, %c_4 : tensor<i1>, tensor<i32>
    %234 = stablehlo.compare  LT, %c_5, %c_5,  SIGNED : (tensor<i32>, tensor<i32>) -> tensor<i1>
    %235 = stablehlo.add %c_2, %c_8 : tensor<i32>
    %236 = stablehlo.select %234, %235, %c_5 : tensor<i1>, tensor<i32>
    %237 = stablehlo.compare  LT, %c_5, %c_5,  SIGNED : (tensor<i32>, tensor<i32>) -> tensor<i1>
    %238 = stablehlo.add %c_2, %c_9 : tensor<i32>
    %239 = stablehlo.select %237, %238, %c_5 : tensor<i1>, tensor<i32>
    %240 = stablehlo.dynamic_slice %212, %233, %236, %239, sizes = [1, 8, 128] : (tensor<2x8x128xf32>, tensor<i32>, tensor<i32>, tensor<i32>) -> tensor<1x8x128xf32>
    %241 = stablehlo.reshape %240 : (tensor<1x8x128xf32>) -> tensor<8x128xf32>
    %242 = "stablehlo.collective_permute"(%241) <{channel_handle = #stablehlo.channel_handle<handle = 9, type = 1>, source_target_pairs = dense<[[0, 4], [4, 0], [1, 5], [5, 1], [2, 6], [6, 2], [3, 7], [7, 3]]> : tensor<8x2xi64>}> : (tensor<8x128xf32>) -> tensor<8x128xf32>
    %243 = stablehlo.broadcast_in_dim %c_2, dims = [] : (tensor<i32>) -> tensor<1xi32>
    %244 = "stablehlo.scatter"(%230, %243, %242) <{indices_are_sorted = true, scatter_dimension_numbers = #stablehlo.scatter<update_window_dims = [0, 1], inserted_window_dims = [0], scatter_dims_to_operand_dims = [0]>, unique_indices = true}> ({
    ^bb0(%arg8: tensor<f32>, %arg9: tensor<f32>):
      stablehlo.return %arg9 : tensor<f32>
    }) : (tensor<2x8x128xf32>, tensor<1xi32>, tensor<8x128xf32>) -> tensor<2x8x128xf32>
    %245 = "stablehlo.collective_permute"(%229) <{channel_handle = #stablehlo.channel_handle<handle = 10, type = 1>, source_target_pairs = dense<[[0, 4], [4, 0], [1, 5], [5, 1], [2, 6], [6, 2], [3, 7], [7, 3]]> : tensor<8x2xi64>}> : (tensor<2x8x128xf32>) -> tensor<2x8x128xf32>
    %246 = stablehlo.compare  EQ, %9, %c_2,  SIGNED : (tensor<i32>, tensor<i32>) -> tensor<i1>
    %247 = stablehlo.slice %199 [1:2, 0:8, 0:128] : (tensor<2x8x128xf32>) -> tensor<1x8x128xf32>
    %248 = stablehlo.reshape %247 : (tensor<1x8x128xf32>) -> tensor<8x128xf32>
    %249 = stablehlo.slice %244 [0:1, 0:8, 0:128] : (tensor<2x8x128xf32>) -> tensor<1x8x128xf32>
    %250 = stablehlo.reshape %249 : (tensor<1x8x128xf32>) -> tensor<8x128xf32>
    %251 = call @_where_18(%246, %248, %250) : (tensor<i1>, tensor<8x128xf32>, tensor<8x128xf32>) -> tensor<8x128xf32>
    %252 = stablehlo.broadcast_in_dim %c_2, dims = [] : (tensor<i32>) -> tensor<1xi32>
    %253 = "stablehlo.scatter"(%244, %252, %251) <{indices_are_sorted = true, scatter_dimension_numbers = #stablehlo.scatter<update_window_dims = [0, 1], inserted_window_dims = [0], scatter_dims_to_operand_dims = [0]>, unique_indices = true}> ({
    ^bb0(%arg8: tensor<f32>, %arg9: tensor<f32>):
      stablehlo.return %arg9 : tensor<f32>
    }) : (tensor<2x8x128xf32>, tensor<1xi32>, tensor<8x128xf32>) -> tensor<2x8x128xf32>
    %254 = stablehlo.dot_general %253, %arg2, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<2x8x128xf32>, tensor<2x128x128xf32>) -> tensor<2x8x128xf32>
    %255 = stablehlo.broadcast_in_dim %arg3, dims = [0, 2] : (tensor<2x128xf32>) -> tensor<2x1x128xf32>
    %256 = stablehlo.broadcast_in_dim %255, dims = [0, 1, 2] : (tensor<2x1x128xf32>) -> tensor<2x8x128xf32>
    %257 = stablehlo.add %254, %256 : tensor<2x8x128xf32>
    %258 = call @relu_19(%257) : (tensor<2x8x128xf32>) -> tensor<2x8x128xf32>
    %259 = stablehlo.compare  EQ, %9, %c_3,  SIGNED : (tensor<i32>, tensor<i32>) -> tensor<i1>
    %260 = stablehlo.compare  LT, %c_4, %c_5,  SIGNED : (tensor<i32>, tensor<i32>) -> tensor<i1>
    %261 = stablehlo.add %c_6, %c_7 : tensor<i32>
    %262 = stablehlo.select %260, %261, %c_4 : tensor<i1>, tensor<i32>
    %263 = stablehlo.compare  LT, %c_5, %c_5,  SIGNED : (tensor<i32>, tensor<i32>) -> tensor<i1>
    %264 = stablehlo.add %c_2, %c_8 : tensor<i32>
    %265 = stablehlo.select %263, %264, %c_5 : tensor<i1>, tensor<i32>
    %266 = stablehlo.compare  LT, %c_5, %c_5,  SIGNED : (tensor<i32>, tensor<i32>) -> tensor<i1>
    %267 = stablehlo.add %c_2, %c_9 : tensor<i32>
    %268 = stablehlo.select %266, %267, %c_5 : tensor<i1>, tensor<i32>
    %269 = stablehlo.dynamic_slice %258, %262, %265, %268, sizes = [1, 8, 128] : (tensor<2x8x128xf32>, tensor<i32>, tensor<i32>, tensor<i32>) -> tensor<1x8x128xf32>
    %270 = stablehlo.reshape %269 : (tensor<1x8x128xf32>) -> tensor<8x128xf32>
    %271 = stablehlo.slice %245 [0:1, 0:8, 0:128] : (tensor<2x8x128xf32>) -> tensor<1x8x128xf32>
    %272 = stablehlo.reshape %271 : (tensor<1x8x128xf32>) -> tensor<8x128xf32>
    %273 = call @_where_20(%259, %270, %272) : (tensor<i1>, tensor<8x128xf32>, tensor<8x128xf32>) -> tensor<8x128xf32>
    %274 = stablehlo.broadcast_in_dim %c_2, dims = [] : (tensor<i32>) -> tensor<1xi32>
    %275 = "stablehlo.scatter"(%245, %274, %273) <{indices_are_sorted = true, scatter_dimension_numbers = #stablehlo.scatter<update_window_dims = [0, 1], inserted_window_dims = [0], scatter_dims_to_operand_dims = [0]>, unique_indices = true}> ({
    ^bb0(%arg8: tensor<f32>, %arg9: tensor<f32>):
      stablehlo.return %arg9 : tensor<f32>
    }) : (tensor<2x8x128xf32>, tensor<1xi32>, tensor<8x128xf32>) -> tensor<2x8x128xf32>
    %276 = call @_roll_static_21(%258) : (tensor<2x8x128xf32>) -> tensor<2x8x128xf32>
    %277 = stablehlo.compare  LT, %c_4, %c_5,  SIGNED : (tensor<i32>, tensor<i32>) -> tensor<i1>
    %278 = stablehlo.add %c_6, %c_7 : tensor<i32>
    %279 = stablehlo.select %277, %278, %c_4 : tensor<i1>, tensor<i32>
    %280 = stablehlo.compare  LT, %c_5, %c_5,  SIGNED : (tensor<i32>, tensor<i32>) -> tensor<i1>
    %281 = stablehlo.add %c_2, %c_8 : tensor<i32>
    %282 = stablehlo.select %280, %281, %c_5 : tensor<i1>, tensor<i32>
    %283 = stablehlo.compare  LT, %c_5, %c_5,  SIGNED : (tensor<i32>, tensor<i32>) -> tensor<i1>
    %284 = stablehlo.add %c_2, %c_9 : tensor<i32>
    %285 = stablehlo.select %283, %284, %c_5 : tensor<i1>, tensor<i32>
    %286 = stablehlo.dynamic_slice %258, %279, %282, %285, sizes = [1, 8, 128] : (tensor<2x8x128xf32>, tensor<i32>, tensor<i32>, tensor<i32>) -> tensor<1x8x128xf32>
    %287 = stablehlo.reshape %286 : (tensor<1x8x128xf32>) -> tensor<8x128xf32>
    %288 = "stablehlo.collective_permute"(%287) <{channel_handle = #stablehlo.channel_handle<handle = 11, type = 1>, source_target_pairs = dense<[[0, 4], [4, 0], [1, 5], [5, 1], [2, 6], [6, 2], [3, 7], [7, 3]]> : tensor<8x2xi64>}> : (tensor<8x128xf32>) -> tensor<8x128xf32>
    %289 = stablehlo.broadcast_in_dim %c_2, dims = [] : (tensor<i32>) -> tensor<1xi32>
    %290 = "stablehlo.scatter"(%276, %289, %288) <{indices_are_sorted = true, scatter_dimension_numbers = #stablehlo.scatter<update_window_dims = [0, 1], inserted_window_dims = [0], scatter_dims_to_operand_dims = [0]>, unique_indices = true}> ({
    ^bb0(%arg8: tensor<f32>, %arg9: tensor<f32>):
      stablehlo.return %arg9 : tensor<f32>
    }) : (tensor<2x8x128xf32>, tensor<1xi32>, tensor<8x128xf32>) -> tensor<2x8x128xf32>
    %291 = "stablehlo.collective_permute"(%199) <{channel_handle = #stablehlo.channel_handle<handle = 12, type = 1>, source_target_pairs = dense<[[0, 4], [4, 0], [1, 5], [5, 1], [2, 6], [6, 2], [3, 7], [7, 3]]> : tensor<8x2xi64>}> : (tensor<2x8x128xf32>) -> tensor<2x8x128xf32>
    %292 = stablehlo.compare  EQ, %9, %c_2,  SIGNED : (tensor<i32>, tensor<i32>) -> tensor<i1>
    %293 = stablehlo.slice %291 [0:1, 0:8, 0:128] : (tensor<2x8x128xf32>) -> tensor<1x8x128xf32>
    %294 = stablehlo.reshape %293 : (tensor<1x8x128xf32>) -> tensor<8x128xf32>
    %295 = stablehlo.slice %290 [0:1, 0:8, 0:128] : (tensor<2x8x128xf32>) -> tensor<1x8x128xf32>
    %296 = stablehlo.reshape %295 : (tensor<1x8x128xf32>) -> tensor<8x128xf32>
    %297 = call @_where_22(%292, %294, %296) : (tensor<i1>, tensor<8x128xf32>, tensor<8x128xf32>) -> tensor<8x128xf32>
    %298 = stablehlo.broadcast_in_dim %c_2, dims = [] : (tensor<i32>) -> tensor<1xi32>
    %299 = "stablehlo.scatter"(%290, %298, %297) <{indices_are_sorted = true, scatter_dimension_numbers = #stablehlo.scatter<update_window_dims = [0, 1], inserted_window_dims = [0], scatter_dims_to_operand_dims = [0]>, unique_indices = true}> ({
    ^bb0(%arg8: tensor<f32>, %arg9: tensor<f32>):
      stablehlo.return %arg9 : tensor<f32>
    }) : (tensor<2x8x128xf32>, tensor<1xi32>, tensor<8x128xf32>) -> tensor<2x8x128xf32>
    %300 = stablehlo.dot_general %299, %arg2, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<2x8x128xf32>, tensor<2x128x128xf32>) -> tensor<2x8x128xf32>
    %301 = stablehlo.broadcast_in_dim %arg3, dims = [0, 2] : (tensor<2x128xf32>) -> tensor<2x1x128xf32>
    %302 = stablehlo.broadcast_in_dim %301, dims = [0, 1, 2] : (tensor<2x1x128xf32>) -> tensor<2x8x128xf32>
    %303 = stablehlo.add %300, %302 : tensor<2x8x128xf32>
    %304 = call @relu_23(%303) : (tensor<2x8x128xf32>) -> tensor<2x8x128xf32>
    %305 = stablehlo.compare  EQ, %9, %c_3,  SIGNED : (tensor<i32>, tensor<i32>) -> tensor<i1>
    %306 = stablehlo.compare  LT, %c_4, %c_5,  SIGNED : (tensor<i32>, tensor<i32>) -> tensor<i1>
    %307 = stablehlo.add %c_6, %c_7 : tensor<i32>
    %308 = stablehlo.select %306, %307, %c_4 : tensor<i1>, tensor<i32>
    %309 = stablehlo.compare  LT, %c_5, %c_5,  SIGNED : (tensor<i32>, tensor<i32>) -> tensor<i1>
    %310 = stablehlo.add %c_2, %c_8 : tensor<i32>
    %311 = stablehlo.select %309, %310, %c_5 : tensor<i1>, tensor<i32>
    %312 = stablehlo.compare  LT, %c_5, %c_5,  SIGNED : (tensor<i32>, tensor<i32>) -> tensor<i1>
    %313 = stablehlo.add %c_2, %c_9 : tensor<i32>
    %314 = stablehlo.select %312, %313, %c_5 : tensor<i1>, tensor<i32>
    %315 = stablehlo.dynamic_slice %304, %308, %311, %314, sizes = [1, 8, 128] : (tensor<2x8x128xf32>, tensor<i32>, tensor<i32>, tensor<i32>) -> tensor<1x8x128xf32>
    %316 = stablehlo.reshape %315 : (tensor<1x8x128xf32>) -> tensor<8x128xf32>
    %317 = stablehlo.slice %275 [1:2, 0:8, 0:128] : (tensor<2x8x128xf32>) -> tensor<1x8x128xf32>
    %318 = stablehlo.reshape %317 : (tensor<1x8x128xf32>) -> tensor<8x128xf32>
    %319 = call @_where_24(%305, %316, %318) : (tensor<i1>, tensor<8x128xf32>, tensor<8x128xf32>) -> tensor<8x128xf32>
    %320 = stablehlo.broadcast_in_dim %c_3, dims = [] : (tensor<i32>) -> tensor<1xi32>
    %321 = "stablehlo.scatter"(%275, %320, %319) <{indices_are_sorted = true, scatter_dimension_numbers = #stablehlo.scatter<update_window_dims = [0, 1], inserted_window_dims = [0], scatter_dims_to_operand_dims = [0]>, unique_indices = true}> ({
    ^bb0(%arg8: tensor<f32>, %arg9: tensor<f32>):
      stablehlo.return %arg9 : tensor<f32>
    }) : (tensor<2x8x128xf32>, tensor<1xi32>, tensor<8x128xf32>) -> tensor<2x8x128xf32>
    %322 = "stablehlo.collective_permute"(%321) <{channel_handle = #stablehlo.channel_handle<handle = 13, type = 1>, source_target_pairs = dense<[[0, 4], [4, 0], [1, 5], [5, 1], [2, 6], [6, 2], [3, 7], [7, 3]]> : tensor<8x2xi64>}> : (tensor<2x8x128xf32>) -> tensor<2x8x128xf32>
    %323 = "stablehlo.collective_permute"(%322) <{channel_handle = #stablehlo.channel_handle<handle = 14, type = 1>, source_target_pairs = dense<[[0, 4], [4, 0], [1, 5], [5, 1], [2, 6], [6, 2], [3, 7], [7, 3]]> : tensor<8x2xi64>}> : (tensor<2x8x128xf32>) -> tensor<2x8x128xf32>
    %324 = stablehlo.dot_general %323, %arg4, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<2x8x128xf32>, tensor<128x8xf32>) -> tensor<2x8x8xf32>
    %325 = stablehlo.broadcast_in_dim %arg5, dims = [2] : (tensor<8xf32>) -> tensor<1x1x8xf32>
    %326 = stablehlo.broadcast_in_dim %325, dims = [0, 1, 2] : (tensor<1x1x8xf32>) -> tensor<2x8x8xf32>
    %327 = stablehlo.add %324, %326 : tensor<2x8x8xf32>
    %328 = stablehlo.reshape %327 : (tensor<2x8x8xf32>) -> tensor<16x8xf32>
    %329 = stablehlo.subtract %328, %arg7 : tensor<16x8xf32>
    %330 = stablehlo.multiply %329, %329 : tensor<16x8xf32>
    %cst_10 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %331 = stablehlo.reduce(%330 init: %cst_10) applies stablehlo.add across dimensions = [1] : (tensor<16x8xf32>, tensor<f32>) -> tensor<16xf32>
    %cst_11 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %332 = stablehlo.reduce(%331 init: %cst_11) applies stablehlo.add across dimensions = [0] : (tensor<16xf32>, tensor<f32>) -> tensor<f32>
    %cst_12 = stablehlo.constant dense<1.600000e+01> : tensor<f32>
    %333 = stablehlo.divide %332, %cst_12 : tensor<f32>
    %334 = "stablehlo.all_reduce"(%333) <{channel_handle = #stablehlo.channel_handle<handle = 15, type = 1>, replica_groups = dense<[[0, 4], [1, 5], [2, 6], [3, 7]]> : tensor<4x2xi64>, use_global_device_ids}> ({
    ^bb0(%arg8: tensor<f32>, %arg9: tensor<f32>):
      %336 = stablehlo.add %arg8, %arg9 : tensor<f32>
      stablehlo.return %336 : tensor<f32>
    }) : (tensor<f32>) -> tensor<f32>
    %cst_13 = stablehlo.constant dense<2.000000e+00> : tensor<f32>
    %335 = stablehlo.divide %334, %cst_13 : tensor<f32>
    return %335 : tensor<f32>
  }
  func.func private @relu(%arg0: tensor<2x8x128xf32>) -> tensor<2x8x128xf32> {
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<2x8x128xf32>
    %1 = stablehlo.maximum %arg0, %0 : tensor<2x8x128xf32>
    return %1 : tensor<2x8x128xf32>
  }
  func.func private @_where(%arg0: tensor<i1>, %arg1: tensor<8x128xf32>, %arg2: tensor<8x128xf32>) -> tensor<8x128xf32> {
    %0 = stablehlo.select %arg0, %arg1, %arg2 : tensor<i1>, tensor<8x128xf32>
    return %0 : tensor<8x128xf32>
  }
  func.func private @relu_0(%arg0: tensor<2x8x128xf32>) -> tensor<2x8x128xf32> {
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<2x8x128xf32>
    %1 = stablehlo.maximum %arg0, %0 : tensor<2x8x128xf32>
    return %1 : tensor<2x8x128xf32>
  }
  func.func private @_where_1(%arg0: tensor<i1>, %arg1: tensor<8x128xf32>, %arg2: tensor<8x128xf32>) -> tensor<8x128xf32> {
    %0 = stablehlo.select %arg0, %arg1, %arg2 : tensor<i1>, tensor<8x128xf32>
    return %0 : tensor<8x128xf32>
  }
  func.func private @_roll_static(%arg0: tensor<2x8x128xf32>) -> tensor<2x8x128xf32> {
    %0 = stablehlo.slice %arg0 [1:2, 0:8, 0:128] : (tensor<2x8x128xf32>) -> tensor<1x8x128xf32>
    %1 = stablehlo.slice %arg0 [0:1, 0:8, 0:128] : (tensor<2x8x128xf32>) -> tensor<1x8x128xf32>
    %2 = stablehlo.concatenate %0, %1, dim = 0 : (tensor<1x8x128xf32>, tensor<1x8x128xf32>) -> tensor<2x8x128xf32>
    return %2 : tensor<2x8x128xf32>
  }
  func.func private @_where_2(%arg0: tensor<i1>, %arg1: tensor<8x128xf32>, %arg2: tensor<8x128xf32>) -> tensor<8x128xf32> {
    %0 = stablehlo.select %arg0, %arg1, %arg2 : tensor<i1>, tensor<8x128xf32>
    return %0 : tensor<8x128xf32>
  }
  func.func private @relu_3(%arg0: tensor<2x8x128xf32>) -> tensor<2x8x128xf32> {
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<2x8x128xf32>
    %1 = stablehlo.maximum %arg0, %0 : tensor<2x8x128xf32>
    return %1 : tensor<2x8x128xf32>
  }
  func.func private @_where_4(%arg0: tensor<i1>, %arg1: tensor<8x128xf32>, %arg2: tensor<8x128xf32>) -> tensor<8x128xf32> {
    %0 = stablehlo.select %arg0, %arg1, %arg2 : tensor<i1>, tensor<8x128xf32>
    return %0 : tensor<8x128xf32>
  }
  func.func private @_roll_static_5(%arg0: tensor<2x8x128xf32>) -> tensor<2x8x128xf32> {
    %0 = stablehlo.slice %arg0 [1:2, 0:8, 0:128] : (tensor<2x8x128xf32>) -> tensor<1x8x128xf32>
    %1 = stablehlo.slice %arg0 [0:1, 0:8, 0:128] : (tensor<2x8x128xf32>) -> tensor<1x8x128xf32>
    %2 = stablehlo.concatenate %0, %1, dim = 0 : (tensor<1x8x128xf32>, tensor<1x8x128xf32>) -> tensor<2x8x128xf32>
    return %2 : tensor<2x8x128xf32>
  }
  func.func private @_where_6(%arg0: tensor<i1>, %arg1: tensor<8x128xf32>, %arg2: tensor<8x128xf32>) -> tensor<8x128xf32> {
    %0 = stablehlo.select %arg0, %arg1, %arg2 : tensor<i1>, tensor<8x128xf32>
    return %0 : tensor<8x128xf32>
  }
  func.func private @relu_7(%arg0: tensor<2x8x128xf32>) -> tensor<2x8x128xf32> {
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<2x8x128xf32>
    %1 = stablehlo.maximum %arg0, %0 : tensor<2x8x128xf32>
    return %1 : tensor<2x8x128xf32>
  }
  func.func private @_where_8(%arg0: tensor<i1>, %arg1: tensor<8x128xf32>, %arg2: tensor<8x128xf32>) -> tensor<8x128xf32> {
    %0 = stablehlo.select %arg0, %arg1, %arg2 : tensor<i1>, tensor<8x128xf32>
    return %0 : tensor<8x128xf32>
  }
  func.func private @_roll_static_9(%arg0: tensor<2x8x128xf32>) -> tensor<2x8x128xf32> {
    %0 = stablehlo.slice %arg0 [1:2, 0:8, 0:128] : (tensor<2x8x128xf32>) -> tensor<1x8x128xf32>
    %1 = stablehlo.slice %arg0 [0:1, 0:8, 0:128] : (tensor<2x8x128xf32>) -> tensor<1x8x128xf32>
    %2 = stablehlo.concatenate %0, %1, dim = 0 : (tensor<1x8x128xf32>, tensor<1x8x128xf32>) -> tensor<2x8x128xf32>
    return %2 : tensor<2x8x128xf32>
  }
  func.func private @_where_10(%arg0: tensor<i1>, %arg1: tensor<8x128xf32>, %arg2: tensor<8x128xf32>) -> tensor<8x128xf32> {
    %0 = stablehlo.select %arg0, %arg1, %arg2 : tensor<i1>, tensor<8x128xf32>
    return %0 : tensor<8x128xf32>
  }
  func.func private @relu_11(%arg0: tensor<2x8x128xf32>) -> tensor<2x8x128xf32> {
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<2x8x128xf32>
    %1 = stablehlo.maximum %arg0, %0 : tensor<2x8x128xf32>
    return %1 : tensor<2x8x128xf32>
  }
  func.func private @_where_12(%arg0: tensor<i1>, %arg1: tensor<8x128xf32>, %arg2: tensor<8x128xf32>) -> tensor<8x128xf32> {
    %0 = stablehlo.select %arg0, %arg1, %arg2 : tensor<i1>, tensor<8x128xf32>
    return %0 : tensor<8x128xf32>
  }
  func.func private @_roll_static_13(%arg0: tensor<2x8x128xf32>) -> tensor<2x8x128xf32> {
    %0 = stablehlo.slice %arg0 [1:2, 0:8, 0:128] : (tensor<2x8x128xf32>) -> tensor<1x8x128xf32>
    %1 = stablehlo.slice %arg0 [0:1, 0:8, 0:128] : (tensor<2x8x128xf32>) -> tensor<1x8x128xf32>
    %2 = stablehlo.concatenate %0, %1, dim = 0 : (tensor<1x8x128xf32>, tensor<1x8x128xf32>) -> tensor<2x8x128xf32>
    return %2 : tensor<2x8x128xf32>
  }
  func.func private @_where_14(%arg0: tensor<i1>, %arg1: tensor<8x128xf32>, %arg2: tensor<8x128xf32>) -> tensor<8x128xf32> {
    %0 = stablehlo.select %arg0, %arg1, %arg2 : tensor<i1>, tensor<8x128xf32>
    return %0 : tensor<8x128xf32>
  }
  func.func private @relu_15(%arg0: tensor<2x8x128xf32>) -> tensor<2x8x128xf32> {
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<2x8x128xf32>
    %1 = stablehlo.maximum %arg0, %0 : tensor<2x8x128xf32>
    return %1 : tensor<2x8x128xf32>
  }
  func.func private @_where_16(%arg0: tensor<i1>, %arg1: tensor<8x128xf32>, %arg2: tensor<8x128xf32>) -> tensor<8x128xf32> {
    %0 = stablehlo.select %arg0, %arg1, %arg2 : tensor<i1>, tensor<8x128xf32>
    return %0 : tensor<8x128xf32>
  }
  func.func private @_roll_static_17(%arg0: tensor<2x8x128xf32>) -> tensor<2x8x128xf32> {
    %0 = stablehlo.slice %arg0 [1:2, 0:8, 0:128] : (tensor<2x8x128xf32>) -> tensor<1x8x128xf32>
    %1 = stablehlo.slice %arg0 [0:1, 0:8, 0:128] : (tensor<2x8x128xf32>) -> tensor<1x8x128xf32>
    %2 = stablehlo.concatenate %0, %1, dim = 0 : (tensor<1x8x128xf32>, tensor<1x8x128xf32>) -> tensor<2x8x128xf32>
    return %2 : tensor<2x8x128xf32>
  }
  func.func private @_where_18(%arg0: tensor<i1>, %arg1: tensor<8x128xf32>, %arg2: tensor<8x128xf32>) -> tensor<8x128xf32> {
    %0 = stablehlo.select %arg0, %arg1, %arg2 : tensor<i1>, tensor<8x128xf32>
    return %0 : tensor<8x128xf32>
  }
  func.func private @relu_19(%arg0: tensor<2x8x128xf32>) -> tensor<2x8x128xf32> {
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<2x8x128xf32>
    %1 = stablehlo.maximum %arg0, %0 : tensor<2x8x128xf32>
    return %1 : tensor<2x8x128xf32>
  }
  func.func private @_where_20(%arg0: tensor<i1>, %arg1: tensor<8x128xf32>, %arg2: tensor<8x128xf32>) -> tensor<8x128xf32> {
    %0 = stablehlo.select %arg0, %arg1, %arg2 : tensor<i1>, tensor<8x128xf32>
    return %0 : tensor<8x128xf32>
  }
  func.func private @_roll_static_21(%arg0: tensor<2x8x128xf32>) -> tensor<2x8x128xf32> {
    %0 = stablehlo.slice %arg0 [1:2, 0:8, 0:128] : (tensor<2x8x128xf32>) -> tensor<1x8x128xf32>
    %1 = stablehlo.slice %arg0 [0:1, 0:8, 0:128] : (tensor<2x8x128xf32>) -> tensor<1x8x128xf32>
    %2 = stablehlo.concatenate %0, %1, dim = 0 : (tensor<1x8x128xf32>, tensor<1x8x128xf32>) -> tensor<2x8x128xf32>
    return %2 : tensor<2x8x128xf32>
  }
  func.func private @_where_22(%arg0: tensor<i1>, %arg1: tensor<8x128xf32>, %arg2: tensor<8x128xf32>) -> tensor<8x128xf32> {
    %0 = stablehlo.select %arg0, %arg1, %arg2 : tensor<i1>, tensor<8x128xf32>
    return %0 : tensor<8x128xf32>
  }
  func.func private @relu_23(%arg0: tensor<2x8x128xf32>) -> tensor<2x8x128xf32> {
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<2x8x128xf32>
    %1 = stablehlo.maximum %arg0, %0 : tensor<2x8x128xf32>
    return %1 : tensor<2x8x128xf32>
  }
  func.func private @_where_24(%arg0: tensor<i1>, %arg1: tensor<8x128xf32>, %arg2: tensor<8x128xf32>) -> tensor<8x128xf32> {
    %0 = stablehlo.select %arg0, %arg1, %arg2 : tensor<i1>, tensor<8x128xf32>
    return %0 : tensor<8x128xf32>
  }
}

// CHECK-LABEL @main
