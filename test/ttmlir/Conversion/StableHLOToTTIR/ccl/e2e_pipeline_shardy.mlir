// REQUIRES: stablehlo
// RUN: ttmlir-opt -split-input-file --stablehlo-to-ttir-pipeline -o %t %s
// RUN: FileCheck %s --input-file=%t
// UNSUPPORTED: true

module @jit_loss_pp attributes {mhlo.num_partitions = 8 : i32, mhlo.num_replicas = 1 : i32} {
  sdy.mesh @mesh = <["stages"=2, "y"=4]>
  func.func public @main(%arg0: tensor<784x128xf32> {sdy.sharding = #sdy.sharding<@mesh, [{}, {}]>}, %arg1: tensor<128xf32> {sdy.sharding = #sdy.sharding<@mesh, [{}]>}, %arg2: tensor<4x128x128xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"stages"}, {}, {}]>}, %arg3: tensor<4x128xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"stages"}, {}]>}, %arg4: tensor<128x8xf32> {sdy.sharding = #sdy.sharding<@mesh, [{}, {}]>}, %arg5: tensor<8xf32> {sdy.sharding = #sdy.sharding<@mesh, [{}]>}, %arg6: tensor<32x784xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"stages"}, {}]>}, %arg7: tensor<32x8xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"stages"}, {}]>}) -> (tensor<f32> {jax.result_info = ""}) {
    %0 = sdy.manual_computation(%arg0, %arg1, %arg2, %arg3, %arg4, %arg5, %arg6, %arg7) in_shardings=[<@mesh, [{}, {}]>, <@mesh, [{}]>, <@mesh, [{"stages"}, {}, {}]>, <@mesh, [{"stages"}, {}]>, <@mesh, [{}, {}]>, <@mesh, [{}]>, <@mesh, [{"stages"}, {}]>, <@mesh, [{"stages"}, {}]>] out_shardings=[<@mesh, []>] manual_axes={"stages", "y"} (%arg8: tensor<784x128xf32>, %arg9: tensor<128xf32>, %arg10: tensor<2x128x128xf32>, %arg11: tensor<2x128xf32>, %arg12: tensor<128x8xf32>, %arg13: tensor<8xf32>, %arg14: tensor<16x784xf32>, %arg15: tensor<16x8xf32>) {
      %1 = stablehlo.reshape %arg14 : (tensor<16x784xf32>) -> tensor<2x8x784xf32>
      %2 = stablehlo.dot_general %1, %arg8, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<2x8x784xf32>, tensor<784x128xf32>) -> tensor<2x8x128xf32>
      %3 = stablehlo.broadcast_in_dim %arg9, dims = [2] : (tensor<128xf32>) -> tensor<1x1x128xf32>
      %4 = stablehlo.broadcast_in_dim %3, dims = [0, 1, 2] : (tensor<1x1x128xf32>) -> tensor<2x8x128xf32>
      %5 = stablehlo.add %2, %4 : tensor<2x8x128xf32>
      %6 = func.call @relu(%5) : (tensor<2x8x128xf32>) -> tensor<2x8x128xf32>
      %c = stablehlo.constant dense<4> : tensor<ui32>
      %c_0 = stablehlo.constant dense<2> : tensor<ui32>
      %7 = stablehlo.partition_id : tensor<ui32>
      %8 = stablehlo.divide %7, %c : tensor<ui32>
      %9 = stablehlo.remainder %8, %c_0 : tensor<ui32>
      %10 = stablehlo.convert %9 : (tensor<ui32>) -> tensor<i32>
      %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %11 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<2x8x128xf32>
      %cst_1 = stablehlo.constant dense<0x7FC00000> : tensor<f32>
      %12 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<2x8x128xf32>
      %13 = stablehlo.multiply %11, %12 : tensor<2x8x128xf32>
      %14 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<2x8x128xf32>
      %15 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<2x8x128xf32>
      %16 = stablehlo.multiply %14, %15 : tensor<2x8x128xf32>
      %c_2 = stablehlo.constant dense<0> : tensor<i32>
      %17 = stablehlo.compare  EQ, %10, %c_2,  SIGNED : (tensor<i32>, tensor<i32>) -> tensor<i1>
      %18 = stablehlo.slice %6 [0:1, 0:8, 0:128] : (tensor<2x8x128xf32>) -> tensor<1x8x128xf32>
      %19 = stablehlo.reshape %18 : (tensor<1x8x128xf32>) -> tensor<8x128xf32>
      %20 = stablehlo.slice %16 [0:1, 0:8, 0:128] : (tensor<2x8x128xf32>) -> tensor<1x8x128xf32>
      %21 = stablehlo.reshape %20 : (tensor<1x8x128xf32>) -> tensor<8x128xf32>
      %22 = func.call @_where(%17, %19, %21) : (tensor<i1>, tensor<8x128xf32>, tensor<8x128xf32>) -> tensor<8x128xf32>
      %23 = stablehlo.broadcast_in_dim %c_2, dims = [] : (tensor<i32>) -> tensor<1xi32>
      %24 = "stablehlo.scatter"(%16, %23, %22) <{indices_are_sorted = true, scatter_dimension_numbers = #stablehlo.scatter<update_window_dims = [0, 1], inserted_window_dims = [0], scatter_dims_to_operand_dims = [0]>, unique_indices = true}> ({
      ^bb0(%arg16: tensor<f32>, %arg17: tensor<f32>):
        stablehlo.return %arg17 : tensor<f32>
      }) : (tensor<2x8x128xf32>, tensor<1xi32>, tensor<8x128xf32>) -> tensor<2x8x128xf32>
      %25 = stablehlo.dot_general %24, %arg10, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<2x8x128xf32>, tensor<2x128x128xf32>) -> tensor<2x8x128xf32>
      %26 = stablehlo.broadcast_in_dim %arg11, dims = [0, 2] : (tensor<2x128xf32>) -> tensor<2x1x128xf32>
      %27 = stablehlo.broadcast_in_dim %26, dims = [0, 1, 2] : (tensor<2x1x128xf32>) -> tensor<2x8x128xf32>
      %28 = stablehlo.add %25, %27 : tensor<2x8x128xf32>
      %29 = func.call @relu_0(%28) : (tensor<2x8x128xf32>) -> tensor<2x8x128xf32>
      %c_3 = stablehlo.constant dense<1> : tensor<i32>
      %30 = stablehlo.compare  EQ, %10, %c_3,  SIGNED : (tensor<i32>, tensor<i32>) -> tensor<i1>
      %c_4 = stablehlo.constant dense<-1> : tensor<i32>
      %c_5 = stablehlo.constant dense<0> : tensor<i32>
      %31 = stablehlo.compare  LT, %c_4, %c_5,  SIGNED : (tensor<i32>, tensor<i32>) -> tensor<i1>
      %c_6 = stablehlo.constant dense<-1> : tensor<i32>
      %c_7 = stablehlo.constant dense<2> : tensor<i32>
      %32 = stablehlo.add %c_6, %c_7 : tensor<i32>
      %33 = stablehlo.select %31, %32, %c_4 : tensor<i1>, tensor<i32>
      %34 = stablehlo.compare  LT, %c_5, %c_5,  SIGNED : (tensor<i32>, tensor<i32>) -> tensor<i1>
      %c_8 = stablehlo.constant dense<8> : tensor<i32>
      %35 = stablehlo.add %c_2, %c_8 : tensor<i32>
      %36 = stablehlo.select %34, %35, %c_5 : tensor<i1>, tensor<i32>
      %37 = stablehlo.compare  LT, %c_5, %c_5,  SIGNED : (tensor<i32>, tensor<i32>) -> tensor<i1>
      %c_9 = stablehlo.constant dense<128> : tensor<i32>
      %38 = stablehlo.add %c_2, %c_9 : tensor<i32>
      %39 = stablehlo.select %37, %38, %c_5 : tensor<i1>, tensor<i32>
      %40 = stablehlo.dynamic_slice %29, %33, %36, %39, sizes = [1, 8, 128] : (tensor<2x8x128xf32>, tensor<i32>, tensor<i32>, tensor<i32>) -> tensor<1x8x128xf32>
      %41 = stablehlo.reshape %40 : (tensor<1x8x128xf32>) -> tensor<8x128xf32>
      %42 = stablehlo.slice %13 [1:2, 0:8, 0:128] : (tensor<2x8x128xf32>) -> tensor<1x8x128xf32>
      %43 = stablehlo.reshape %42 : (tensor<1x8x128xf32>) -> tensor<8x128xf32>
      %44 = func.call @_where_1(%30, %41, %43) : (tensor<i1>, tensor<8x128xf32>, tensor<8x128xf32>) -> tensor<8x128xf32>
      %45 = stablehlo.broadcast_in_dim %c_3, dims = [] : (tensor<i32>) -> tensor<1xi32>
      %46 = "stablehlo.scatter"(%13, %45, %44) <{indices_are_sorted = true, scatter_dimension_numbers = #stablehlo.scatter<update_window_dims = [0, 1], inserted_window_dims = [0], scatter_dims_to_operand_dims = [0]>, unique_indices = true}> ({
      ^bb0(%arg16: tensor<f32>, %arg17: tensor<f32>):
        stablehlo.return %arg17 : tensor<f32>
      }) : (tensor<2x8x128xf32>, tensor<1xi32>, tensor<8x128xf32>) -> tensor<2x8x128xf32>
      %47 = func.call @_roll_static(%29) : (tensor<2x8x128xf32>) -> tensor<2x8x128xf32>
      %48 = stablehlo.compare  LT, %c_4, %c_5,  SIGNED : (tensor<i32>, tensor<i32>) -> tensor<i1>
      %49 = stablehlo.add %c_6, %c_7 : tensor<i32>
      %50 = stablehlo.select %48, %49, %c_4 : tensor<i1>, tensor<i32>
      %51 = stablehlo.compare  LT, %c_5, %c_5,  SIGNED : (tensor<i32>, tensor<i32>) -> tensor<i1>
      %52 = stablehlo.add %c_2, %c_8 : tensor<i32>
      %53 = stablehlo.select %51, %52, %c_5 : tensor<i1>, tensor<i32>
      %54 = stablehlo.compare  LT, %c_5, %c_5,  SIGNED : (tensor<i32>, tensor<i32>) -> tensor<i1>
      %55 = stablehlo.add %c_2, %c_9 : tensor<i32>
      %56 = stablehlo.select %54, %55, %c_5 : tensor<i1>, tensor<i32>
      %57 = stablehlo.dynamic_slice %29, %50, %53, %56, sizes = [1, 8, 128] : (tensor<2x8x128xf32>, tensor<i32>, tensor<i32>, tensor<i32>) -> tensor<1x8x128xf32>
      %58 = stablehlo.reshape %57 : (tensor<1x8x128xf32>) -> tensor<8x128xf32>
      %59 = "stablehlo.collective_permute"(%58) <{channel_handle = #stablehlo.channel_handle<handle = 1, type = 1>, source_target_pairs = dense<[[0, 4], [4, 0], [1, 5], [5, 1], [2, 6], [6, 2], [3, 7], [7, 3]]> : tensor<8x2xi64>}> : (tensor<8x128xf32>) -> tensor<8x128xf32>
      %60 = stablehlo.broadcast_in_dim %c_2, dims = [] : (tensor<i32>) -> tensor<1xi32>
      %61 = "stablehlo.scatter"(%47, %60, %59) <{indices_are_sorted = true, scatter_dimension_numbers = #stablehlo.scatter<update_window_dims = [0, 1], inserted_window_dims = [0], scatter_dims_to_operand_dims = [0]>, unique_indices = true}> ({
      ^bb0(%arg16: tensor<f32>, %arg17: tensor<f32>):
        stablehlo.return %arg17 : tensor<f32>
      }) : (tensor<2x8x128xf32>, tensor<1xi32>, tensor<8x128xf32>) -> tensor<2x8x128xf32>
      %62 = "stablehlo.collective_permute"(%46) <{channel_handle = #stablehlo.channel_handle<handle = 2, type = 1>, source_target_pairs = dense<[[0, 4], [4, 0], [1, 5], [5, 1], [2, 6], [6, 2], [3, 7], [7, 3]]> : tensor<8x2xi64>}> : (tensor<2x8x128xf32>) -> tensor<2x8x128xf32>
      %63 = stablehlo.compare  EQ, %10, %c_2,  SIGNED : (tensor<i32>, tensor<i32>) -> tensor<i1>
      %64 = stablehlo.slice %6 [1:2, 0:8, 0:128] : (tensor<2x8x128xf32>) -> tensor<1x8x128xf32>
      %65 = stablehlo.reshape %64 : (tensor<1x8x128xf32>) -> tensor<8x128xf32>
      %66 = stablehlo.slice %61 [0:1, 0:8, 0:128] : (tensor<2x8x128xf32>) -> tensor<1x8x128xf32>
      %67 = stablehlo.reshape %66 : (tensor<1x8x128xf32>) -> tensor<8x128xf32>
      %68 = func.call @_where_2(%63, %65, %67) : (tensor<i1>, tensor<8x128xf32>, tensor<8x128xf32>) -> tensor<8x128xf32>
      %69 = stablehlo.broadcast_in_dim %c_2, dims = [] : (tensor<i32>) -> tensor<1xi32>
      %70 = "stablehlo.scatter"(%61, %69, %68) <{indices_are_sorted = true, scatter_dimension_numbers = #stablehlo.scatter<update_window_dims = [0, 1], inserted_window_dims = [0], scatter_dims_to_operand_dims = [0]>, unique_indices = true}> ({
      ^bb0(%arg16: tensor<f32>, %arg17: tensor<f32>):
        stablehlo.return %arg17 : tensor<f32>
      }) : (tensor<2x8x128xf32>, tensor<1xi32>, tensor<8x128xf32>) -> tensor<2x8x128xf32>
      %71 = stablehlo.dot_general %70, %arg10, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<2x8x128xf32>, tensor<2x128x128xf32>) -> tensor<2x8x128xf32>
      %72 = stablehlo.broadcast_in_dim %arg11, dims = [0, 2] : (tensor<2x128xf32>) -> tensor<2x1x128xf32>
      %73 = stablehlo.broadcast_in_dim %72, dims = [0, 1, 2] : (tensor<2x1x128xf32>) -> tensor<2x8x128xf32>
      %74 = stablehlo.add %71, %73 : tensor<2x8x128xf32>
      %75 = func.call @relu_3(%74) : (tensor<2x8x128xf32>) -> tensor<2x8x128xf32>
      %76 = stablehlo.compare  EQ, %10, %c_3,  SIGNED : (tensor<i32>, tensor<i32>) -> tensor<i1>
      %77 = stablehlo.compare  LT, %c_4, %c_5,  SIGNED : (tensor<i32>, tensor<i32>) -> tensor<i1>
      %78 = stablehlo.add %c_6, %c_7 : tensor<i32>
      %79 = stablehlo.select %77, %78, %c_4 : tensor<i1>, tensor<i32>
      %80 = stablehlo.compare  LT, %c_5, %c_5,  SIGNED : (tensor<i32>, tensor<i32>) -> tensor<i1>
      %81 = stablehlo.add %c_2, %c_8 : tensor<i32>
      %82 = stablehlo.select %80, %81, %c_5 : tensor<i1>, tensor<i32>
      %83 = stablehlo.compare  LT, %c_5, %c_5,  SIGNED : (tensor<i32>, tensor<i32>) -> tensor<i1>
      %84 = stablehlo.add %c_2, %c_9 : tensor<i32>
      %85 = stablehlo.select %83, %84, %c_5 : tensor<i1>, tensor<i32>
      %86 = stablehlo.dynamic_slice %75, %79, %82, %85, sizes = [1, 8, 128] : (tensor<2x8x128xf32>, tensor<i32>, tensor<i32>, tensor<i32>) -> tensor<1x8x128xf32>
      %87 = stablehlo.reshape %86 : (tensor<1x8x128xf32>) -> tensor<8x128xf32>
      %88 = stablehlo.slice %62 [0:1, 0:8, 0:128] : (tensor<2x8x128xf32>) -> tensor<1x8x128xf32>
      %89 = stablehlo.reshape %88 : (tensor<1x8x128xf32>) -> tensor<8x128xf32>
      %90 = func.call @_where_4(%76, %87, %89) : (tensor<i1>, tensor<8x128xf32>, tensor<8x128xf32>) -> tensor<8x128xf32>
      %91 = stablehlo.broadcast_in_dim %c_2, dims = [] : (tensor<i32>) -> tensor<1xi32>
      %92 = "stablehlo.scatter"(%62, %91, %90) <{indices_are_sorted = true, scatter_dimension_numbers = #stablehlo.scatter<update_window_dims = [0, 1], inserted_window_dims = [0], scatter_dims_to_operand_dims = [0]>, unique_indices = true}> ({
      ^bb0(%arg16: tensor<f32>, %arg17: tensor<f32>):
        stablehlo.return %arg17 : tensor<f32>
      }) : (tensor<2x8x128xf32>, tensor<1xi32>, tensor<8x128xf32>) -> tensor<2x8x128xf32>
      %93 = func.call @_roll_static_5(%75) : (tensor<2x8x128xf32>) -> tensor<2x8x128xf32>
      %94 = stablehlo.compare  LT, %c_4, %c_5,  SIGNED : (tensor<i32>, tensor<i32>) -> tensor<i1>
      %95 = stablehlo.add %c_6, %c_7 : tensor<i32>
      %96 = stablehlo.select %94, %95, %c_4 : tensor<i1>, tensor<i32>
      %97 = stablehlo.compare  LT, %c_5, %c_5,  SIGNED : (tensor<i32>, tensor<i32>) -> tensor<i1>
      %98 = stablehlo.add %c_2, %c_8 : tensor<i32>
      %99 = stablehlo.select %97, %98, %c_5 : tensor<i1>, tensor<i32>
      %100 = stablehlo.compare  LT, %c_5, %c_5,  SIGNED : (tensor<i32>, tensor<i32>) -> tensor<i1>
      %101 = stablehlo.add %c_2, %c_9 : tensor<i32>
      %102 = stablehlo.select %100, %101, %c_5 : tensor<i1>, tensor<i32>
      %103 = stablehlo.dynamic_slice %75, %96, %99, %102, sizes = [1, 8, 128] : (tensor<2x8x128xf32>, tensor<i32>, tensor<i32>, tensor<i32>) -> tensor<1x8x128xf32>
      %104 = stablehlo.reshape %103 : (tensor<1x8x128xf32>) -> tensor<8x128xf32>
      %105 = "stablehlo.collective_permute"(%104) <{channel_handle = #stablehlo.channel_handle<handle = 3, type = 1>, source_target_pairs = dense<[[0, 4], [4, 0], [1, 5], [5, 1], [2, 6], [6, 2], [3, 7], [7, 3]]> : tensor<8x2xi64>}> : (tensor<8x128xf32>) -> tensor<8x128xf32>
      %106 = stablehlo.broadcast_in_dim %c_2, dims = [] : (tensor<i32>) -> tensor<1xi32>
      %107 = "stablehlo.scatter"(%93, %106, %105) <{indices_are_sorted = true, scatter_dimension_numbers = #stablehlo.scatter<update_window_dims = [0, 1], inserted_window_dims = [0], scatter_dims_to_operand_dims = [0]>, unique_indices = true}> ({
      ^bb0(%arg16: tensor<f32>, %arg17: tensor<f32>):
        stablehlo.return %arg17 : tensor<f32>
      }) : (tensor<2x8x128xf32>, tensor<1xi32>, tensor<8x128xf32>) -> tensor<2x8x128xf32>
      %108 = "stablehlo.collective_permute"(%6) <{channel_handle = #stablehlo.channel_handle<handle = 4, type = 1>, source_target_pairs = dense<[[0, 4], [4, 0], [1, 5], [5, 1], [2, 6], [6, 2], [3, 7], [7, 3]]> : tensor<8x2xi64>}> : (tensor<2x8x128xf32>) -> tensor<2x8x128xf32>
      %109 = stablehlo.compare  EQ, %10, %c_2,  SIGNED : (tensor<i32>, tensor<i32>) -> tensor<i1>
      %110 = stablehlo.slice %108 [0:1, 0:8, 0:128] : (tensor<2x8x128xf32>) -> tensor<1x8x128xf32>
      %111 = stablehlo.reshape %110 : (tensor<1x8x128xf32>) -> tensor<8x128xf32>
      %112 = stablehlo.slice %107 [0:1, 0:8, 0:128] : (tensor<2x8x128xf32>) -> tensor<1x8x128xf32>
      %113 = stablehlo.reshape %112 : (tensor<1x8x128xf32>) -> tensor<8x128xf32>
      %114 = func.call @_where_6(%109, %111, %113) : (tensor<i1>, tensor<8x128xf32>, tensor<8x128xf32>) -> tensor<8x128xf32>
      %115 = stablehlo.broadcast_in_dim %c_2, dims = [] : (tensor<i32>) -> tensor<1xi32>
      %116 = "stablehlo.scatter"(%107, %115, %114) <{indices_are_sorted = true, scatter_dimension_numbers = #stablehlo.scatter<update_window_dims = [0, 1], inserted_window_dims = [0], scatter_dims_to_operand_dims = [0]>, unique_indices = true}> ({
      ^bb0(%arg16: tensor<f32>, %arg17: tensor<f32>):
        stablehlo.return %arg17 : tensor<f32>
      }) : (tensor<2x8x128xf32>, tensor<1xi32>, tensor<8x128xf32>) -> tensor<2x8x128xf32>
      %117 = stablehlo.dot_general %116, %arg10, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<2x8x128xf32>, tensor<2x128x128xf32>) -> tensor<2x8x128xf32>
      %118 = stablehlo.broadcast_in_dim %arg11, dims = [0, 2] : (tensor<2x128xf32>) -> tensor<2x1x128xf32>
      %119 = stablehlo.broadcast_in_dim %118, dims = [0, 1, 2] : (tensor<2x1x128xf32>) -> tensor<2x8x128xf32>
      %120 = stablehlo.add %117, %119 : tensor<2x8x128xf32>
      %121 = func.call @relu_7(%120) : (tensor<2x8x128xf32>) -> tensor<2x8x128xf32>
      %122 = stablehlo.compare  EQ, %10, %c_3,  SIGNED : (tensor<i32>, tensor<i32>) -> tensor<i1>
      %123 = stablehlo.compare  LT, %c_4, %c_5,  SIGNED : (tensor<i32>, tensor<i32>) -> tensor<i1>
      %124 = stablehlo.add %c_6, %c_7 : tensor<i32>
      %125 = stablehlo.select %123, %124, %c_4 : tensor<i1>, tensor<i32>
      %126 = stablehlo.compare  LT, %c_5, %c_5,  SIGNED : (tensor<i32>, tensor<i32>) -> tensor<i1>
      %127 = stablehlo.add %c_2, %c_8 : tensor<i32>
      %128 = stablehlo.select %126, %127, %c_5 : tensor<i1>, tensor<i32>
      %129 = stablehlo.compare  LT, %c_5, %c_5,  SIGNED : (tensor<i32>, tensor<i32>) -> tensor<i1>
      %130 = stablehlo.add %c_2, %c_9 : tensor<i32>
      %131 = stablehlo.select %129, %130, %c_5 : tensor<i1>, tensor<i32>
      %132 = stablehlo.dynamic_slice %121, %125, %128, %131, sizes = [1, 8, 128] : (tensor<2x8x128xf32>, tensor<i32>, tensor<i32>, tensor<i32>) -> tensor<1x8x128xf32>
      %133 = stablehlo.reshape %132 : (tensor<1x8x128xf32>) -> tensor<8x128xf32>
      %134 = stablehlo.slice %92 [1:2, 0:8, 0:128] : (tensor<2x8x128xf32>) -> tensor<1x8x128xf32>
      %135 = stablehlo.reshape %134 : (tensor<1x8x128xf32>) -> tensor<8x128xf32>
      %136 = func.call @_where_8(%122, %133, %135) : (tensor<i1>, tensor<8x128xf32>, tensor<8x128xf32>) -> tensor<8x128xf32>
      %137 = stablehlo.broadcast_in_dim %c_3, dims = [] : (tensor<i32>) -> tensor<1xi32>
      %138 = "stablehlo.scatter"(%92, %137, %136) <{indices_are_sorted = true, scatter_dimension_numbers = #stablehlo.scatter<update_window_dims = [0, 1], inserted_window_dims = [0], scatter_dims_to_operand_dims = [0]>, unique_indices = true}> ({
      ^bb0(%arg16: tensor<f32>, %arg17: tensor<f32>):
        stablehlo.return %arg17 : tensor<f32>
      }) : (tensor<2x8x128xf32>, tensor<1xi32>, tensor<8x128xf32>) -> tensor<2x8x128xf32>
      %139 = func.call @_roll_static_9(%121) : (tensor<2x8x128xf32>) -> tensor<2x8x128xf32>
      %140 = stablehlo.compare  LT, %c_4, %c_5,  SIGNED : (tensor<i32>, tensor<i32>) -> tensor<i1>
      %141 = stablehlo.add %c_6, %c_7 : tensor<i32>
      %142 = stablehlo.select %140, %141, %c_4 : tensor<i1>, tensor<i32>
      %143 = stablehlo.compare  LT, %c_5, %c_5,  SIGNED : (tensor<i32>, tensor<i32>) -> tensor<i1>
      %144 = stablehlo.add %c_2, %c_8 : tensor<i32>
      %145 = stablehlo.select %143, %144, %c_5 : tensor<i1>, tensor<i32>
      %146 = stablehlo.compare  LT, %c_5, %c_5,  SIGNED : (tensor<i32>, tensor<i32>) -> tensor<i1>
      %147 = stablehlo.add %c_2, %c_9 : tensor<i32>
      %148 = stablehlo.select %146, %147, %c_5 : tensor<i1>, tensor<i32>
      %149 = stablehlo.dynamic_slice %121, %142, %145, %148, sizes = [1, 8, 128] : (tensor<2x8x128xf32>, tensor<i32>, tensor<i32>, tensor<i32>) -> tensor<1x8x128xf32>
      %150 = stablehlo.reshape %149 : (tensor<1x8x128xf32>) -> tensor<8x128xf32>
      %151 = "stablehlo.collective_permute"(%150) <{channel_handle = #stablehlo.channel_handle<handle = 5, type = 1>, source_target_pairs = dense<[[0, 4], [4, 0], [1, 5], [5, 1], [2, 6], [6, 2], [3, 7], [7, 3]]> : tensor<8x2xi64>}> : (tensor<8x128xf32>) -> tensor<8x128xf32>
      %152 = stablehlo.broadcast_in_dim %c_2, dims = [] : (tensor<i32>) -> tensor<1xi32>
      %153 = "stablehlo.scatter"(%139, %152, %151) <{indices_are_sorted = true, scatter_dimension_numbers = #stablehlo.scatter<update_window_dims = [0, 1], inserted_window_dims = [0], scatter_dims_to_operand_dims = [0]>, unique_indices = true}> ({
      ^bb0(%arg16: tensor<f32>, %arg17: tensor<f32>):
        stablehlo.return %arg17 : tensor<f32>
      }) : (tensor<2x8x128xf32>, tensor<1xi32>, tensor<8x128xf32>) -> tensor<2x8x128xf32>
      %154 = "stablehlo.collective_permute"(%138) <{channel_handle = #stablehlo.channel_handle<handle = 6, type = 1>, source_target_pairs = dense<[[0, 4], [4, 0], [1, 5], [5, 1], [2, 6], [6, 2], [3, 7], [7, 3]]> : tensor<8x2xi64>}> : (tensor<2x8x128xf32>) -> tensor<2x8x128xf32>
      %155 = stablehlo.compare  EQ, %10, %c_2,  SIGNED : (tensor<i32>, tensor<i32>) -> tensor<i1>
      %156 = stablehlo.slice %108 [1:2, 0:8, 0:128] : (tensor<2x8x128xf32>) -> tensor<1x8x128xf32>
      %157 = stablehlo.reshape %156 : (tensor<1x8x128xf32>) -> tensor<8x128xf32>
      %158 = stablehlo.slice %153 [0:1, 0:8, 0:128] : (tensor<2x8x128xf32>) -> tensor<1x8x128xf32>
      %159 = stablehlo.reshape %158 : (tensor<1x8x128xf32>) -> tensor<8x128xf32>
      %160 = func.call @_where_10(%155, %157, %159) : (tensor<i1>, tensor<8x128xf32>, tensor<8x128xf32>) -> tensor<8x128xf32>
      %161 = stablehlo.broadcast_in_dim %c_2, dims = [] : (tensor<i32>) -> tensor<1xi32>
      %162 = "stablehlo.scatter"(%153, %161, %160) <{indices_are_sorted = true, scatter_dimension_numbers = #stablehlo.scatter<update_window_dims = [0, 1], inserted_window_dims = [0], scatter_dims_to_operand_dims = [0]>, unique_indices = true}> ({
      ^bb0(%arg16: tensor<f32>, %arg17: tensor<f32>):
        stablehlo.return %arg17 : tensor<f32>
      }) : (tensor<2x8x128xf32>, tensor<1xi32>, tensor<8x128xf32>) -> tensor<2x8x128xf32>
      %163 = stablehlo.dot_general %162, %arg10, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<2x8x128xf32>, tensor<2x128x128xf32>) -> tensor<2x8x128xf32>
      %164 = stablehlo.broadcast_in_dim %arg11, dims = [0, 2] : (tensor<2x128xf32>) -> tensor<2x1x128xf32>
      %165 = stablehlo.broadcast_in_dim %164, dims = [0, 1, 2] : (tensor<2x1x128xf32>) -> tensor<2x8x128xf32>
      %166 = stablehlo.add %163, %165 : tensor<2x8x128xf32>
      %167 = func.call @relu_11(%166) : (tensor<2x8x128xf32>) -> tensor<2x8x128xf32>
      %168 = stablehlo.compare  EQ, %10, %c_3,  SIGNED : (tensor<i32>, tensor<i32>) -> tensor<i1>
      %169 = stablehlo.compare  LT, %c_4, %c_5,  SIGNED : (tensor<i32>, tensor<i32>) -> tensor<i1>
      %170 = stablehlo.add %c_6, %c_7 : tensor<i32>
      %171 = stablehlo.select %169, %170, %c_4 : tensor<i1>, tensor<i32>
      %172 = stablehlo.compare  LT, %c_5, %c_5,  SIGNED : (tensor<i32>, tensor<i32>) -> tensor<i1>
      %173 = stablehlo.add %c_2, %c_8 : tensor<i32>
      %174 = stablehlo.select %172, %173, %c_5 : tensor<i1>, tensor<i32>
      %175 = stablehlo.compare  LT, %c_5, %c_5,  SIGNED : (tensor<i32>, tensor<i32>) -> tensor<i1>
      %176 = stablehlo.add %c_2, %c_9 : tensor<i32>
      %177 = stablehlo.select %175, %176, %c_5 : tensor<i1>, tensor<i32>
      %178 = stablehlo.dynamic_slice %167, %171, %174, %177, sizes = [1, 8, 128] : (tensor<2x8x128xf32>, tensor<i32>, tensor<i32>, tensor<i32>) -> tensor<1x8x128xf32>
      %179 = stablehlo.reshape %178 : (tensor<1x8x128xf32>) -> tensor<8x128xf32>
      %180 = stablehlo.slice %154 [0:1, 0:8, 0:128] : (tensor<2x8x128xf32>) -> tensor<1x8x128xf32>
      %181 = stablehlo.reshape %180 : (tensor<1x8x128xf32>) -> tensor<8x128xf32>
      %182 = func.call @_where_12(%168, %179, %181) : (tensor<i1>, tensor<8x128xf32>, tensor<8x128xf32>) -> tensor<8x128xf32>
      %183 = stablehlo.broadcast_in_dim %c_2, dims = [] : (tensor<i32>) -> tensor<1xi32>
      %184 = "stablehlo.scatter"(%154, %183, %182) <{indices_are_sorted = true, scatter_dimension_numbers = #stablehlo.scatter<update_window_dims = [0, 1], inserted_window_dims = [0], scatter_dims_to_operand_dims = [0]>, unique_indices = true}> ({
      ^bb0(%arg16: tensor<f32>, %arg17: tensor<f32>):
        stablehlo.return %arg17 : tensor<f32>
      }) : (tensor<2x8x128xf32>, tensor<1xi32>, tensor<8x128xf32>) -> tensor<2x8x128xf32>
      %185 = func.call @_roll_static_13(%167) : (tensor<2x8x128xf32>) -> tensor<2x8x128xf32>
      %186 = stablehlo.compare  LT, %c_4, %c_5,  SIGNED : (tensor<i32>, tensor<i32>) -> tensor<i1>
      %187 = stablehlo.add %c_6, %c_7 : tensor<i32>
      %188 = stablehlo.select %186, %187, %c_4 : tensor<i1>, tensor<i32>
      %189 = stablehlo.compare  LT, %c_5, %c_5,  SIGNED : (tensor<i32>, tensor<i32>) -> tensor<i1>
      %190 = stablehlo.add %c_2, %c_8 : tensor<i32>
      %191 = stablehlo.select %189, %190, %c_5 : tensor<i1>, tensor<i32>
      %192 = stablehlo.compare  LT, %c_5, %c_5,  SIGNED : (tensor<i32>, tensor<i32>) -> tensor<i1>
      %193 = stablehlo.add %c_2, %c_9 : tensor<i32>
      %194 = stablehlo.select %192, %193, %c_5 : tensor<i1>, tensor<i32>
      %195 = stablehlo.dynamic_slice %167, %188, %191, %194, sizes = [1, 8, 128] : (tensor<2x8x128xf32>, tensor<i32>, tensor<i32>, tensor<i32>) -> tensor<1x8x128xf32>
      %196 = stablehlo.reshape %195 : (tensor<1x8x128xf32>) -> tensor<8x128xf32>
      %197 = "stablehlo.collective_permute"(%196) <{channel_handle = #stablehlo.channel_handle<handle = 7, type = 1>, source_target_pairs = dense<[[0, 4], [4, 0], [1, 5], [5, 1], [2, 6], [6, 2], [3, 7], [7, 3]]> : tensor<8x2xi64>}> : (tensor<8x128xf32>) -> tensor<8x128xf32>
      %198 = stablehlo.broadcast_in_dim %c_2, dims = [] : (tensor<i32>) -> tensor<1xi32>
      %199 = "stablehlo.scatter"(%185, %198, %197) <{indices_are_sorted = true, scatter_dimension_numbers = #stablehlo.scatter<update_window_dims = [0, 1], inserted_window_dims = [0], scatter_dims_to_operand_dims = [0]>, unique_indices = true}> ({
      ^bb0(%arg16: tensor<f32>, %arg17: tensor<f32>):
        stablehlo.return %arg17 : tensor<f32>
      }) : (tensor<2x8x128xf32>, tensor<1xi32>, tensor<8x128xf32>) -> tensor<2x8x128xf32>
      %200 = "stablehlo.collective_permute"(%108) <{channel_handle = #stablehlo.channel_handle<handle = 8, type = 1>, source_target_pairs = dense<[[0, 4], [4, 0], [1, 5], [5, 1], [2, 6], [6, 2], [3, 7], [7, 3]]> : tensor<8x2xi64>}> : (tensor<2x8x128xf32>) -> tensor<2x8x128xf32>
      %201 = stablehlo.compare  EQ, %10, %c_2,  SIGNED : (tensor<i32>, tensor<i32>) -> tensor<i1>
      %202 = stablehlo.slice %200 [0:1, 0:8, 0:128] : (tensor<2x8x128xf32>) -> tensor<1x8x128xf32>
      %203 = stablehlo.reshape %202 : (tensor<1x8x128xf32>) -> tensor<8x128xf32>
      %204 = stablehlo.slice %199 [0:1, 0:8, 0:128] : (tensor<2x8x128xf32>) -> tensor<1x8x128xf32>
      %205 = stablehlo.reshape %204 : (tensor<1x8x128xf32>) -> tensor<8x128xf32>
      %206 = func.call @_where_14(%201, %203, %205) : (tensor<i1>, tensor<8x128xf32>, tensor<8x128xf32>) -> tensor<8x128xf32>
      %207 = stablehlo.broadcast_in_dim %c_2, dims = [] : (tensor<i32>) -> tensor<1xi32>
      %208 = "stablehlo.scatter"(%199, %207, %206) <{indices_are_sorted = true, scatter_dimension_numbers = #stablehlo.scatter<update_window_dims = [0, 1], inserted_window_dims = [0], scatter_dims_to_operand_dims = [0]>, unique_indices = true}> ({
      ^bb0(%arg16: tensor<f32>, %arg17: tensor<f32>):
        stablehlo.return %arg17 : tensor<f32>
      }) : (tensor<2x8x128xf32>, tensor<1xi32>, tensor<8x128xf32>) -> tensor<2x8x128xf32>
      %209 = stablehlo.dot_general %208, %arg10, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<2x8x128xf32>, tensor<2x128x128xf32>) -> tensor<2x8x128xf32>
      %210 = stablehlo.broadcast_in_dim %arg11, dims = [0, 2] : (tensor<2x128xf32>) -> tensor<2x1x128xf32>
      %211 = stablehlo.broadcast_in_dim %210, dims = [0, 1, 2] : (tensor<2x1x128xf32>) -> tensor<2x8x128xf32>
      %212 = stablehlo.add %209, %211 : tensor<2x8x128xf32>
      %213 = func.call @relu_15(%212) : (tensor<2x8x128xf32>) -> tensor<2x8x128xf32>
      %214 = stablehlo.compare  EQ, %10, %c_3,  SIGNED : (tensor<i32>, tensor<i32>) -> tensor<i1>
      %215 = stablehlo.compare  LT, %c_4, %c_5,  SIGNED : (tensor<i32>, tensor<i32>) -> tensor<i1>
      %216 = stablehlo.add %c_6, %c_7 : tensor<i32>
      %217 = stablehlo.select %215, %216, %c_4 : tensor<i1>, tensor<i32>
      %218 = stablehlo.compare  LT, %c_5, %c_5,  SIGNED : (tensor<i32>, tensor<i32>) -> tensor<i1>
      %219 = stablehlo.add %c_2, %c_8 : tensor<i32>
      %220 = stablehlo.select %218, %219, %c_5 : tensor<i1>, tensor<i32>
      %221 = stablehlo.compare  LT, %c_5, %c_5,  SIGNED : (tensor<i32>, tensor<i32>) -> tensor<i1>
      %222 = stablehlo.add %c_2, %c_9 : tensor<i32>
      %223 = stablehlo.select %221, %222, %c_5 : tensor<i1>, tensor<i32>
      %224 = stablehlo.dynamic_slice %213, %217, %220, %223, sizes = [1, 8, 128] : (tensor<2x8x128xf32>, tensor<i32>, tensor<i32>, tensor<i32>) -> tensor<1x8x128xf32>
      %225 = stablehlo.reshape %224 : (tensor<1x8x128xf32>) -> tensor<8x128xf32>
      %226 = stablehlo.slice %184 [1:2, 0:8, 0:128] : (tensor<2x8x128xf32>) -> tensor<1x8x128xf32>
      %227 = stablehlo.reshape %226 : (tensor<1x8x128xf32>) -> tensor<8x128xf32>
      %228 = func.call @_where_16(%214, %225, %227) : (tensor<i1>, tensor<8x128xf32>, tensor<8x128xf32>) -> tensor<8x128xf32>
      %229 = stablehlo.broadcast_in_dim %c_3, dims = [] : (tensor<i32>) -> tensor<1xi32>
      %230 = "stablehlo.scatter"(%184, %229, %228) <{indices_are_sorted = true, scatter_dimension_numbers = #stablehlo.scatter<update_window_dims = [0, 1], inserted_window_dims = [0], scatter_dims_to_operand_dims = [0]>, unique_indices = true}> ({
      ^bb0(%arg16: tensor<f32>, %arg17: tensor<f32>):
        stablehlo.return %arg17 : tensor<f32>
      }) : (tensor<2x8x128xf32>, tensor<1xi32>, tensor<8x128xf32>) -> tensor<2x8x128xf32>
      %231 = func.call @_roll_static_17(%213) : (tensor<2x8x128xf32>) -> tensor<2x8x128xf32>
      %232 = stablehlo.compare  LT, %c_4, %c_5,  SIGNED : (tensor<i32>, tensor<i32>) -> tensor<i1>
      %233 = stablehlo.add %c_6, %c_7 : tensor<i32>
      %234 = stablehlo.select %232, %233, %c_4 : tensor<i1>, tensor<i32>
      %235 = stablehlo.compare  LT, %c_5, %c_5,  SIGNED : (tensor<i32>, tensor<i32>) -> tensor<i1>
      %236 = stablehlo.add %c_2, %c_8 : tensor<i32>
      %237 = stablehlo.select %235, %236, %c_5 : tensor<i1>, tensor<i32>
      %238 = stablehlo.compare  LT, %c_5, %c_5,  SIGNED : (tensor<i32>, tensor<i32>) -> tensor<i1>
      %239 = stablehlo.add %c_2, %c_9 : tensor<i32>
      %240 = stablehlo.select %238, %239, %c_5 : tensor<i1>, tensor<i32>
      %241 = stablehlo.dynamic_slice %213, %234, %237, %240, sizes = [1, 8, 128] : (tensor<2x8x128xf32>, tensor<i32>, tensor<i32>, tensor<i32>) -> tensor<1x8x128xf32>
      %242 = stablehlo.reshape %241 : (tensor<1x8x128xf32>) -> tensor<8x128xf32>
      %243 = "stablehlo.collective_permute"(%242) <{channel_handle = #stablehlo.channel_handle<handle = 9, type = 1>, source_target_pairs = dense<[[0, 4], [4, 0], [1, 5], [5, 1], [2, 6], [6, 2], [3, 7], [7, 3]]> : tensor<8x2xi64>}> : (tensor<8x128xf32>) -> tensor<8x128xf32>
      %244 = stablehlo.broadcast_in_dim %c_2, dims = [] : (tensor<i32>) -> tensor<1xi32>
      %245 = "stablehlo.scatter"(%231, %244, %243) <{indices_are_sorted = true, scatter_dimension_numbers = #stablehlo.scatter<update_window_dims = [0, 1], inserted_window_dims = [0], scatter_dims_to_operand_dims = [0]>, unique_indices = true}> ({
      ^bb0(%arg16: tensor<f32>, %arg17: tensor<f32>):
        stablehlo.return %arg17 : tensor<f32>
      }) : (tensor<2x8x128xf32>, tensor<1xi32>, tensor<8x128xf32>) -> tensor<2x8x128xf32>
      %246 = "stablehlo.collective_permute"(%230) <{channel_handle = #stablehlo.channel_handle<handle = 10, type = 1>, source_target_pairs = dense<[[0, 4], [4, 0], [1, 5], [5, 1], [2, 6], [6, 2], [3, 7], [7, 3]]> : tensor<8x2xi64>}> : (tensor<2x8x128xf32>) -> tensor<2x8x128xf32>
      %247 = stablehlo.compare  EQ, %10, %c_2,  SIGNED : (tensor<i32>, tensor<i32>) -> tensor<i1>
      %248 = stablehlo.slice %200 [1:2, 0:8, 0:128] : (tensor<2x8x128xf32>) -> tensor<1x8x128xf32>
      %249 = stablehlo.reshape %248 : (tensor<1x8x128xf32>) -> tensor<8x128xf32>
      %250 = stablehlo.slice %245 [0:1, 0:8, 0:128] : (tensor<2x8x128xf32>) -> tensor<1x8x128xf32>
      %251 = stablehlo.reshape %250 : (tensor<1x8x128xf32>) -> tensor<8x128xf32>
      %252 = func.call @_where_18(%247, %249, %251) : (tensor<i1>, tensor<8x128xf32>, tensor<8x128xf32>) -> tensor<8x128xf32>
      %253 = stablehlo.broadcast_in_dim %c_2, dims = [] : (tensor<i32>) -> tensor<1xi32>
      %254 = "stablehlo.scatter"(%245, %253, %252) <{indices_are_sorted = true, scatter_dimension_numbers = #stablehlo.scatter<update_window_dims = [0, 1], inserted_window_dims = [0], scatter_dims_to_operand_dims = [0]>, unique_indices = true}> ({
      ^bb0(%arg16: tensor<f32>, %arg17: tensor<f32>):
        stablehlo.return %arg17 : tensor<f32>
      }) : (tensor<2x8x128xf32>, tensor<1xi32>, tensor<8x128xf32>) -> tensor<2x8x128xf32>
      %255 = stablehlo.dot_general %254, %arg10, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<2x8x128xf32>, tensor<2x128x128xf32>) -> tensor<2x8x128xf32>
      %256 = stablehlo.broadcast_in_dim %arg11, dims = [0, 2] : (tensor<2x128xf32>) -> tensor<2x1x128xf32>
      %257 = stablehlo.broadcast_in_dim %256, dims = [0, 1, 2] : (tensor<2x1x128xf32>) -> tensor<2x8x128xf32>
      %258 = stablehlo.add %255, %257 : tensor<2x8x128xf32>
      %259 = func.call @relu_19(%258) : (tensor<2x8x128xf32>) -> tensor<2x8x128xf32>
      %260 = stablehlo.compare  EQ, %10, %c_3,  SIGNED : (tensor<i32>, tensor<i32>) -> tensor<i1>
      %261 = stablehlo.compare  LT, %c_4, %c_5,  SIGNED : (tensor<i32>, tensor<i32>) -> tensor<i1>
      %262 = stablehlo.add %c_6, %c_7 : tensor<i32>
      %263 = stablehlo.select %261, %262, %c_4 : tensor<i1>, tensor<i32>
      %264 = stablehlo.compare  LT, %c_5, %c_5,  SIGNED : (tensor<i32>, tensor<i32>) -> tensor<i1>
      %265 = stablehlo.add %c_2, %c_8 : tensor<i32>
      %266 = stablehlo.select %264, %265, %c_5 : tensor<i1>, tensor<i32>
      %267 = stablehlo.compare  LT, %c_5, %c_5,  SIGNED : (tensor<i32>, tensor<i32>) -> tensor<i1>
      %268 = stablehlo.add %c_2, %c_9 : tensor<i32>
      %269 = stablehlo.select %267, %268, %c_5 : tensor<i1>, tensor<i32>
      %270 = stablehlo.dynamic_slice %259, %263, %266, %269, sizes = [1, 8, 128] : (tensor<2x8x128xf32>, tensor<i32>, tensor<i32>, tensor<i32>) -> tensor<1x8x128xf32>
      %271 = stablehlo.reshape %270 : (tensor<1x8x128xf32>) -> tensor<8x128xf32>
      %272 = stablehlo.slice %246 [0:1, 0:8, 0:128] : (tensor<2x8x128xf32>) -> tensor<1x8x128xf32>
      %273 = stablehlo.reshape %272 : (tensor<1x8x128xf32>) -> tensor<8x128xf32>
      %274 = func.call @_where_20(%260, %271, %273) : (tensor<i1>, tensor<8x128xf32>, tensor<8x128xf32>) -> tensor<8x128xf32>
      %275 = stablehlo.broadcast_in_dim %c_2, dims = [] : (tensor<i32>) -> tensor<1xi32>
      %276 = "stablehlo.scatter"(%246, %275, %274) <{indices_are_sorted = true, scatter_dimension_numbers = #stablehlo.scatter<update_window_dims = [0, 1], inserted_window_dims = [0], scatter_dims_to_operand_dims = [0]>, unique_indices = true}> ({
      ^bb0(%arg16: tensor<f32>, %arg17: tensor<f32>):
        stablehlo.return %arg17 : tensor<f32>
      }) : (tensor<2x8x128xf32>, tensor<1xi32>, tensor<8x128xf32>) -> tensor<2x8x128xf32>
      %277 = func.call @_roll_static_21(%259) : (tensor<2x8x128xf32>) -> tensor<2x8x128xf32>
      %278 = stablehlo.compare  LT, %c_4, %c_5,  SIGNED : (tensor<i32>, tensor<i32>) -> tensor<i1>
      %279 = stablehlo.add %c_6, %c_7 : tensor<i32>
      %280 = stablehlo.select %278, %279, %c_4 : tensor<i1>, tensor<i32>
      %281 = stablehlo.compare  LT, %c_5, %c_5,  SIGNED : (tensor<i32>, tensor<i32>) -> tensor<i1>
      %282 = stablehlo.add %c_2, %c_8 : tensor<i32>
      %283 = stablehlo.select %281, %282, %c_5 : tensor<i1>, tensor<i32>
      %284 = stablehlo.compare  LT, %c_5, %c_5,  SIGNED : (tensor<i32>, tensor<i32>) -> tensor<i1>
      %285 = stablehlo.add %c_2, %c_9 : tensor<i32>
      %286 = stablehlo.select %284, %285, %c_5 : tensor<i1>, tensor<i32>
      %287 = stablehlo.dynamic_slice %259, %280, %283, %286, sizes = [1, 8, 128] : (tensor<2x8x128xf32>, tensor<i32>, tensor<i32>, tensor<i32>) -> tensor<1x8x128xf32>
      %288 = stablehlo.reshape %287 : (tensor<1x8x128xf32>) -> tensor<8x128xf32>
      %289 = "stablehlo.collective_permute"(%288) <{channel_handle = #stablehlo.channel_handle<handle = 11, type = 1>, source_target_pairs = dense<[[0, 4], [4, 0], [1, 5], [5, 1], [2, 6], [6, 2], [3, 7], [7, 3]]> : tensor<8x2xi64>}> : (tensor<8x128xf32>) -> tensor<8x128xf32>
      %290 = stablehlo.broadcast_in_dim %c_2, dims = [] : (tensor<i32>) -> tensor<1xi32>
      %291 = "stablehlo.scatter"(%277, %290, %289) <{indices_are_sorted = true, scatter_dimension_numbers = #stablehlo.scatter<update_window_dims = [0, 1], inserted_window_dims = [0], scatter_dims_to_operand_dims = [0]>, unique_indices = true}> ({
      ^bb0(%arg16: tensor<f32>, %arg17: tensor<f32>):
        stablehlo.return %arg17 : tensor<f32>
      }) : (tensor<2x8x128xf32>, tensor<1xi32>, tensor<8x128xf32>) -> tensor<2x8x128xf32>
      %292 = "stablehlo.collective_permute"(%200) <{channel_handle = #stablehlo.channel_handle<handle = 12, type = 1>, source_target_pairs = dense<[[0, 4], [4, 0], [1, 5], [5, 1], [2, 6], [6, 2], [3, 7], [7, 3]]> : tensor<8x2xi64>}> : (tensor<2x8x128xf32>) -> tensor<2x8x128xf32>
      %293 = stablehlo.compare  EQ, %10, %c_2,  SIGNED : (tensor<i32>, tensor<i32>) -> tensor<i1>
      %294 = stablehlo.slice %292 [0:1, 0:8, 0:128] : (tensor<2x8x128xf32>) -> tensor<1x8x128xf32>
      %295 = stablehlo.reshape %294 : (tensor<1x8x128xf32>) -> tensor<8x128xf32>
      %296 = stablehlo.slice %291 [0:1, 0:8, 0:128] : (tensor<2x8x128xf32>) -> tensor<1x8x128xf32>
      %297 = stablehlo.reshape %296 : (tensor<1x8x128xf32>) -> tensor<8x128xf32>
      %298 = func.call @_where_22(%293, %295, %297) : (tensor<i1>, tensor<8x128xf32>, tensor<8x128xf32>) -> tensor<8x128xf32>
      %299 = stablehlo.broadcast_in_dim %c_2, dims = [] : (tensor<i32>) -> tensor<1xi32>
      %300 = "stablehlo.scatter"(%291, %299, %298) <{indices_are_sorted = true, scatter_dimension_numbers = #stablehlo.scatter<update_window_dims = [0, 1], inserted_window_dims = [0], scatter_dims_to_operand_dims = [0]>, unique_indices = true}> ({
      ^bb0(%arg16: tensor<f32>, %arg17: tensor<f32>):
        stablehlo.return %arg17 : tensor<f32>
      }) : (tensor<2x8x128xf32>, tensor<1xi32>, tensor<8x128xf32>) -> tensor<2x8x128xf32>
      %301 = stablehlo.dot_general %300, %arg10, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<2x8x128xf32>, tensor<2x128x128xf32>) -> tensor<2x8x128xf32>
      %302 = stablehlo.broadcast_in_dim %arg11, dims = [0, 2] : (tensor<2x128xf32>) -> tensor<2x1x128xf32>
      %303 = stablehlo.broadcast_in_dim %302, dims = [0, 1, 2] : (tensor<2x1x128xf32>) -> tensor<2x8x128xf32>
      %304 = stablehlo.add %301, %303 : tensor<2x8x128xf32>
      %305 = func.call @relu_23(%304) : (tensor<2x8x128xf32>) -> tensor<2x8x128xf32>
      %306 = stablehlo.compare  EQ, %10, %c_3,  SIGNED : (tensor<i32>, tensor<i32>) -> tensor<i1>
      %307 = stablehlo.compare  LT, %c_4, %c_5,  SIGNED : (tensor<i32>, tensor<i32>) -> tensor<i1>
      %308 = stablehlo.add %c_6, %c_7 : tensor<i32>
      %309 = stablehlo.select %307, %308, %c_4 : tensor<i1>, tensor<i32>
      %310 = stablehlo.compare  LT, %c_5, %c_5,  SIGNED : (tensor<i32>, tensor<i32>) -> tensor<i1>
      %311 = stablehlo.add %c_2, %c_8 : tensor<i32>
      %312 = stablehlo.select %310, %311, %c_5 : tensor<i1>, tensor<i32>
      %313 = stablehlo.compare  LT, %c_5, %c_5,  SIGNED : (tensor<i32>, tensor<i32>) -> tensor<i1>
      %314 = stablehlo.add %c_2, %c_9 : tensor<i32>
      %315 = stablehlo.select %313, %314, %c_5 : tensor<i1>, tensor<i32>
      %316 = stablehlo.dynamic_slice %305, %309, %312, %315, sizes = [1, 8, 128] : (tensor<2x8x128xf32>, tensor<i32>, tensor<i32>, tensor<i32>) -> tensor<1x8x128xf32>
      %317 = stablehlo.reshape %316 : (tensor<1x8x128xf32>) -> tensor<8x128xf32>
      %318 = stablehlo.slice %276 [1:2, 0:8, 0:128] : (tensor<2x8x128xf32>) -> tensor<1x8x128xf32>
      %319 = stablehlo.reshape %318 : (tensor<1x8x128xf32>) -> tensor<8x128xf32>
      %320 = func.call @_where_24(%306, %317, %319) : (tensor<i1>, tensor<8x128xf32>, tensor<8x128xf32>) -> tensor<8x128xf32>
      %321 = stablehlo.broadcast_in_dim %c_3, dims = [] : (tensor<i32>) -> tensor<1xi32>
      %322 = "stablehlo.scatter"(%276, %321, %320) <{indices_are_sorted = true, scatter_dimension_numbers = #stablehlo.scatter<update_window_dims = [0, 1], inserted_window_dims = [0], scatter_dims_to_operand_dims = [0]>, unique_indices = true}> ({
      ^bb0(%arg16: tensor<f32>, %arg17: tensor<f32>):
        stablehlo.return %arg17 : tensor<f32>
      }) : (tensor<2x8x128xf32>, tensor<1xi32>, tensor<8x128xf32>) -> tensor<2x8x128xf32>
      %323 = "stablehlo.collective_permute"(%322) <{channel_handle = #stablehlo.channel_handle<handle = 13, type = 1>, source_target_pairs = dense<[[0, 4], [4, 0], [1, 5], [5, 1], [2, 6], [6, 2], [3, 7], [7, 3]]> : tensor<8x2xi64>}> : (tensor<2x8x128xf32>) -> tensor<2x8x128xf32>
      %324 = "stablehlo.collective_permute"(%323) <{channel_handle = #stablehlo.channel_handle<handle = 14, type = 1>, source_target_pairs = dense<[[0, 4], [4, 0], [1, 5], [5, 1], [2, 6], [6, 2], [3, 7], [7, 3]]> : tensor<8x2xi64>}> : (tensor<2x8x128xf32>) -> tensor<2x8x128xf32>
      %325 = stablehlo.dot_general %324, %arg12, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<2x8x128xf32>, tensor<128x8xf32>) -> tensor<2x8x8xf32>
      %326 = stablehlo.broadcast_in_dim %arg13, dims = [2] : (tensor<8xf32>) -> tensor<1x1x8xf32>
      %327 = stablehlo.broadcast_in_dim %326, dims = [0, 1, 2] : (tensor<1x1x8xf32>) -> tensor<2x8x8xf32>
      %328 = stablehlo.add %325, %327 : tensor<2x8x8xf32>
      %329 = stablehlo.reshape %328 : (tensor<2x8x8xf32>) -> tensor<16x8xf32>
      %330 = stablehlo.subtract %329, %arg15 : tensor<16x8xf32>
      %331 = stablehlo.multiply %330, %330 : tensor<16x8xf32>
      %cst_10 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %332 = stablehlo.reduce(%331 init: %cst_10) applies stablehlo.add across dimensions = [1] : (tensor<16x8xf32>, tensor<f32>) -> tensor<16xf32>
      %cst_11 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %333 = stablehlo.reduce(%332 init: %cst_11) applies stablehlo.add across dimensions = [0] : (tensor<16xf32>, tensor<f32>) -> tensor<f32>
      %cst_12 = stablehlo.constant dense<1.600000e+01> : tensor<f32>
      %334 = stablehlo.divide %333, %cst_12 : tensor<f32>
      %335 = "stablehlo.all_reduce"(%334) <{channel_handle = #stablehlo.channel_handle<handle = 15, type = 1>, replica_groups = dense<[[0, 4], [1, 5], [2, 6], [3, 7]]> : tensor<4x2xi64>, use_global_device_ids}> ({
      ^bb0(%arg16: tensor<f32>, %arg17: tensor<f32>):
        %337 = stablehlo.add %arg16, %arg17 : tensor<f32>
        stablehlo.return %337 : tensor<f32>
      }) : (tensor<f32>) -> tensor<f32>
      %cst_13 = stablehlo.constant dense<2.000000e+00> : tensor<f32>
      %336 = stablehlo.divide %335, %cst_13 : tensor<f32>
      sdy.return %336 : tensor<f32>
    } : (tensor<784x128xf32>, tensor<128xf32>, tensor<4x128x128xf32>, tensor<4x128xf32>, tensor<128x8xf32>, tensor<8xf32>, tensor<32x784xf32>, tensor<32x8xf32>) -> tensor<f32>
    return %0 : tensor<f32>
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
