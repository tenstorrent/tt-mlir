// RUN: ttmlir-opt --d2m-scalarize-const-tensors %s | FileCheck %s

#l1_ = #ttcore.memory_space<l1>
#map = affine_map<(d0, d1) -> (d0, d1)>
#parallel = #ttcore.iterator_type<parallel>
#layout = #ttcore.metal_layout<logical_shape = 128x128, dim_alignments = 32x32, collapsed_intervals = dense<[[0, 1], [1, 2]]> : tensor<2x2xi64>, undef, l1, sharded>
#view_map = affine_map<(d0, d1, d2, d3) -> (d0 * 8 + d2, d1 * 8 + d3, 0, 0)>

!ttype_f32 = !ttcore.tile<32x32, f32>

module {
  // CHECK-LABEL: func.func @test_add_const_tensor_scalarize
  func.func @test_add_const_tensor_scalarize(%arg0: tensor<1x1x4x4x!ttype_f32, #layout>) -> tensor<1x1x4x4x!ttype_f32, #layout> {
    %cst = d2m.full {fill_value = 2.5 : f32, shape = array<i32: 128, 128>} : tensor<128x128xf32>
    %0 = d2m.empty() : tensor<1x1x4x4x!ttype_f32, #layout>
    %1 = d2m.to_layout %cst, %0 : tensor<128x128xf32> into tensor<1x1x4x4x!ttype_f32, #layout> -> tensor<1x1x4x4x!ttype_f32, #layout>
    %2 = d2m.empty() : tensor<1x1x4x4x!ttype_f32, #layout>

    // CHECK: d2m.generic
    // CHECK: ins(%arg0 :
    // CHECK-NOT: ins(%arg0, %
    %3 = d2m.generic {block_factors = [1, 1], grid = #ttcore.grid<1x1>, indexing_maps = [#map, #map, #map], iterator_types = [#parallel, #parallel], threads = [#d2m.thread<unified>]}
        ins(%arg0, %1 : tensor<1x1x4x4x!ttype_f32, #layout>, tensor<1x1x4x4x!ttype_f32, #layout>)
        outs(%2 : tensor<1x1x4x4x!ttype_f32, #layout>)  {
    ^unified0(%cb0: !d2m.cb<tensor<4x4x!ttype_f32>>, %cb1: !d2m.cb<tensor<4x4x!ttype_f32>>, %cb2: !d2m.cb<tensor<4x4x!ttype_f32>>):
      // CHECK: ^{{.*}}(%[[CB0:.*]]: !d2m.cb<tensor<4x4x!ttcore.tile<32x32, f32>>>, %[[CB_OUT:.*]]: !d2m.cb<tensor<4x4x!ttcore.tile<32x32, f32>>>):
      // CHECK-NOT: %cb1
      // CHECK: %[[SCALAR:.*]] = arith.constant 2.500000e+00 : f32
      %iter0 = d2m.block_index(0) : index
      %iter1 = d2m.block_index(1) : index
      %buffer0 = tensor.empty() : tensor<4x4x!ttype_f32>
      %buffer1 = tensor.empty() : tensor<4x4x!ttype_f32>
      %10 = d2m.remote_load %buffer0 %arg0[%iter0, %iter1] : tensor<4x4x!ttype_f32>, tensor<1x1x4x4x!ttype_f32, #layout> -> tensor<4x4x!ttype_f32>
      %11 = d2m.remote_load %buffer1 %1[%iter0, %iter1] : tensor<4x4x!ttype_f32>, tensor<1x1x4x4x!ttype_f32, #layout> -> tensor<4x4x!ttype_f32>
      // CHECK: %[[LOAD:.*]] = d2m.remote_load %{{.*}} %arg0
      // CHECK-NOT: d2m.remote_load
      %12 = d2m.reserve %cb2 : <tensor<4x4x!ttype_f32>> -> tensor<4x4x!ttype_f32>
      // CHECK: %[[RESERVE:.*]] = d2m.reserve %[[CB_OUT]]
      %13 = linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = ["parallel", "parallel"]} ins(%10, %11 : tensor<4x4x!ttype_f32>, tensor<4x4x!ttype_f32>) outs(%12 : tensor<4x4x!ttype_f32>) {
      ^bb0(%in: !ttype_f32, %in_0: !ttype_f32, %out: !ttype_f32):
        // CHECK: linalg.generic {{.*}} ins(%[[LOAD]] :
        // CHECK-NOT: ins(%[[LOAD]], %
        // CHECK: ^bb0(%[[IN:.*]]: !ttcore.tile<32x32, f32>, %[[OUT:.*]]: !ttcore.tile<32x32, f32>):
        // CHECK-NOT: %in_0
        // CHECK: "d2m.tile_add"(%[[IN]], %[[SCALAR]]) : (!ttcore.tile<32x32, f32>, f32)
        %14 = "d2m.tile_add"(%in, %in_0) : (!ttype_f32, !ttype_f32) -> !ttype_f32
        linalg.yield %14 : !ttype_f32
      } -> tensor<4x4x!ttype_f32>
      d2m.yield %13 : (tensor<4x4x!ttype_f32>)
    } : tensor<1x1x4x4x!ttype_f32, #layout>
    return %3 : tensor<1x1x4x4x!ttype_f32, #layout>
  }

  // CHECK-LABEL: func.func @test_mul_const_tensor_scalarize
  func.func @test_mul_const_tensor_scalarize(%arg0: tensor<1x1x4x4x!ttype_f32, #layout>) -> tensor<1x1x4x4x!ttype_f32, #layout> {
    %cst = d2m.full {fill_value = 3.0 : f32, shape = array<i32: 128, 128>} : tensor<128x128xf32>
    %0 = d2m.empty() : tensor<1x1x4x4x!ttype_f32, #layout>
    %1 = d2m.to_layout %cst, %0 : tensor<128x128xf32> into tensor<1x1x4x4x!ttype_f32, #layout> -> tensor<1x1x4x4x!ttype_f32, #layout>
    %2 = d2m.empty() : tensor<1x1x4x4x!ttype_f32, #layout>

    // CHECK: d2m.generic
    // CHECK: ins(%arg0 :
    // CHECK-NOT: ins(%arg0, %
    %3 = d2m.generic {block_factors = [1, 1], grid = #ttcore.grid<1x1>, indexing_maps = [#map, #map, #map], iterator_types = [#parallel, #parallel], threads = [#d2m.thread<unified>]}
        ins(%arg0, %1 : tensor<1x1x4x4x!ttype_f32, #layout>, tensor<1x1x4x4x!ttype_f32, #layout>)
        outs(%2 : tensor<1x1x4x4x!ttype_f32, #layout>)  {
    ^unified0(%cb0: !d2m.cb<tensor<4x4x!ttype_f32>>, %cb1: !d2m.cb<tensor<4x4x!ttype_f32>>, %cb2: !d2m.cb<tensor<4x4x!ttype_f32>>):
      // CHECK: ^{{.*}}(%[[CB0:.*]]: !d2m.cb<tensor<4x4x!ttcore.tile<32x32, f32>>>, %[[CB_OUT:.*]]: !d2m.cb<tensor<4x4x!ttcore.tile<32x32, f32>>>):
      // CHECK-NOT: %cb1
      // CHECK: %[[SCALAR:.*]] = arith.constant 3.000000e+00 : f32
      %iter0 = d2m.block_index(0) : index
      %iter1 = d2m.block_index(1) : index
      %buffer0 = tensor.empty() : tensor<4x4x!ttype_f32>
      %buffer1 = tensor.empty() : tensor<4x4x!ttype_f32>
      %10 = d2m.remote_load %buffer0 %arg0[%iter0, %iter1] : tensor<4x4x!ttype_f32>, tensor<1x1x4x4x!ttype_f32, #layout> -> tensor<4x4x!ttype_f32>
      %11 = d2m.remote_load %buffer1 %1[%iter0, %iter1] : tensor<4x4x!ttype_f32>, tensor<1x1x4x4x!ttype_f32, #layout> -> tensor<4x4x!ttype_f32>
      // CHECK: %[[LOAD:.*]] = d2m.remote_load %{{.*}} %arg0
      // CHECK-NOT: d2m.remote_load
      %12 = d2m.reserve %cb2 : <tensor<4x4x!ttype_f32>> -> tensor<4x4x!ttype_f32>
      // CHECK: %[[RESERVE:.*]] = d2m.reserve %[[CB_OUT]]
      %13 = linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = ["parallel", "parallel"]} ins(%10, %11 : tensor<4x4x!ttype_f32>, tensor<4x4x!ttype_f32>) outs(%12 : tensor<4x4x!ttype_f32>) {
      ^bb0(%in: !ttype_f32, %in_0: !ttype_f32, %out: !ttype_f32):
        // CHECK: linalg.generic {{.*}} ins(%[[LOAD]] :
        // CHECK-NOT: ins(%[[LOAD]], %
        // CHECK: ^bb0(%[[IN:.*]]: !ttcore.tile<32x32, f32>, %[[OUT:.*]]: !ttcore.tile<32x32, f32>):
        // CHECK-NOT: %in_0
        // CHECK: "d2m.tile_mul"(%[[IN]], %[[SCALAR]]) : (!ttcore.tile<32x32, f32>, f32)
        %14 = "d2m.tile_mul"(%in, %in_0) : (!ttype_f32, !ttype_f32) -> !ttype_f32
        linalg.yield %14 : !ttype_f32
      } -> tensor<4x4x!ttype_f32>
      d2m.yield %13 : (tensor<4x4x!ttype_f32>)
    } : tensor<1x1x4x4x!ttype_f32, #layout>
    return %3 : tensor<1x1x4x4x!ttype_f32, #layout>
  }

  // CHECK-LABEL: func.func @test_non_scalar_op_no_change
  // Test that operations that don't support scalars are not modified
  func.func @test_non_scalar_op_no_change(%arg0: tensor<1x1x4x4x!ttype_f32, #layout>) -> tensor<1x1x4x4x!ttype_f32, #layout> {
    %cst = d2m.full {fill_value = 2.5 : f32, shape = array<i32: 128, 128>} : tensor<128x128xf32>
    %0 = d2m.empty() : tensor<1x1x4x4x!ttype_f32, #layout>
    %1 = d2m.to_layout %cst, %0 : tensor<128x128xf32> into tensor<1x1x4x4x!ttype_f32, #layout> -> tensor<1x1x4x4x!ttype_f32, #layout>
    %2 = d2m.empty() : tensor<1x1x4x4x!ttype_f32, #layout>

    // CHECK: d2m.generic
    %3 = d2m.generic {block_factors = [1, 1], grid = #ttcore.grid<1x1>, indexing_maps = [#map, #map, #map], iterator_types = [#parallel, #parallel], threads = [#d2m.thread<unified>]}
        ins(%arg0, %1 : tensor<1x1x4x4x!ttype_f32, #layout>, tensor<1x1x4x4x!ttype_f32, #layout>)
        outs(%2 : tensor<1x1x4x4x!ttype_f32, #layout>)  {
    ^unified0(%cb0: !d2m.cb<tensor<4x4x!ttype_f32>>, %cb1: !d2m.cb<tensor<4x4x!ttype_f32>>, %cb2: !d2m.cb<tensor<4x4x!ttype_f32>>):
      %iter0 = d2m.block_index(0) : index
      %iter1 = d2m.block_index(1) : index
      %buffer0 = tensor.empty() : tensor<4x4x!ttype_f32>
      %buffer1 = tensor.empty() : tensor<4x4x!ttype_f32>
      %10 = d2m.remote_load %buffer0 %arg0[%iter0, %iter1] : tensor<4x4x!ttype_f32>, tensor<1x1x4x4x!ttype_f32, #layout> -> tensor<4x4x!ttype_f32>
      %11 = d2m.remote_load %buffer1 %1[%iter0, %iter1] : tensor<4x4x!ttype_f32>, tensor<1x1x4x4x!ttype_f32, #layout> -> tensor<4x4x!ttype_f32>
      %12 = d2m.reserve %cb2 : <tensor<4x4x!ttype_f32>> -> tensor<4x4x!ttype_f32>
      %13 = linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = ["parallel", "parallel"]} ins(%10, %11 : tensor<4x4x!ttype_f32>, tensor<4x4x!ttype_f32>) outs(%12 : tensor<4x4x!ttype_f32>) {
      ^bb0(%in: !ttype_f32, %in_0: !ttype_f32, %out: !ttype_f32):
        // Use an operation that doesn't support scalars (e.g., maximum)
        // CHECK-NOT: arith.constant
        // CHECK: "d2m.tile_maximum"(%{{.*}}, %{{.*}}) : (!ttcore.tile<32x32, f32>, !ttcore.tile<32x32, f32>)
        %14 = "d2m.tile_maximum"(%in, %in_0) : (!ttype_f32, !ttype_f32) -> !ttype_f32
        linalg.yield %14 : !ttype_f32
      } -> tensor<4x4x!ttype_f32>
      d2m.yield %13 : (tensor<4x4x!ttype_f32>)
    } : tensor<1x1x4x4x!ttype_f32, #layout>
    return %3 : tensor<1x1x4x4x!ttype_f32, #layout>
  }

  // CHECK-LABEL: func.func @test_scalarize_through_cast_and_view
  func.func @test_scalarize_through_cast_and_view(%arg0: tensor<256x256xbf16, #ttnn.ttnn_layout<(d0, d1) -> (d0, d1), <8x8>, memref<1x1x!ttcore.tile<32x32, bf16>, #ttnn.buffer_type<l1>>, <block_sharded>, exactGrid = true>>) -> tensor<256x256xbf16, #ttnn.ttnn_layout<(d0, d1) -> (d0, d1), <8x8>, memref<1x1x!ttcore.tile<32x32, bf16>, #ttnn.buffer_type<l1>>, <block_sharded>, exactGrid = true>> {
    %0 = d2m.full {fill_value = 5.000000e-01 : f32, shape = array<i32: 256, 256>} : tensor<256x256xbf16, #ttnn.ttnn_layout<(d0, d1) -> (d0, d1), <8x8>, memref<1x1x!ttcore.tile<32x32, bf16>, #ttnn.buffer_type<l1>>, <block_sharded>, exactGrid = true>>
    %1 = d2m.empty() : tensor<256x256xbf16, #ttnn.ttnn_layout<(d0, d1) -> (d0, d1), <8x8>, memref<1x1x!ttcore.tile<32x32, bf16>, #ttnn.buffer_type<l1>>, <block_sharded>, exactGrid = true>>
    %cast = ttir.ttnn_metal_layout_cast %arg0 : tensor<256x256xbf16, #ttnn.ttnn_layout<(d0, d1) -> (d0, d1), <8x8>, memref<1x1x!ttcore.tile<32x32, bf16>, #ttnn.buffer_type<l1>>, <block_sharded>, exactGrid = true>> -> tensor<8x8x1x1x!ttcore.tile<32x32, bf16>, #ttcore.metal_layout<logical_shape = 256x256, dim_alignments = 32x32, collapsed_intervals = dense<[[0, -1]]> : tensor<1x2xi64>, undef, l1, sharded>>
    %view = d2m.view_layout %cast remapping = #view_map : tensor<8x8x1x1x!ttcore.tile<32x32, bf16>, #ttcore.metal_layout<logical_shape = 256x256, dim_alignments = 32x32, collapsed_intervals = dense<[[0, -1]]> : tensor<1x2xi64>, undef, l1, sharded>> -> tensor<1x1x8x8x!ttcore.tile<32x32, bf16>, #ttcore.metal_layout<logical_shape = 256x256, dim_alignments = 32x32, collapsed_intervals = dense<[[0, -1]]> : tensor<1x2xi64>, undef, l1, sharded>>
    %cast_0 = ttir.ttnn_metal_layout_cast %0 : tensor<256x256xbf16, #ttnn.ttnn_layout<(d0, d1) -> (d0, d1), <8x8>, memref<1x1x!ttcore.tile<32x32, bf16>, #ttnn.buffer_type<l1>>, <block_sharded>, exactGrid = true>> -> tensor<8x8x1x1x!ttcore.tile<32x32, bf16>, #ttcore.metal_layout<logical_shape = 256x256, dim_alignments = 32x32, collapsed_intervals = dense<[[0, -1]]> : tensor<1x2xi64>, undef, l1, sharded>>
    %view_0 = d2m.view_layout %cast_0 remapping = #view_map : tensor<8x8x1x1x!ttcore.tile<32x32, bf16>, #ttcore.metal_layout<logical_shape = 256x256, dim_alignments = 32x32, collapsed_intervals = dense<[[0, -1]]> : tensor<1x2xi64>, undef, l1, sharded>> -> tensor<1x1x8x8x!ttcore.tile<32x32, bf16>, #ttcore.metal_layout<logical_shape = 256x256, dim_alignments = 32x32, collapsed_intervals = dense<[[0, -1]]> : tensor<1x2xi64>, undef, l1, sharded>>
    %cast_1 = ttir.ttnn_metal_layout_cast %1 : tensor<256x256xbf16, #ttnn.ttnn_layout<(d0, d1) -> (d0, d1), <8x8>, memref<1x1x!ttcore.tile<32x32, bf16>, #ttnn.buffer_type<l1>>, <block_sharded>, exactGrid = true>> -> tensor<8x8x1x1x!ttcore.tile<32x32, bf16>, #ttcore.metal_layout<logical_shape = 256x256, dim_alignments = 32x32, collapsed_intervals = dense<[[0, -1]]> : tensor<1x2xi64>, undef, l1, sharded>>
    %view_1 = d2m.view_layout %cast_1 remapping = #view_map : tensor<8x8x1x1x!ttcore.tile<32x32, bf16>, #ttcore.metal_layout<logical_shape = 256x256, dim_alignments = 32x32, collapsed_intervals = dense<[[0, -1]]> : tensor<1x2xi64>, undef, l1, sharded>> -> tensor<1x1x8x8x!ttcore.tile<32x32, bf16>, #ttcore.metal_layout<logical_shape = 256x256, dim_alignments = 32x32, collapsed_intervals = dense<[[0, -1]]> : tensor<1x2xi64>, undef, l1, sharded>>
    // CHECK-NOT: d2m.full
    // CHECK: d2m.generic
    %2 = d2m.generic {block_factors = [1, 1], grid = #ttcore.grid<1x1>, indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = [#ttcore.iterator_type<parallel>, #ttcore.iterator_type<parallel>], threads = [#d2m.thread<unified>]}
        ins(%view, %view_0 : tensor<1x1x8x8x!ttcore.tile<32x32, bf16>, #ttcore.metal_layout<logical_shape = 256x256, dim_alignments = 32x32, collapsed_intervals = dense<[[0, -1]]> : tensor<1x2xi64>, undef, l1, sharded>>, tensor<1x1x8x8x!ttcore.tile<32x32, bf16>, #ttcore.metal_layout<logical_shape = 256x256, dim_alignments = 32x32, collapsed_intervals = dense<[[0, -1]]> : tensor<1x2xi64>, undef, l1, sharded>>)
        outs(%view_1 : tensor<1x1x8x8x!ttcore.tile<32x32, bf16>, #ttcore.metal_layout<logical_shape = 256x256, dim_alignments = 32x32, collapsed_intervals = dense<[[0, -1]]> : tensor<1x2xi64>, undef, l1, sharded>>)  {
    // CHECK: ^unified0(%cb0: !d2m.cb<tensor<8x8x!ttcore.tile<32x32, bf16>>>, %cb1: !d2m.cb<tensor<8x8x!ttcore.tile<32x32, bf16>>>)
    ^unified0(%cb0: !d2m.cb<tensor<8x8x!ttcore.tile<32x32, bf16>>>, %cb1: !d2m.cb<tensor<8x8x!ttcore.tile<32x32, bf16>>>, %cb2: !d2m.cb<tensor<8x8x!ttcore.tile<32x32, bf16>>>):
      // CHECK: %[[SCALAR:.*]] = arith.constant 5.000000e-01 : f32
      %iter0 = d2m.block_index(0) : index
      %iter1 = d2m.block_index(1) : index
      %buffer0 = tensor.empty() : tensor<8x8x!ttcore.tile<32x32, bf16>>
      %buffer1 = tensor.empty() : tensor<8x8x!ttcore.tile<32x32, bf16>>
      %3 = d2m.remote_load %buffer0 %view[%iter0, %iter1] : tensor<8x8x!ttcore.tile<32x32, bf16>>, tensor<1x1x8x8x!ttcore.tile<32x32, bf16>, #ttcore.metal_layout<logical_shape = 256x256, dim_alignments = 32x32, collapsed_intervals = dense<[[0, -1]]> : tensor<1x2xi64>, undef, l1, sharded>> -> tensor<8x8x!ttcore.tile<32x32, bf16>>
      %4 = d2m.remote_load %buffer1 %view_0[%iter0, %iter1] : tensor<8x8x!ttcore.tile<32x32, bf16>>, tensor<1x1x8x8x!ttcore.tile<32x32, bf16>, #ttcore.metal_layout<logical_shape = 256x256, dim_alignments = 32x32, collapsed_intervals = dense<[[0, -1]]> : tensor<1x2xi64>, undef, l1, sharded>> -> tensor<8x8x!ttcore.tile<32x32, bf16>>
      // CHECK: d2m.remote_load %{{.*}} %view
      // CHECK-NOT: d2m.remote_load
      %5 = d2m.reserve %cb2 : <tensor<8x8x!ttcore.tile<32x32, bf16>>> -> tensor<8x8x!ttcore.tile<32x32, bf16>>
      %6 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%3, %4 : tensor<8x8x!ttcore.tile<32x32, bf16>>, tensor<8x8x!ttcore.tile<32x32, bf16>>) outs(%5 : tensor<8x8x!ttcore.tile<32x32, bf16>>) {
      // CHECK: ^bb0(%[[IN:.*]]: !ttcore.tile<32x32, bf16>, %out: !ttcore.tile<32x32, bf16>):
      ^bb0(%in: !ttcore.tile<32x32, bf16>, %in_171: !ttcore.tile<32x32, bf16>, %out: !ttcore.tile<32x32, bf16>):
        // CHECK: "d2m.tile_mul"(%[[IN]], %[[SCALAR]]) : (!ttcore.tile<32x32, bf16>, f32) -> !ttcore.tile<32x32, bf16>
        %7 = "d2m.tile_mul"(%in, %in_171) : (!ttcore.tile<32x32, bf16>, !ttcore.tile<32x32, bf16>) -> !ttcore.tile<32x32, bf16>
        linalg.yield %7 : !ttcore.tile<32x32, bf16>
      } -> tensor<8x8x!ttcore.tile<32x32, bf16>>
      d2m.yield %6 : (tensor<8x8x!ttcore.tile<32x32, bf16>>)
    } : tensor<1x1x8x8x!ttcore.tile<32x32, bf16>, #ttcore.metal_layout<logical_shape = 256x256, dim_alignments = 32x32, collapsed_intervals = dense<[[0, -1]]> : tensor<1x2xi64>, undef, l1, sharded>>
    %cast_2 = ttir.ttnn_metal_layout_cast %2 : tensor<1x1x8x8x!ttcore.tile<32x32, bf16>, #ttcore.metal_layout<logical_shape = 256x256, dim_alignments = 32x32, collapsed_intervals = dense<[[0, -1]]> : tensor<1x2xi64>, undef, l1, sharded>> -> tensor<256x256xbf16, #ttnn.ttnn_layout<(d0, d1) -> (d0, d1), <8x8>, memref<1x1x!ttcore.tile<32x32, bf16>, #ttnn.buffer_type<l1>>, <block_sharded>, exactGrid = true>>
    return %cast_2 : tensor<256x256xbf16, #ttnn.ttnn_layout<(d0, d1) -> (d0, d1), <8x8>, memref<1x1x!ttcore.tile<32x32, bf16>, #ttnn.buffer_type<l1>>, <block_sharded>, exactGrid = true>>
  }
}
