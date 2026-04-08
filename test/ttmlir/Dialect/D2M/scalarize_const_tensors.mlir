// RUN: ttmlir-opt --d2m-scalarize-const-tensors %s -o %t
// RUN: FileCheck %s --input-file=%t

#map = affine_map<(d0, d1) -> (d0, d1)>
#parallel = #ttcore.iterator_type<parallel>
#layout = #ttcore.metal_layout<logical_shape = 128x128, dim_alignments = 32x32, collapsed_intervals = dense<[[0, 1], [1, 2]]> : tensor<2x2xi64>, undef, l1, sharded>

!ttype_f32 = !ttcore.tile<32x32, f32>

module {

  // CHECK-LABEL: func.func @test_add_const_tensor_scalarize
  func.func @test_add_const_tensor_scalarize(%arg0: tensor<1x1x4x4x!ttype_f32, #layout>) -> tensor<1x1x4x4x!ttype_f32, #layout> {
    %e0 = d2m.empty() : tensor<128x128xf32>
    %e1 = d2m.empty() : tensor<1x1x4x4x!ttype_f32, #layout>
    %to_fill = d2m.to_layout %e0, %e1 : tensor<128x128xf32> into tensor<1x1x4x4x!ttype_f32, #layout> -> tensor<1x1x4x4x!ttype_f32, #layout>
    %fill = d2m.generic {block_factors = [1, 1], grid = #ttcore.grid<1x1>, indexing_maps = [#map], iterator_types = [#parallel, #parallel], threads = [#d2m.thread<unified>]}
        ins()
        outs(%to_fill : tensor<1x1x4x4x!ttype_f32, #layout>) {
    ^bb0:
      %fe = tensor.empty() : tensor<4x4x!ttype_f32>
      %fl = linalg.generic {indexing_maps = [#map], iterator_types = ["parallel", "parallel"]} outs(%fe : tensor<4x4x!ttype_f32>) {
      ^bb0(%out: !ttype_f32):
        %cst_fill = arith.constant 2.500000e+00 : f32
        %ft = d2m.tile_fill(%cst_fill) : f32 -> <32x32, f32>
        linalg.yield %ft : !ttype_f32
      } -> tensor<4x4x!ttype_f32>
      %b0 = d2m.block_index(0) : index
      %b1 = d2m.block_index(1) : index
      %st = d2m.remote_store %to_fill[%b0, %b1] %fl : tensor<1x1x4x4x!ttype_f32, #layout>, tensor<4x4x!ttype_f32> -> tensor<1x1x4x4x!ttype_f32, #layout>
      d2m.yield %st : (tensor<1x1x4x4x!ttype_f32, #layout>)
    } : tensor<1x1x4x4x!ttype_f32, #layout>
    %e_rm = d2m.empty() : tensor<128x128xf32>
    %rm = d2m.to_layout %fill, %e_rm : tensor<1x1x4x4x!ttype_f32, #layout> into tensor<128x128xf32> -> tensor<128x128xf32>
    %e_splat = d2m.empty() : tensor<1x1x4x4x!ttype_f32, #layout>
    %splat = d2m.to_layout %rm, %e_splat : tensor<128x128xf32> into tensor<1x1x4x4x!ttype_f32, #layout> -> tensor<1x1x4x4x!ttype_f32, #layout>
    %generic_out = d2m.empty() : tensor<1x1x4x4x!ttype_f32, #layout>
    // CHECK: d2m.generic
    %3 = d2m.generic {block_factors = [1, 1], grid = #ttcore.grid<1x1>, indexing_maps = [#map, #map, #map], iterator_types = [#parallel, #parallel], threads = [#d2m.thread<unified>]}
        ins(%arg0, %splat : tensor<1x1x4x4x!ttype_f32, #layout>, tensor<1x1x4x4x!ttype_f32, #layout>)
        outs(%generic_out : tensor<1x1x4x4x!ttype_f32, #layout>) {
    ^bb1:
      // CHECK: %[[S:.*]] = arith.constant 2.500000e+00 : f32
      %iter0 = d2m.block_index(0) : index
      %iter1 = d2m.block_index(1) : index
      %buffer0 = tensor.empty() : tensor<4x4x!ttype_f32>
      %buffer1 = tensor.empty() : tensor<4x4x!ttype_f32>
      %10 = d2m.remote_load %buffer0 %arg0[%iter0, %iter1] : tensor<4x4x!ttype_f32>, tensor<1x1x4x4x!ttype_f32, #layout> -> tensor<4x4x!ttype_f32>
      %11 = d2m.remote_load %buffer1 %splat[%iter0, %iter1] : tensor<4x4x!ttype_f32>, tensor<1x1x4x4x!ttype_f32, #layout> -> tensor<4x4x!ttype_f32>
      %12 = tensor.empty() : tensor<4x4x!ttype_f32>
      %13 = linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = ["parallel", "parallel"]} ins(%10, %11 : tensor<4x4x!ttype_f32>, tensor<4x4x!ttype_f32>) outs(%12 : tensor<4x4x!ttype_f32>) {
      ^bb0(%in: !ttype_f32, %in_0: !ttype_f32, %out: !ttype_f32):
        // CHECK: "d2m.tile_add"(%{{.*}}, %[[S]]) : (!ttcore.tile<32x32, f32>, f32)
        %14 = "d2m.tile_add"(%in, %in_0) : (!ttype_f32, !ttype_f32) -> !ttype_f32
        linalg.yield %14 : !ttype_f32
      } -> tensor<4x4x!ttype_f32>
      d2m.yield %13 : (tensor<4x4x!ttype_f32>)
    } : tensor<1x1x4x4x!ttype_f32, #layout>
    return %3 : tensor<1x1x4x4x!ttype_f32, #layout>
  }

  // CHECK-LABEL: func.func @test_mul_const_tensor_scalarize
  func.func @test_mul_const_tensor_scalarize(%arg0: tensor<1x1x4x4x!ttype_f32, #layout>) -> tensor<1x1x4x4x!ttype_f32, #layout> {
    %e0 = d2m.empty() : tensor<128x128xf32>
    %e1 = d2m.empty() : tensor<1x1x4x4x!ttype_f32, #layout>
    %to_fill = d2m.to_layout %e0, %e1 : tensor<128x128xf32> into tensor<1x1x4x4x!ttype_f32, #layout> -> tensor<1x1x4x4x!ttype_f32, #layout>
    %fill = d2m.generic {block_factors = [1, 1], grid = #ttcore.grid<1x1>, indexing_maps = [#map], iterator_types = [#parallel, #parallel], threads = [#d2m.thread<unified>]}
        ins()
        outs(%to_fill : tensor<1x1x4x4x!ttype_f32, #layout>) {
    ^bb0:
      %fe = tensor.empty() : tensor<4x4x!ttype_f32>
      %fl = linalg.generic {indexing_maps = [#map], iterator_types = ["parallel", "parallel"]} outs(%fe : tensor<4x4x!ttype_f32>) {
      ^bb0(%out: !ttype_f32):
        %cst_fill = arith.constant 3.000000e+00 : f32
        %ft = d2m.tile_fill(%cst_fill) : f32 -> <32x32, f32>
        linalg.yield %ft : !ttype_f32
      } -> tensor<4x4x!ttype_f32>
      %b0 = d2m.block_index(0) : index
      %b1 = d2m.block_index(1) : index
      %st = d2m.remote_store %to_fill[%b0, %b1] %fl : tensor<1x1x4x4x!ttype_f32, #layout>, tensor<4x4x!ttype_f32> -> tensor<1x1x4x4x!ttype_f32, #layout>
      d2m.yield %st : (tensor<1x1x4x4x!ttype_f32, #layout>)
    } : tensor<1x1x4x4x!ttype_f32, #layout>
    %e_rm = d2m.empty() : tensor<128x128xf32>
    %rm = d2m.to_layout %fill, %e_rm : tensor<1x1x4x4x!ttype_f32, #layout> into tensor<128x128xf32> -> tensor<128x128xf32>
    %e_splat = d2m.empty() : tensor<1x1x4x4x!ttype_f32, #layout>
    %splat = d2m.to_layout %rm, %e_splat : tensor<128x128xf32> into tensor<1x1x4x4x!ttype_f32, #layout> -> tensor<1x1x4x4x!ttype_f32, #layout>
    %generic_out = d2m.empty() : tensor<1x1x4x4x!ttype_f32, #layout>
    // CHECK: d2m.generic
    %3 = d2m.generic {block_factors = [1, 1], grid = #ttcore.grid<1x1>, indexing_maps = [#map, #map, #map], iterator_types = [#parallel, #parallel], threads = [#d2m.thread<unified>]}
        ins(%arg0, %splat : tensor<1x1x4x4x!ttype_f32, #layout>, tensor<1x1x4x4x!ttype_f32, #layout>)
        outs(%generic_out : tensor<1x1x4x4x!ttype_f32, #layout>) {
    ^bb1:
      // CHECK: %[[S:.*]] = arith.constant 3.000000e+00 : f32
      %iter0 = d2m.block_index(0) : index
      %iter1 = d2m.block_index(1) : index
      %buffer0 = tensor.empty() : tensor<4x4x!ttype_f32>
      %buffer1 = tensor.empty() : tensor<4x4x!ttype_f32>
      %10 = d2m.remote_load %buffer0 %arg0[%iter0, %iter1] : tensor<4x4x!ttype_f32>, tensor<1x1x4x4x!ttype_f32, #layout> -> tensor<4x4x!ttype_f32>
      %11 = d2m.remote_load %buffer1 %splat[%iter0, %iter1] : tensor<4x4x!ttype_f32>, tensor<1x1x4x4x!ttype_f32, #layout> -> tensor<4x4x!ttype_f32>
      %12 = tensor.empty() : tensor<4x4x!ttype_f32>
      %13 = linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = ["parallel", "parallel"]} ins(%10, %11 : tensor<4x4x!ttype_f32>, tensor<4x4x!ttype_f32>) outs(%12 : tensor<4x4x!ttype_f32>) {
      ^bb0(%in: !ttype_f32, %in_0: !ttype_f32, %out: !ttype_f32):
        // CHECK: "d2m.tile_mul"(%{{.*}}, %[[S]]) : (!ttcore.tile<32x32, f32>, f32)
        %14 = "d2m.tile_mul"(%in, %in_0) : (!ttype_f32, !ttype_f32) -> !ttype_f32
        linalg.yield %14 : !ttype_f32
      } -> tensor<4x4x!ttype_f32>
      d2m.yield %13 : (tensor<4x4x!ttype_f32>)
    } : tensor<1x1x4x4x!ttype_f32, #layout>
    return %3 : tensor<1x1x4x4x!ttype_f32, #layout>
  }

  // CHECK-LABEL: func.func @test_non_scalar_op_no_change
  // Tile op has no scalar RHS: inner linalg must keep two tile operands (splat still
  // lowered via a separate fill generic with its own arith.constant).
  func.func @test_non_scalar_op_no_change(%arg0: tensor<1x1x4x4x!ttype_f32, #layout>) -> tensor<1x1x4x4x!ttype_f32, #layout> {
    %e0 = d2m.empty() : tensor<128x128xf32>
    %e1 = d2m.empty() : tensor<1x1x4x4x!ttype_f32, #layout>
    %to_fill = d2m.to_layout %e0, %e1 : tensor<128x128xf32> into tensor<1x1x4x4x!ttype_f32, #layout> -> tensor<1x1x4x4x!ttype_f32, #layout>
    %fill = d2m.generic {block_factors = [1, 1], grid = #ttcore.grid<1x1>, indexing_maps = [#map], iterator_types = [#parallel, #parallel], threads = [#d2m.thread<unified>]}
        ins()
        outs(%to_fill : tensor<1x1x4x4x!ttype_f32, #layout>) {
    ^bb0:
      %fe = tensor.empty() : tensor<4x4x!ttype_f32>
      %fl = linalg.generic {indexing_maps = [#map], iterator_types = ["parallel", "parallel"]} outs(%fe : tensor<4x4x!ttype_f32>) {
      ^bb0(%out: !ttype_f32):
        %cst_fill = arith.constant 2.500000e+00 : f32
        %ft = d2m.tile_fill(%cst_fill) : f32 -> <32x32, f32>
        linalg.yield %ft : !ttype_f32
      } -> tensor<4x4x!ttype_f32>
      %b0 = d2m.block_index(0) : index
      %b1 = d2m.block_index(1) : index
      %st = d2m.remote_store %to_fill[%b0, %b1] %fl : tensor<1x1x4x4x!ttype_f32, #layout>, tensor<4x4x!ttype_f32> -> tensor<1x1x4x4x!ttype_f32, #layout>
      d2m.yield %st : (tensor<1x1x4x4x!ttype_f32, #layout>)
    } : tensor<1x1x4x4x!ttype_f32, #layout>
    %e_rm = d2m.empty() : tensor<128x128xf32>
    %rm = d2m.to_layout %fill, %e_rm : tensor<1x1x4x4x!ttype_f32, #layout> into tensor<128x128xf32> -> tensor<128x128xf32>
    %e_splat = d2m.empty() : tensor<1x1x4x4x!ttype_f32, #layout>
    %splat = d2m.to_layout %rm, %e_splat : tensor<128x128xf32> into tensor<1x1x4x4x!ttype_f32, #layout> -> tensor<1x1x4x4x!ttype_f32, #layout>
    %generic_out = d2m.empty() : tensor<1x1x4x4x!ttype_f32, #layout>
    // CHECK: d2m.generic
    %3 = d2m.generic {block_factors = [1, 1], grid = #ttcore.grid<1x1>, indexing_maps = [#map, #map, #map], iterator_types = [#parallel, #parallel], threads = [#d2m.thread<unified>]}
        ins(%arg0, %splat : tensor<1x1x4x4x!ttype_f32, #layout>, tensor<1x1x4x4x!ttype_f32, #layout>)
        outs(%generic_out : tensor<1x1x4x4x!ttype_f32, #layout>) {
    ^bb1:
      %iter0 = d2m.block_index(0) : index
      %iter1 = d2m.block_index(1) : index
      %buffer0 = tensor.empty() : tensor<4x4x!ttype_f32>
      %buffer1 = tensor.empty() : tensor<4x4x!ttype_f32>
      %10 = d2m.remote_load %buffer0 %arg0[%iter0, %iter1] : tensor<4x4x!ttype_f32>, tensor<1x1x4x4x!ttype_f32, #layout> -> tensor<4x4x!ttype_f32>
      %11 = d2m.remote_load %buffer1 %splat[%iter0, %iter1] : tensor<4x4x!ttype_f32>, tensor<1x1x4x4x!ttype_f32, #layout> -> tensor<4x4x!ttype_f32>
      %12 = tensor.empty() : tensor<4x4x!ttype_f32>
      %13 = linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = ["parallel", "parallel"]} ins(%10, %11 : tensor<4x4x!ttype_f32>, tensor<4x4x!ttype_f32>) outs(%12 : tensor<4x4x!ttype_f32>) {
      ^bb0(%in: !ttype_f32, %in_0: !ttype_f32, %out: !ttype_f32):
        // CHECK: "d2m.tile_maximum"(%{{.*}}, %{{.*}}) : (!ttcore.tile<32x32, f32>, !ttcore.tile<32x32, f32>)
        %14 = "d2m.tile_maximum"(%in, %in_0) : (!ttype_f32, !ttype_f32) -> !ttype_f32
        linalg.yield %14 : !ttype_f32
      } -> tensor<4x4x!ttype_f32>
      d2m.yield %13 : (tensor<4x4x!ttype_f32>)
    } : tensor<1x1x4x4x!ttype_f32, #layout>
    return %3 : tensor<1x1x4x4x!ttype_f32, #layout>
  }
}
