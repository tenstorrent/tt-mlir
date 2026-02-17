// RUN: ttmlir-opt --canonicalize %s --split-input-file | FileCheck %s

#l1_ = #ttcore.memory_space<l1>
#map = affine_map<(d0, d1) -> (d0, d1)>
#parallel = #ttcore.iterator_type<parallel>
#layout = #ttcore.metal_layout<logical_shape = 128x128, dim_alignments = 32x32, collapsed_intervals = dense<[[0, 1], [1, 2]]> : tensor<2x2xi64>, undef, l1, sharded, index_map = map(0)>

!ttype_f32 = !ttcore.tile<32x32, f32>

// CHECK-LABEL: func.func @fold_tile_add_zero
func.func @fold_tile_add_zero(%arg0: tensor<1x1x4x4x!ttype_f32, #layout>) -> tensor<1x1x4x4x!ttype_f32, #layout> {
  %0 = d2m.empty() : tensor<1x1x4x4x!ttype_f32, #layout>
  %1 = d2m.generic {block_factors = [1, 1], grid = #ttcore.grid<1x1>,
      indexing_maps = [#map, #map], iterator_types = [#parallel, #parallel],
      threads = [#d2m.thread<unified>]}
      ins(%arg0 : tensor<1x1x4x4x!ttype_f32, #layout>)
      outs(%0 : tensor<1x1x4x4x!ttype_f32, #layout>) {
  ^unified0(%cb0: !d2m.cb<tensor<4x4x!ttype_f32>>, %cb1: !d2m.cb<tensor<4x4x!ttype_f32>>):
    %buffer = tensor.empty() : tensor<4x4x!ttype_f32>
    %iter0 = d2m.block_index(0) : index
    %iter1 = d2m.block_index(1) : index
    %load = d2m.remote_load %buffer %arg0[%iter0, %iter1] : tensor<4x4x!ttype_f32>, tensor<1x1x4x4x!ttype_f32, #layout> -> tensor<4x4x!ttype_f32>
    %reserve = d2m.reserve %cb1 : <tensor<4x4x!ttype_f32>> -> tensor<4x4x!ttype_f32>
    %result = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel", "parallel"]}
        ins(%load : tensor<4x4x!ttype_f32>) outs(%reserve : tensor<4x4x!ttype_f32>) {
    ^bb0(%in: !ttype_f32, %out: !ttype_f32):
      %zero = arith.constant 0.0 : f32
      %one = arith.constant 1.0 : f32
      // CHECK-NOT: d2m.tile_add
      // CHECK-NOT: d2m.tile_sub
      // CHECK-NOT: d2m.tile_mul
      // CHECK-NOT: d2m.tile_div
      // CHECK-NOT: d2m.tile_pow
      %added = "d2m.tile_add"(%in, %zero) : (!ttype_f32, f32) -> !ttype_f32
      %subtract = "d2m.tile_sub"(%added, %zero) : (!ttype_f32, f32) -> !ttype_f32
      %multiply = "d2m.tile_mul"(%subtract, %one) : (!ttype_f32, f32) -> !ttype_f32
      %divide = "d2m.tile_div"(%multiply, %one) : (!ttype_f32, f32) -> !ttype_f32
      %power = "d2m.tile_pow"(%divide, %one) : (!ttype_f32, f32) -> !ttype_f32
      linalg.yield %power : !ttype_f32
    } -> tensor<4x4x!ttype_f32>
    d2m.yield %result : (tensor<4x4x!ttype_f32>)
  } : tensor<1x1x4x4x!ttype_f32, #layout>
  return %1 : tensor<1x1x4x4x!ttype_f32, #layout>
}
