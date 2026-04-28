// RUN: not ttmlir-opt --split-input-file %s 2>&1 | FileCheck %s

#map = affine_map<(d0, d1) -> (d0, d1)>
#parallel = #ttcore.iterator_type<parallel>
#l1_alias = #ttcore.memory_space<l1>

func.func @reduce_dim_arg(%arg0: tensor<64x128xf32>, %arg1: tensor<64x128xf32>) -> tensor<64x128xf32> {
  %0 = d2m.empty() : tensor<64x128xf32>
  %1 = "d2m.generic"(%arg0, %arg1, %0) <{
      block_factors = [1, 1],
      grid = #ttcore.grid<1x1>,
      indexing_maps = [#map, #map, #map],
      iterator_types = [#parallel, #parallel],
      threads = [#d2m.thread<compute>],
      operandSegmentSizes = array<i32: 2, 1, 0>
      }> ({
      ^bb0:
          %cb0 = d2m.get_cb(0) : !d2m.cb<tensor<2x4x!ttcore.tile<32x32, f32>, #l1_alias>>
          %cb1 = d2m.get_cb(1) : !d2m.cb<tensor<2x4x!ttcore.tile<32x32, f32>, #l1_alias>>
          %cb2 = d2m.get_cb(2) : !d2m.cb<tensor<2x4x!ttcore.tile<32x32, f32>, #l1_alias>>
          %arg2 = d2m.wait %cb0 : !d2m.cb<tensor<2x4x!ttcore.tile<32x32, f32>, #l1_alias>> -> tensor<2x4x!ttcore.tile<32x32, f32>, #l1_alias>
          %arg3 = d2m.wait %cb1 : !d2m.cb<tensor<2x4x!ttcore.tile<32x32, f32>, #l1_alias>> -> tensor<2x4x!ttcore.tile<32x32, f32>, #l1_alias>
          %arg4 = d2m.reserve %cb2 : !d2m.cb<tensor<2x4x!ttcore.tile<32x32, f32>, #l1_alias>> -> tensor<2x4x!ttcore.tile<32x32, f32>, #l1_alias>
          linalg.generic {
              indexing_maps = [#map, #map, #map],
              iterator_types = ["parallel", "parallel"]
              }
              ins(%arg2, %arg3: tensor<2x4x!ttcore.tile<32x32, f32>, #l1_alias>, tensor<2x4x!ttcore.tile<32x32, f32>, #l1_alias>)
              outs(%arg4: tensor<2x4x!ttcore.tile<32x32, f32>, #l1_alias>) {
              ^bb0(%a: !ttcore.tile<32x32, f32>, %b: !ttcore.tile<32x32, f32>, %c: !ttcore.tile<32x32, f32>):
                  // CHECK: error: 'd2m.tile_reduce_max' op requires attribute 'reduce_dim'
                  %4 = "d2m.tile_reduce_max" (%a, %b, %c) : (!ttcore.tile<32x32, f32>, !ttcore.tile<32x32, f32>, !ttcore.tile<32x32, f32>) -> !ttcore.tile<32x32, f32>
                  linalg.yield %4: !ttcore.tile<32x32, f32>
              }
      d2m.yield %0 : (tensor<64x128xf32>)
      }) : (tensor<64x128xf32>, tensor<64x128xf32>, tensor<64x128xf32>) -> tensor<64x128xf32>
  return %1 : tensor<64x128xf32>
}

// -----

#map = affine_map<(d0, d1) -> (d0, d1)>
#parallel = #ttcore.iterator_type<parallel>
#l1_alias = #ttcore.memory_space<l1>

func.func @reduce_mean_missing_dim(%arg0: tensor<64x128xf32>, %arg1: tensor<64x128xf32>) -> tensor<64x128xf32> {
  %0 = d2m.empty() : tensor<64x128xf32>
  %1 = "d2m.generic"(%arg0, %arg1, %0) <{
      block_factors = [1, 1],
      grid = #ttcore.grid<1x1>,
      indexing_maps = [#map, #map, #map],
      iterator_types = [#parallel, #parallel],
      threads = [#d2m.thread<compute>],
      operandSegmentSizes = array<i32: 2, 1, 0>
      }> ({
      ^bb0:
          %cb0 = d2m.get_cb(0) : !d2m.cb<tensor<2x4x!ttcore.tile<32x32, f32>, #l1_alias>>
          %cb1 = d2m.get_cb(1) : !d2m.cb<tensor<2x4x!ttcore.tile<32x32, f32>, #l1_alias>>
          %cb2 = d2m.get_cb(2) : !d2m.cb<tensor<2x4x!ttcore.tile<32x32, f32>, #l1_alias>>
          %arg2 = d2m.wait %cb0 : !d2m.cb<tensor<2x4x!ttcore.tile<32x32, f32>, #l1_alias>> -> tensor<2x4x!ttcore.tile<32x32, f32>, #l1_alias>
          %arg3 = d2m.wait %cb1 : !d2m.cb<tensor<2x4x!ttcore.tile<32x32, f32>, #l1_alias>> -> tensor<2x4x!ttcore.tile<32x32, f32>, #l1_alias>
          %arg4 = d2m.reserve %cb2 : !d2m.cb<tensor<2x4x!ttcore.tile<32x32, f32>, #l1_alias>> -> tensor<2x4x!ttcore.tile<32x32, f32>, #l1_alias>
          linalg.generic {
              indexing_maps = [#map, #map, #map],
              iterator_types = ["parallel", "parallel"]
              }
              ins(%arg2, %arg3: tensor<2x4x!ttcore.tile<32x32, f32>, #l1_alias>, tensor<2x4x!ttcore.tile<32x32, f32>, #l1_alias>)
              outs(%arg4: tensor<2x4x!ttcore.tile<32x32, f32>, #l1_alias>) {
              ^bb0(%a: !ttcore.tile<32x32, f32>, %b: !ttcore.tile<32x32, f32>, %c: !ttcore.tile<32x32, f32>):
                  // CHECK: error: 'd2m.tile_reduce_mean' op requires attribute 'reduce_dim'
                  %4 = "d2m.tile_reduce_mean" (%a, %b, %c) : (!ttcore.tile<32x32, f32>, !ttcore.tile<32x32, f32>, !ttcore.tile<32x32, f32>) -> !ttcore.tile<32x32, f32>
                  linalg.yield %4: !ttcore.tile<32x32, f32>
              }
      d2m.yield %0 : (tensor<64x128xf32>)
      }) : (tensor<64x128xf32>, tensor<64x128xf32>, tensor<64x128xf32>) -> tensor<64x128xf32>
  return %1 : tensor<64x128xf32>
}

// -----

#map = affine_map<(d0, d1) -> (d0, d1)>
#parallel = #ttcore.iterator_type<parallel>
#l1_alias = #ttcore.memory_space<l1>

// FPU tile_reduce_* ops are float-only; using an integer tile must fail verification.
func.func @reduce_sum_rejects_int(%arg0: tensor<64x128xsi32>, %arg1: tensor<64x128xsi32>) -> tensor<64x128xsi32> {
  %0 = d2m.empty() : tensor<64x128xsi32>
  %1 = "d2m.generic"(%arg0, %arg1, %0) <{
      block_factors = [1, 1],
      grid = #ttcore.grid<1x1>,
      indexing_maps = [#map, #map, #map],
      iterator_types = [#parallel, #parallel],
      threads = [#d2m.thread<compute>],
      operandSegmentSizes = array<i32: 2, 1, 0>
      }> ({
      ^bb0:
          %cb0 = d2m.get_cb(0) : !d2m.cb<tensor<2x4x!ttcore.tile<32x32, si32>, #l1_alias>>
          %cb1 = d2m.get_cb(1) : !d2m.cb<tensor<2x4x!ttcore.tile<32x32, si32>, #l1_alias>>
          %cb2 = d2m.get_cb(2) : !d2m.cb<tensor<2x4x!ttcore.tile<32x32, si32>, #l1_alias>>
          %arg2 = d2m.wait %cb0 : !d2m.cb<tensor<2x4x!ttcore.tile<32x32, si32>, #l1_alias>> -> tensor<2x4x!ttcore.tile<32x32, si32>, #l1_alias>
          %arg3 = d2m.wait %cb1 : !d2m.cb<tensor<2x4x!ttcore.tile<32x32, si32>, #l1_alias>> -> tensor<2x4x!ttcore.tile<32x32, si32>, #l1_alias>
          %arg4 = d2m.reserve %cb2 : !d2m.cb<tensor<2x4x!ttcore.tile<32x32, si32>, #l1_alias>> -> tensor<2x4x!ttcore.tile<32x32, si32>, #l1_alias>
          linalg.generic {
              indexing_maps = [#map, #map, #map],
              iterator_types = ["parallel", "parallel"]
              }
              ins(%arg2, %arg3: tensor<2x4x!ttcore.tile<32x32, si32>, #l1_alias>, tensor<2x4x!ttcore.tile<32x32, si32>, #l1_alias>)
              outs(%arg4: tensor<2x4x!ttcore.tile<32x32, si32>, #l1_alias>) {
              ^bb0(%a: !ttcore.tile<32x32, si32>, %b: !ttcore.tile<32x32, si32>, %c: !ttcore.tile<32x32, si32>):
                  // CHECK: error: 'd2m.tile_reduce_sum' op requires float tile element types
                  %4 = "d2m.tile_reduce_sum" (%a, %b, %c) {reduce_dim = #d2m<reduce_dim R>} : (!ttcore.tile<32x32, si32>, !ttcore.tile<32x32, si32>, !ttcore.tile<32x32, si32>) -> !ttcore.tile<32x32, si32>
                  linalg.yield %4: !ttcore.tile<32x32, si32>
              }
      d2m.yield %0 : (tensor<64x128xsi32>)
      }) : (tensor<64x128xsi32>, tensor<64x128xsi32>, tensor<64x128xsi32>) -> tensor<64x128xsi32>
  return %1 : tensor<64x128xsi32>
}

// -----

#map = affine_map<(d0, d1) -> (d0, d1)>
#parallel = #ttcore.iterator_type<parallel>
#l1_alias = #ttcore.memory_space<l1>

// SFPU tile_sfpu_reduce_* ops are integer-only; using a float tile must fail verification.
func.func @sfpu_reduce_max_rejects_float(%arg0: tensor<64x128xf32>) -> tensor<64x128xf32> {
  %0 = d2m.empty() : tensor<64x128xf32>
  %1 = "d2m.generic"(%arg0, %0) <{
      block_factors = [1, 1],
      grid = #ttcore.grid<1x1>,
      indexing_maps = [#map, #map],
      iterator_types = [#parallel, #parallel],
      threads = [#d2m.thread<compute>],
      operandSegmentSizes = array<i32: 1, 1, 0>
      }> ({
      ^bb0:
          %cb0 = d2m.get_cb(0) : !d2m.cb<tensor<2x4x!ttcore.tile<32x32, f32>, #l1_alias>>
          %cb1 = d2m.get_cb(1) : !d2m.cb<tensor<2x4x!ttcore.tile<32x32, f32>, #l1_alias>>
          %arg2 = d2m.wait %cb0 : !d2m.cb<tensor<2x4x!ttcore.tile<32x32, f32>, #l1_alias>> -> tensor<2x4x!ttcore.tile<32x32, f32>, #l1_alias>
          %arg4 = d2m.reserve %cb1 : !d2m.cb<tensor<2x4x!ttcore.tile<32x32, f32>, #l1_alias>> -> tensor<2x4x!ttcore.tile<32x32, f32>, #l1_alias>
          linalg.generic {
              indexing_maps = [#map, #map],
              iterator_types = ["parallel", "parallel"]
              }
              ins(%arg2: tensor<2x4x!ttcore.tile<32x32, f32>, #l1_alias>)
              outs(%arg4: tensor<2x4x!ttcore.tile<32x32, f32>, #l1_alias>) {
              ^bb0(%a: !ttcore.tile<32x32, f32>, %c: !ttcore.tile<32x32, f32>):
                  // CHECK: error: 'd2m.tile_sfpu_reduce_max' op requires signed 32-bit integer tile element types
                  %4 = "d2m.tile_sfpu_reduce_max" (%a, %c) {reduce_dim = #d2m<reduce_dim R>} : (!ttcore.tile<32x32, f32>, !ttcore.tile<32x32, f32>) -> !ttcore.tile<32x32, f32>
                  linalg.yield %4: !ttcore.tile<32x32, f32>
              }
      d2m.yield %0 : (tensor<64x128xf32>)
      }) : (tensor<64x128xf32>, tensor<64x128xf32>) -> tensor<64x128xf32>
  return %1 : tensor<64x128xf32>
}
