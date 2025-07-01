// RUN: not ttmlir-opt --split-input-file %s 2>&1 | FileCheck %s

#map = affine_map<(d0, d1) -> (d0, d1)>
#parallel = #ttcore.iterator_type<parallel>
#l1_alias = #ttcore.memory_space<l1>

func.func @reduce_dim_arg(%arg0: tensor<64x128xf32>, %arg1: tensor<64x128xf32>) -> tensor<64x128xf32> {
  %0 = ttir.empty() : tensor<64x128xf32>
  %1 = "ttir.generic"(%arg0, %arg1, %0) <{
      block_factors = [1, 1],
      grid = #ttcore.grid<1x1>,
      indexing_maps = [#map, #map, #map],
      iterator_types = [#parallel, #parallel],
      threads = [#ttir.thread<compute>],
      operandSegmentSizes = array<i32: 2, 1>
      }> ({
      ^bb0(%arg2: memref<2x4x!ttcore.tile<32x32, f32>, #l1_alias>,
          %arg3: memref<2x4x!ttcore.tile<32x32, f32>, #l1_alias>,
          %arg4: memref<2x4x!ttcore.tile<32x32, f32>, #l1_alias>):
          linalg.generic {
              indexing_maps = [#map, #map, #map],
              iterator_types = ["parallel", "parallel"]
              }
              ins(%arg2, %arg3: memref<2x4x!ttcore.tile<32x32, f32>, #l1_alias>, memref<2x4x!ttcore.tile<32x32, f32>, #l1_alias>)
              outs(%arg4: memref<2x4x!ttcore.tile<32x32, f32>, #l1_alias>) {
              ^bb0(%a: !ttcore.tile<32x32, f32>, %b: !ttcore.tile<32x32, f32>, %c: !ttcore.tile<32x32, f32>):
                  // CHECK: error: 'ttir.tile_reduce_max' op requires attribute 'reduce_dim'
                  %4 = "ttir.tile_reduce_max" (%a, %b, %c) : (!ttcore.tile<32x32, f32>, !ttcore.tile<32x32, f32>, !ttcore.tile<32x32, f32>) -> !ttcore.tile<32x32, f32>
                  linalg.yield %4: !ttcore.tile<32x32, f32>
              }
      ttir.yield %arg4 : (memref<2x4x!ttcore.tile<32x32, f32>, #l1_alias>)
      }) : (tensor<64x128xf32>, tensor<64x128xf32>, tensor<64x128xf32>) -> tensor<64x128xf32>
  return %1 : tensor<64x128xf32>
}
