// RUN: ttmlir-opt -o %t %s
// RUN: FileCheck %s --input-file=%t

#map = affine_map<(d0, d1) -> (d0, d1)>
#parallel = #ttcore.iterator_type<parallel>
#l1_alias = #ttcore.memory_space<l1>

module {
func.func @reduce_max(%arg0: tensor<64x128xf32>, %arg1: tensor<64x128xf32>) -> tensor<64x128xf32> {
    %0 = d2m.empty() : tensor<64x128xf32>
    %1 = "d2m.generic"(%arg0, %arg1, %0) <{
        block_factors = [1, 1],
        grid = #ttcore.grid<1x1>,
        indexing_maps = [#map, #map, #map],
        iterator_types = [#parallel, #parallel],
        threads = [#d2m.thread<compute>],
        operandSegmentSizes = array<i32: 2, 1>
        }> ({
        ^bb0(%arg2: memref<2x4x!ttcore.tile<32x32, f32>, #l1_alias>,
            %arg3: memref<2x4x!ttcore.tile<32x32, f32>, #l1_alias>,
            %arg4: memref<2x4x!ttcore.tile<32x32, f32>, #l1_alias>):
            linalg.generic {
                indexing_maps = [#map, #map],
                iterator_types = ["parallel", "parallel"]}
                ins(%arg2: memref<2x4x!ttcore.tile<32x32, f32>, #l1_alias>)
                outs(%arg4: memref<2x4x!ttcore.tile<32x32, f32>, #l1_alias>) {
                ^bb0(%a: !ttcore.tile<32x32, f32>, %b: !ttcore.tile<32x32, f32>):
                    // CHECK: %{{[0-9]+}} = "d2m.tile_reduce_max"(%{{[a-z]+}}, %{{[a-z]+}}) <{reduce_dim = #d2m<reduce_dim R>}> : (!ttcore.tile<32x32, f32>, !ttcore.tile<32x32, f32>) -> !ttcore.tile<32x32, f32>
                    %4 = "d2m.tile_reduce_max" (%a, %b) {reduce_dim = #d2m<reduce_dim R>} : (!ttcore.tile<32x32, f32>, !ttcore.tile<32x32, f32>) -> !ttcore.tile<32x32, f32>
                    linalg.yield %4: !ttcore.tile<32x32, f32>
            }
        // Return the updated output shard from the region.
        d2m.yield %arg4 : (memref<2x4x!ttcore.tile<32x32, f32>, #l1_alias>)
        }) : (tensor<64x128xf32>, tensor<64x128xf32>, tensor<64x128xf32>) -> tensor<64x128xf32>
    return %1 : tensor<64x128xf32>
  }
}
