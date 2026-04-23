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
        operandSegmentSizes = array<i32: 2, 1, 0>
        }> ({
        ^bb0:
            %cb2 = d2m.get_cb(0) : !d2m.cb<tensor<2x4x!ttcore.tile<32x32, f32>, #l1_alias>>
            %cb3 = d2m.get_cb(1) : !d2m.cb<tensor<2x4x!ttcore.tile<32x32, f32>, #l1_alias>>
            %cb4 = d2m.get_cb(2) : !d2m.cb<tensor<2x4x!ttcore.tile<32x32, f32>, #l1_alias>>
            %arg2 = d2m.wait %cb2 : !d2m.cb<tensor<2x4x!ttcore.tile<32x32, f32>, #l1_alias>> -> tensor<2x4x!ttcore.tile<32x32, f32>, #l1_alias>
            %arg3 = d2m.wait %cb3 : !d2m.cb<tensor<2x4x!ttcore.tile<32x32, f32>, #l1_alias>> -> tensor<2x4x!ttcore.tile<32x32, f32>, #l1_alias>
            %arg4 = d2m.reserve %cb4 : !d2m.cb<tensor<2x4x!ttcore.tile<32x32, f32>, #l1_alias>> -> tensor<2x4x!ttcore.tile<32x32, f32>, #l1_alias>
            %result = linalg.generic {
                indexing_maps = [#map, #map, #map],
                iterator_types = ["parallel", "parallel"]}
                ins(%arg2, %arg3: tensor<2x4x!ttcore.tile<32x32, f32>, #l1_alias>, tensor<2x4x!ttcore.tile<32x32, f32>, #l1_alias>)
                outs(%arg4: tensor<2x4x!ttcore.tile<32x32, f32>, #l1_alias>) {
                ^bb0(%a: !ttcore.tile<32x32, f32>, %b: !ttcore.tile<32x32, f32>, %c: !ttcore.tile<32x32, f32>):
                    // CHECK: %{{[0-9]+}} = "d2m.tile_reduce_max"(%{{[a-z0-9_]+}}, %{{[a-z0-9_]+}}, %{{[a-z0-9_]+}}) <{reduce_dim = #d2m<reduce_dim R>}> : (!ttcore.tile<32x32, f32>, !ttcore.tile<32x32, f32>, !ttcore.tile<32x32, f32>) -> !ttcore.tile<32x32, f32>
                    %4 = "d2m.tile_reduce_max" (%a, %b, %c) {reduce_dim = #d2m<reduce_dim R>} : (!ttcore.tile<32x32, f32>, !ttcore.tile<32x32, f32>, !ttcore.tile<32x32, f32>) -> !ttcore.tile<32x32, f32>
                    linalg.yield %4: !ttcore.tile<32x32, f32>
            } -> tensor<2x4x!ttcore.tile<32x32, f32>, #l1_alias>
        // Return the updated output shard from the region.
        d2m.yield %0 : (tensor<64x128xf32>)
        }) : (tensor<64x128xf32>, tensor<64x128xf32>, tensor<64x128xf32>) -> tensor<64x128xf32>
    return %1 : tensor<64x128xf32>
  }

func.func @sfpu_reduce_sum(%arg0: tensor<64x128xsi32>) -> tensor<64x128xsi32> {
    %0 = d2m.empty() : tensor<64x128xsi32>
    %1 = "d2m.generic"(%arg0, %0) <{
        block_factors = [1, 1],
        grid = #ttcore.grid<1x1>,
        indexing_maps = [#map, #map],
        iterator_types = [#parallel, #parallel],
        threads = [#d2m.thread<compute>],
        operandSegmentSizes = array<i32: 1, 1, 0>
        }> ({
        ^bb0:
            %cb2 = d2m.get_cb(0) : !d2m.cb<tensor<2x4x!ttcore.tile<32x32, si32>, #l1_alias>>
            %cb3 = d2m.get_cb(1) : !d2m.cb<tensor<2x4x!ttcore.tile<32x32, si32>, #l1_alias>>
            %arg2 = d2m.wait %cb2 : !d2m.cb<tensor<2x4x!ttcore.tile<32x32, si32>, #l1_alias>> -> tensor<2x4x!ttcore.tile<32x32, si32>, #l1_alias>
            %arg4 = d2m.reserve %cb3 : !d2m.cb<tensor<2x4x!ttcore.tile<32x32, si32>, #l1_alias>> -> tensor<2x4x!ttcore.tile<32x32, si32>, #l1_alias>
            %result = linalg.generic {
                indexing_maps = [#map, #map],
                iterator_types = ["parallel", "parallel"]}
                ins(%arg2: tensor<2x4x!ttcore.tile<32x32, si32>, #l1_alias>)
                outs(%arg4: tensor<2x4x!ttcore.tile<32x32, si32>, #l1_alias>) {
                ^bb0(%a: !ttcore.tile<32x32, si32>, %c: !ttcore.tile<32x32, si32>):
                    // CHECK: %{{[0-9]+}} = "d2m.tile_sfpu_reduce_sum"(%{{[a-z0-9_]+}}, %{{[a-z0-9_]+}}) <{dst_scratch_index = -1 : i64, reduce_dim = #d2m<reduce_dim R>}> : (!ttcore.tile<32x32, si32>, !ttcore.tile<32x32, si32>) -> !ttcore.tile<32x32, si32>
                    %4 = "d2m.tile_sfpu_reduce_sum" (%a, %c) {reduce_dim = #d2m<reduce_dim R>} : (!ttcore.tile<32x32, si32>, !ttcore.tile<32x32, si32>) -> !ttcore.tile<32x32, si32>
                    linalg.yield %4: !ttcore.tile<32x32, si32>
            } -> tensor<2x4x!ttcore.tile<32x32, si32>, #l1_alias>
        d2m.yield %0 : (tensor<64x128xsi32>)
        }) : (tensor<64x128xsi32>, tensor<64x128xsi32>) -> tensor<64x128xsi32>
    return %1 : tensor<64x128xsi32>
  }

func.func @sfpu_reduce_max(%arg0: tensor<64x128xsi32>) -> tensor<64x128xsi32> {
    %0 = d2m.empty() : tensor<64x128xsi32>
    %1 = "d2m.generic"(%arg0, %0) <{
        block_factors = [1, 1],
        grid = #ttcore.grid<1x1>,
        indexing_maps = [#map, #map],
        iterator_types = [#parallel, #parallel],
        threads = [#d2m.thread<compute>],
        operandSegmentSizes = array<i32: 1, 1, 0>
        }> ({
        ^bb0:
            %cb2 = d2m.get_cb(0) : !d2m.cb<tensor<2x4x!ttcore.tile<32x32, si32>, #l1_alias>>
            %cb3 = d2m.get_cb(1) : !d2m.cb<tensor<2x4x!ttcore.tile<32x32, si32>, #l1_alias>>
            %arg2 = d2m.wait %cb2 : !d2m.cb<tensor<2x4x!ttcore.tile<32x32, si32>, #l1_alias>> -> tensor<2x4x!ttcore.tile<32x32, si32>, #l1_alias>
            %arg4 = d2m.reserve %cb3 : !d2m.cb<tensor<2x4x!ttcore.tile<32x32, si32>, #l1_alias>> -> tensor<2x4x!ttcore.tile<32x32, si32>, #l1_alias>
            %result = linalg.generic {
                indexing_maps = [#map, #map],
                iterator_types = ["parallel", "parallel"]}
                ins(%arg2: tensor<2x4x!ttcore.tile<32x32, si32>, #l1_alias>)
                outs(%arg4: tensor<2x4x!ttcore.tile<32x32, si32>, #l1_alias>) {
                ^bb0(%a: !ttcore.tile<32x32, si32>, %c: !ttcore.tile<32x32, si32>):
                    // CHECK: %{{[0-9]+}} = "d2m.tile_sfpu_reduce_max"(%{{[a-z0-9_]+}}, %{{[a-z0-9_]+}}) <{dst_scratch_index = -1 : i64, reduce_dim = #d2m<reduce_dim C>}> : (!ttcore.tile<32x32, si32>, !ttcore.tile<32x32, si32>) -> !ttcore.tile<32x32, si32>
                    %4 = "d2m.tile_sfpu_reduce_max" (%a, %c) {reduce_dim = #d2m<reduce_dim C>} : (!ttcore.tile<32x32, si32>, !ttcore.tile<32x32, si32>) -> !ttcore.tile<32x32, si32>
                    linalg.yield %4: !ttcore.tile<32x32, si32>
            } -> tensor<2x4x!ttcore.tile<32x32, si32>, #l1_alias>
        d2m.yield %0 : (tensor<64x128xsi32>)
        }) : (tensor<64x128xsi32>, tensor<64x128xsi32>) -> tensor<64x128xsi32>
    return %1 : tensor<64x128xsi32>
  }

func.func @reduce_mean(%arg0: tensor<64x128xf32>, %arg1: tensor<64x128xf32>) -> tensor<64x128xf32> {
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
            %cb2 = d2m.get_cb(0) : !d2m.cb<tensor<2x4x!ttcore.tile<32x32, f32>, #l1_alias>>
            %cb3 = d2m.get_cb(1) : !d2m.cb<tensor<2x4x!ttcore.tile<32x32, f32>, #l1_alias>>
            %cb4 = d2m.get_cb(2) : !d2m.cb<tensor<2x4x!ttcore.tile<32x32, f32>, #l1_alias>>
            %arg2 = d2m.wait %cb2 : !d2m.cb<tensor<2x4x!ttcore.tile<32x32, f32>, #l1_alias>> -> tensor<2x4x!ttcore.tile<32x32, f32>, #l1_alias>
            %arg3 = d2m.wait %cb3 : !d2m.cb<tensor<2x4x!ttcore.tile<32x32, f32>, #l1_alias>> -> tensor<2x4x!ttcore.tile<32x32, f32>, #l1_alias>
            %arg4 = d2m.reserve %cb4 : !d2m.cb<tensor<2x4x!ttcore.tile<32x32, f32>, #l1_alias>> -> tensor<2x4x!ttcore.tile<32x32, f32>, #l1_alias>
            %result = linalg.generic {
                indexing_maps = [#map, #map, #map],
                iterator_types = ["parallel", "parallel"]}
                ins(%arg2, %arg3: tensor<2x4x!ttcore.tile<32x32, f32>, #l1_alias>, tensor<2x4x!ttcore.tile<32x32, f32>, #l1_alias>)
                outs(%arg4: tensor<2x4x!ttcore.tile<32x32, f32>, #l1_alias>) {
                ^bb0(%a: !ttcore.tile<32x32, f32>, %b: !ttcore.tile<32x32, f32>, %c: !ttcore.tile<32x32, f32>):
                    // CHECK: %{{[0-9]+}} = "d2m.tile_reduce_mean"(%{{[a-z0-9_]+}}, %{{[a-z0-9_]+}}, %{{[a-z0-9_]+}}) <{reduce_dim = #d2m<reduce_dim C>}> : (!ttcore.tile<32x32, f32>, !ttcore.tile<32x32, f32>, !ttcore.tile<32x32, f32>) -> !ttcore.tile<32x32, f32>
                    %4 = "d2m.tile_reduce_mean" (%a, %b, %c) {reduce_dim = #d2m<reduce_dim C>} : (!ttcore.tile<32x32, f32>, !ttcore.tile<32x32, f32>, !ttcore.tile<32x32, f32>) -> !ttcore.tile<32x32, f32>
                    linalg.yield %4: !ttcore.tile<32x32, f32>
            } -> tensor<2x4x!ttcore.tile<32x32, f32>, #l1_alias>
        d2m.yield %0 : (tensor<64x128xf32>)
        }) : (tensor<64x128xf32>, tensor<64x128xf32>, tensor<64x128xf32>) -> tensor<64x128xf32>
    return %1 : tensor<64x128xf32>
  }
}
