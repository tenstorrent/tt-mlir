// RUN: ttmlir-opt %s | FileCheck %s

#map = affine_map<(d0, d1) -> (d0, d1)>
#parallel = #tt.iterator_type<parallel>
#l1_alias = #tt.memory_space<l1>

module {
func.func @reduce_max(%arg0: tensor<64x128xf32>, %arg1: tensor<64x128xf32>) -> tensor<64x128xf32> {
    %0 = tensor.empty() : tensor<64x128xf32>
    %1 = "ttir.generic"(%arg0, %arg1, %0) <{
        grid = #tt.grid<1x1>,
        indexing_maps = [#map, #map, #map],
        iterator_types = [#parallel, #parallel],
        operandSegmentSizes = array<i32: 2, 0, 1>
        }> ({
        ^bb0(%arg2: memref<2x4x!tt.tile<32x32, f32>, #l1_alias>,
            %arg3: memref<2x4x!tt.tile<32x32, f32>, #l1_alias>,
            %arg4: memref<2x4x!tt.tile<32x32, f32>, #l1_alias>):
            linalg.generic {
                indexing_maps = [#map, #map],
                iterator_types = ["parallel", "parallel"]}
                ins(%arg2: memref<2x4x!tt.tile<32x32, f32>, #l1_alias>)
                outs(%arg4: memref<2x4x!tt.tile<32x32, f32>, #l1_alias>) {
                ^bb0(%a: !tt.tile<32x32, f32>, %b: !tt.tile<32x32, f32>):
                    // CHECK: %{{[0-9]+}} = "ttir.tile_reduce_max"(%{{[a-z]+}}, %{{[a-z]+}}) <{reduce_dim = #ttir<reduce_dim R>}> : (!tt.tile<32x32, f32>, !tt.tile<32x32, f32>) -> !tt.tile<32x32, f32>
                    %4 = "ttir.tile_reduce_max" (%a, %b) {reduce_dim = #ttir<reduce_dim R>} : (!tt.tile<32x32, f32>, !tt.tile<32x32, f32>) -> !tt.tile<32x32, f32>
                    linalg.yield %4: !tt.tile<32x32, f32>
            }
        "ttir.yield"() : () -> ()
        }) : (tensor<64x128xf32>, tensor<64x128xf32>, tensor<64x128xf32>) -> tensor<64x128xf32>
    return %1 : tensor<64x128xf32>
  }
}
