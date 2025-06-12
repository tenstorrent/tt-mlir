// RUN: ttmlir-opt --tt-register-device --ttir-optimize-tensor-layout --split-input-file %s 2>&1 | FileCheck %s

#map1 = affine_map<(d0, d1) -> (d0, d1)>
#map2 = affine_map<(d0, d1) -> (d0, 0)>
#sc_map = affine_map<(d0, d1) -> (0, 0)>
#parallel = #tt.iterator_type<parallel>
#reduction = #tt.iterator_type<reduction>
#l1 = #tt.memory_space<l1>
#layout1 = #tt.metal_layout<(d0, d1) -> (d0, d1), undef, <1x1>, memref<8x12x!tt.tile<32x32, f32>, #l1>>
#layout2 = #tt.metal_layout<(d0, d1) -> (d0, d1), undef, <1x1>, memref<8x1x!tt.tile<32x32, f32>, #l1>>

func.func @reduce_large_grid(%arg0: tensor<256x384xf32, #layout1>, %arg1: tensor<256x384xf32, #layout1>) -> tensor<256x32xf32, #layout2> {
  %0 = ttir.empty() : tensor<256x32xf32, #layout2>
  // CHECK: %{{[a-z0-9_]+}} = ttir.to_layout
  // CHECK: %{{[a-z0-9_]+}} = ttir.to_layout
  %1 = "ttir.generic"(%arg0, %arg1, %0) <{
        block_factors = [1, 1],
        grid = #tt.grid<1x1>,
        indexing_maps = [#map1, #map1, #map2],
        iterator_types = [#parallel, #reduction],
        threads = [#ttir.thread<compute>],
        operandSegmentSizes = array<i32: 2, 1>
        }> ({
        // CHECK: ^compute0(%cb0: memref<1x2x!tt.tile<32x32, f32>, #l1>,
        // CHECK-SAME: %cb1: memref<1x2x!tt.tile<32x32, f32>, #l1>,
        // CHECK-SAME: %cb2: memref<1x1x!tt.tile<32x32, f32>, #l1>):
        ^bb0(%arg2: memref<8x12x!tt.tile<32x32, f32>, #l1>,
            %arg3: memref<8x12x!tt.tile<32x32, f32>, #l1>,
            %arg4: memref<8x1x!tt.tile<32x32, f32>, #l1>):
            linalg.generic {
                indexing_maps = [#map1, #sc_map, #map2],
                iterator_types = ["parallel", "parallel"]}
                ins(%arg2, %arg3: memref<8x12x!tt.tile<32x32, f32>, #l1>, memref<8x12x!tt.tile<32x32, f32>, #l1>)
                outs(%arg4: memref<8x1x!tt.tile<32x32, f32>, #l1>) {
                ^bb0(%a: !tt.tile<32x32, f32>, %b: !tt.tile<32x32, f32>, %c: !tt.tile<32x32, f32>):
                    %2 = "ttir.tile_reduce_max" (%a, %b) {reduce_dim = #ttir<reduce_dim R>} : (!tt.tile<32x32, f32>, !tt.tile<32x32, f32>) -> !tt.tile<32x32, f32>
                    linalg.yield %2: !tt.tile<32x32, f32>
            }
        "ttir.yield"() : () -> ()
        }) : (tensor<256x384xf32, #layout1>, tensor<256x384xf32, #layout1>, tensor<256x32xf32, #layout2>) -> tensor<256x32xf32, #layout2>
  // CHECK: %{{[a-z0-9_]+}} = ttir.to_layout
  return %1 : tensor<256x32xf32, #layout2>
}

// -----

#map1 = affine_map<(d0, d1) -> (d0, d1)>
#map2 = affine_map<(d0, d1) -> (d0, 0)>
#sc_map = affine_map<(d0, d1) -> (0, 0)>
#parallel = #tt.iterator_type<parallel>
#reduction = #tt.iterator_type<reduction>
#l1 = #tt.memory_space<l1>
#layout1 = #tt.metal_layout<(d0, d1) -> (d0, d1), undef, <1x1>, memref<1x19x!tt.tile<32x32, f32>, #l1>>
#layout2 = #tt.metal_layout<(d0, d1) -> (d0, d1), undef, <1x1>, memref<1x1x!tt.tile<32x32, f32>, #l1>>

func.func @reduce_prime(%arg0: tensor<32x608xf32, #layout1>, %arg1: tensor<32x608xf32, #layout1>) -> tensor<32x32xf32, #layout2> {
  %0 = ttir.empty() : tensor<32x32xf32, #layout2>
  %1 = "ttir.generic"(%arg0, %arg1, %0) <{
        block_factors = [1, 1],
        grid = #tt.grid<1x1>,
        indexing_maps = [#map1, #map1, #map2],
        iterator_types = [#parallel, #reduction],
        threads = [#ttir.thread<compute>],
        operandSegmentSizes = array<i32: 2, 1>
        }> ({
        // CHECK: ^compute0(%cb0: memref<1x19x!tt.tile<32x32, f32>, #l1>,
        // CHECK-SAME: %cb1: memref<1x19x!tt.tile<32x32, f32>, #l1>,
        // CHECK-SAME: %cb2: memref<1x1x!tt.tile<32x32, f32>, #l1>):
        ^bb0(%arg2: memref<1x19x!tt.tile<32x32, f32>, #l1>,
            %arg3: memref<1x19x!tt.tile<32x32, f32>, #l1>,
            %arg4: memref<1x1x!tt.tile<32x32, f32>, #l1>):
            linalg.generic {
                indexing_maps = [#map1, #sc_map, #map2],
                iterator_types = ["parallel", "parallel"]}
                ins(%arg2, %arg3: memref<1x19x!tt.tile<32x32, f32>, #l1>, memref<1x19x!tt.tile<32x32, f32>, #l1>)
                outs(%arg4: memref<1x1x!tt.tile<32x32, f32>, #l1>) {
                ^bb0(%a: !tt.tile<32x32, f32>, %b: !tt.tile<32x32, f32>, %c: !tt.tile<32x32, f32>):
                    %2 = "ttir.tile_reduce_max" (%a, %b) {reduce_dim = #ttir<reduce_dim R>} : (!tt.tile<32x32, f32>, !tt.tile<32x32, f32>) -> !tt.tile<32x32, f32>
                    linalg.yield %2: !tt.tile<32x32, f32>
            }
        "ttir.yield"() : () -> ()
        }) : (tensor<32x608xf32, #layout1>, tensor<32x608xf32, #layout1>, tensor<32x32xf32, #layout2>) -> tensor<32x32xf32, #layout2>
  return %1 : tensor<32x32xf32, #layout2>
}
