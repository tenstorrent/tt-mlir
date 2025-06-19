// RUN: ttmlir-opt --split-input-file --ttcore-register-device --ttir-prepare-tensors-for-bufferization %s | FileCheck %s

#map1 = affine_map<(d0, d1) -> (d0, d1)>
#map2 = affine_map<(d0, d1) -> (d0, 0)>
#sc_map = affine_map<(d0, d1) -> (0, 0)>
#parallel = #ttcore.iterator_type<parallel>
#reduction = #ttcore.iterator_type<reduction>
#l1_ = #ttcore.memory_space<l1>
#layout1 = #ttcore.metal_layout<(d0, d1) -> (d0, d1), undef, <1x1>, memref<8x12x!ttcore.tile<32x32, f32>, #l1_>>
#layout2 = #ttcore.metal_layout<(d0, d1) -> (d0, d1), undef, <1x1>, memref<8x1x!ttcore.tile<32x32, f32>, #l1_>>

func.func @main(%arg0: tensor<256x384xf32, #layout1>, %arg1: tensor<256x384xf32, #layout1>) -> tensor<256x32xf32, #layout2> {
  // CHECK: ttir.empty() : tensor<1x1x8x1x!ttcore.tile<32x32, f32>, #layout1>
  %0 = ttir.empty() : tensor<256x32xf32, #layout2>
  // CHECK: ins({{.*}} : tensor<1x1x8x12x!ttcore.tile<32x32, f32>, #layout>, tensor<1x1x8x12x!ttcore.tile<32x32, f32>, #layout>)
  // CHECK-NEXT: outs({{.*}} : tensor<1x1x8x1x!ttcore.tile<32x32, f32>, #layout1>)
  %1 = "ttir.generic"(%arg0, %arg1, %0) <{
        block_factors = [1, 1],
        grid = #ttcore.grid<1x1>,
        indexing_maps = [#map1, #map1, #map2],
        iterator_types = [#parallel, #reduction],
        threads = [#ttir.thread<compute>],
        operandSegmentSizes = array<i32: 2, 1>
        }> ({
        ^bb0(%arg2: memref<8x12x!ttcore.tile<32x32, f32>, #l1_>,
            %arg3: memref<8x12x!ttcore.tile<32x32, f32>, #l1_>,
            %arg4: memref<8x1x!ttcore.tile<32x32, f32>, #l1_>):
        "ttir.yield"() : () -> ()
        }) : (tensor<256x384xf32, #layout1>, tensor<256x384xf32, #layout1>, tensor<256x32xf32, #layout2>) -> tensor<256x32xf32, #layout2>
  return %1 : tensor<256x32xf32, #layout2>
}

// -----

#map1 = affine_map<(d0, d1) -> (d0, d1)>
#map2 = affine_map<(d0, d1) -> (d0, 0)>
#sc_map = affine_map<(d0, d1) -> (0, 0)>
#parallel = #ttcore.iterator_type<parallel>
#reduction = #ttcore.iterator_type<reduction>
#l1_ = #ttcore.memory_space<l1>
#layout1 = #ttcore.metal_layout<(d0, d1) -> (d0, d1), undef, <4x3>, memref<2x4x!ttcore.tile<32x32, f32>, #l1_>>
#layout2 = #ttcore.metal_layout<(d0, d1) -> (d0, d1), undef, <4x1>, memref<2x1x!ttcore.tile<32x32, f32>, #l1_>>

func.func @main(%arg0: tensor<256x384xf32, #layout1>, %arg1: tensor<256x384xf32, #layout1>) -> tensor<256x32xf32, #layout2> {
  // CHECK: ttir.empty() : tensor<4x1x2x1x!ttcore.tile<32x32, f32>, #layout1>
  %0 = ttir.empty() : tensor<256x32xf32, #layout2>
  // CHECK: ins({{.*}} : tensor<4x3x2x4x!ttcore.tile<32x32, f32>, #layout>, tensor<4x3x2x4x!ttcore.tile<32x32, f32>, #layout>)
  // CHECK-NEXT: outs({{.*}} : tensor<4x1x2x1x!ttcore.tile<32x32, f32>, #layout1>)
  %1 = "ttir.generic"(%arg0, %arg1, %0) <{
        block_factors = [1, 1],
        grid = #ttcore.grid<4x1>,
        indexing_maps = [#map1, #map1, #map2],
        iterator_types = [#parallel, #reduction],
        threads = [#ttir.thread<compute>],
        operandSegmentSizes = array<i32: 2, 1>
        }> ({
        ^bb0(%arg2: memref<2x4x!ttcore.tile<32x32, f32>, #l1_>,
            %arg3: memref<2x4x!ttcore.tile<32x32, f32>, #l1_>,
            %arg4: memref<2x1x!ttcore.tile<32x32, f32>, #l1_>):
        "ttir.yield"() : () -> ()
        }) : (tensor<256x384xf32, #layout1>, tensor<256x384xf32, #layout1>, tensor<256x32xf32, #layout2>) -> tensor<256x32xf32, #layout2>
  return %1 : tensor<256x32xf32, #layout2>
}
