// How to express dram queues/streams
// How to express multi-device grid
// Core to core communication
// device to device communication
#map0 = affine_map<(c0, c1) -> (c0, c1)>

#map1 = affine_map<(d0, d1, c0, c1) -> (c1, c0)>

#map2 = affine_map<(c0, c1, k0) -> (c0, k0)>
#map3 = affine_map<(c0, c1, k0) -> (k0, c1)>
#map4 = affine_map<(c0, c1, k0) -> (c0, c1)>

#map_explicit = affine_map<(c0, c1) -> ()>

iterator_types = ["parallel", "systolic", "broadcast", "explicit"]

#grid0 = !tt.grid<2x2x2, block, core_mapping = affine_map<(c0, c1, c2) -> (c0 * c1, c2)>>

#layout0 = !tt.layout<strides<0x(4*128)x0x0>, shape<16x0x64x128>, pad_val<undef>, #grid0, memref<8x1x2x!tt.tile<32 x 32, f32>, #l1>>
%2 = tensor<16x15x4x128xf32, #layout0>

// Layout
%post = "tt.layout" %2_rm into %a : tensor<64x128, #layout0> -> tensor<64x128, #layout1>

//
// Eltwise
//
#map0 = affine_map<(c0, c1) -> (c0, c1)>
#grid0 = !tt.grid<2x2, block, core_mapping = affine_map<(c0, c1) -> (c0, c1)>>
#layout0 = !tt.layout<strides<0x0x0x0>, shape<64x128>, pad_val<undef>, #grid0, memref<1x2x!tt.tile<32 x 32, f32>, #l1>>
%0 = tensor<64x128xf32, #layout0>
%1 = tensor<64x128xf32, #layout0>
%2 = tensor<64x128xf32, #layout0>
%6 = "tt.dispatch"(%0, %1, %2) <{#grid0, indexing_maps = [#map0, #map0, #map0], iterator_types = ["parallel", "parallel"], buffer_accesses = [#b0, #b0, #b0]}> ({
^bb0(%arg0: memref<1x2x!tt.tile<32 x 32, f32>, #l1>, %arg1: memref<1x2x!tt.tile<32 x 32, f32>, #l1>, %arg2: memref<1x2x!tt.tile<32 x 32, f32>, #l1>):
  %8 = tt.builtin "add"(%arg0, %arg1) : (memref<1x2x!tt.tile<32 x 32, f32>, #l1>, memref<1x2x!tt.tile<32 x 32, f32>, #l1>) -> memref<1x2x!tt.tile<32 x 32, f32>, #l1>
  "tt.yield"(%8) : (memref<1x2x!tt.tile<32 x 32, f32>, #l1>) -> ()
}) : tensor<64x128xf32, #layout0>

//
// Eltwise (LHS DRAM)
//
#map0 = affine_map<(c0, c1) -> (c0, c1)>
#grid0 = !tt.grid<2x2, block, core_mapping = affine_map<(c0, c1) -> (c0, c1)>>
#layout0 = !tt.layout<strides<0x0x0x0>, shape<64x128>, pad_val<undef>, #grid0, memref<1x2x!tt.tile<32 x 32, f32>, #l1>>
#layout1 = !tt.layout<strides<0x0x0x0>, shape<64x128>, pad_val<undef>, #grid0, memref<1x2x!tt.tile<32 x 32, f32>, #dram>>
%0 = tensor<64x128xf32, #layout0>
%1 = tensor<64x128xf32, #layout1>
%2 = tensor<64x128xf32, #layout0>
%6 = "tt.dispatch"(%0, %1, %2) <{#grid0, indexing_maps = [#map0, #map0, #map0], iterator_types = ["parallel", "parallel"], buffer_accesses = [#b0, #b0, #b0]}> ({
^bb0(%arg0: memref<1x2x!tt.tile<32 x 32, f32>, #dram>, %arg1: memref<1x2x!tt.tile<32 x 32, f32>, #l1>, %arg2: memref<1x2x!tt.tile<32 x 32, f32>, #l1>):
  %7 = memref.load %arg0 : memref<1x2x!tt.tile<32 x 32, f32>, #l1>
  %8 = tt.builtin "add"(%7, %arg1) : (memref<1x2x!tt.tile<32 x 32, f32>, #l1>, memref<1x2x!tt.tile<32 x 32, f32>, #l1>) -> memref<1x2x!tt.tile<32 x 32, f32>, #l1>
  "tt.yield"(%8) : (memref<1x2x!tt.tile<32 x 32, f32>, #l1>) -> ()
}) : tensor<64x128xf32, #layout0>

//
// Matmul
//
#map2 = affine_map<(c0, c1, k0) -> (c0, k0)>
#map3 = affine_map<(c0, c1, k0) -> (k0, c1)>
#map4 = affine_map<(c0, c1, k0) -> (c0, c1)>
#grid0 = !tt.grid<2x2, block, core_mapping = affine_map<(c0, c1) -> (c0, c1)>>
#layout0 = !tt.layout<strides<0x0x0x0>, shape<64x128>, pad_val<undef>, #grid0, memref<1x2x!tt.tile<32 x 32, f32>, #l1>>
#layout1 = !tt.layout<strides<0x0x0x0>, shape<128x64>, pad_val<undef>, #grid0, memref<2x1x!tt.tile<32 x 32, f32>, #l1>>
#layout2 = !tt.layout<strides<0x0x0x0>, shape<64x64>, pad_val<undef>, #grid0, memref<1x1x!tt.tile<32 x 32, f32>, #l1>>
%0 = tensor<64x128xf32, #layout0>
%1 = tensor<128x64xf32, #layout1>
%2 = tensor<64x64xf32, #layout2>

%6 = "tt.dispatch"(%0, %1, %2) <{#grid0, indexing_maps = [#map2, #map3, #map4], iterator_types = ["parallel", "parallel", "systolic"]}> ({
^bb0(%arg0: memref<1x2x!tt.tile<32 x 32, f32>, #l1>, %arg1: memref<2x1x!tt.tile<32 x 32, f32>, #l1>, %arg2: memref<1x1x!tt.tile<32 x 32, f32>, #l1>):
  %8 = tt.builtin "matmul" ins(%arg0, %arg1) intermediates() outs(%arg2): (tensor<2x4x!tt.tile<32 x 32, f32>, tensor<2x4x!tt.tile<32 x 32, f32>>) -> tensor<2x4x!tt.tile<32 x 32, f32>
  "tt.yield"(%8) : (tensor<2x4x!tt.tile<32 x 32, f32>>) -> ()
}) : tensor<64x64xf32, #layout2>

//
// Eltwise Transposed
//
#map0 = affine_map<(c0, c1) -> (c0, c1)>
#gridT = !tt.grid<4x1, block, core_mapping = affine_map<(c0, c1) -> (c1, c0)>>
#layoutT = !tt.layout<strides<0x0x0x0>, shape<64x128>, pad_val<undef>, #gridT, memref<1x2x!tt.tile<32 x 32, f32>, #l1>>
%0 = tensor<128x64xf32, #layoutT>
%1 = tensor<128x64xf32, #layoutT>
%2 = tensor<128x64xf32, #layoutT>

%6 = "tt.dispatch"(%0, %1, %2) <{#grid0, indexing_maps = [#map0, #map0, #map0], iterator_types = ["parallel", "parallel"], buffer_accesses = [#b0, #b0, #b0]}> ({
^bb0(%arg0: memref<1x2x!tt.tile<32 x 32, f32>, #l1>, %arg1: memref<1x2x!tt.tile<32 x 32, f32>, #l1>, %arg2: memref<1x2x!tt.tile<32 x 32, f32>, #l1>):
  %8 = tt.builtin "add"(%arg0, %arg1) : (memref<1x2x!tt.tile<32 x 32, f32>, #l1>, memref<1x2x!tt.tile<32 x 32, f32>, #l1>) -> memref<1x2x!tt.tile<32 x 32, f32>, #l1>
  "tt.yield"(%8) : (memref<1x2x!tt.tile<32 x 32, f32>, #l1>) -> ()
}) : tensor<64x128xf32, #layout0>

//
// Conv
//
#map0 = affine_map<(c0, c1) -> (c0, c1)>
#grid0 = !tt.grid<32x1, height, core_mapping = affine_map<(c0, c1) -> (c0, c1)>>
#grid1 = !tt.grid<1x1, height, core_mapping = affine_map<(c0, c1) -> (c0, c1)>>
#layout0 = !tt.layout<strides<0x0x0x0>, shape<1x0x12544x64>, pad_val<undef>, #grid0, memref<392x64xf16b, #l1>>
#layout1 = !tt.layout<strides<(28*d2*d1*d0)x0x0x0>, shape<1x0x13024x64>, pad_val<undef>, #grid0, memref<407x64xf16b, #l1>>
#layout2 = !tt.layout<strides<0x0x0x0>, shape<1x0x576x128>, pad_val<undef>, #grid1, memref<18x2x!tt.tile<32 x 32, f32>, #l1>>
#layout3 = !tt.layout<strides<0x0x0x0>, shape<1x0x3136x128>, pad_val<undef>, #grid0, memref<98x128xf16b, #l1>>
%0 = tensor<1x112x112x64xf32, #layout0> // nhwc
%weights = tensor<3x3x64x128xf32, #layout2>
%out = tensor<1x56x56x128xf32, #layout3> // nhwc
%5 = tensor.pad %0 low[0, 1, 1, 0] high[0, 1, 1, 0] : tensor<1x112x112x64xf32, #layout0> to tensor<1x114x114x64xf32, #layout1>
%6 = "tt.dispatch"(%5, %weights, %out) <{#grid0, indexing_maps = [#map2, #map3, #map4], iterator_types = ["parallel", "parallel", "systolic"]}> ({
^bb0(%arg2: memref<407x64xf16b, #l1>, %arg3: memref<18x2x!tt.tile<32 x 32, f32>, #l1>, %arg4: memref<98x128xf16b, #l1>>):
  %9 = tt.builtin "conv2d_matmul" ins(%arg2, %arg3) intermediates() outs(%arg4): (tensor<2x4x!tt.tile<32 x 32, f32>, tensor<2x4x!tt.tile<32 x 32, f32>>) -> tensor<2x4x!tt.tile<32 x 32, f32>
  "tt.yield"(%9) : (memref<98x128xf16b, #l1>) -> ()
}) : tensor<1x56x56x128xf32, #layout3>

//
// WIP, linalg -> tensix
//
%6 = "tt.dispatch"(%2, %4, %5) <{indexing_maps = [#map0, #map0, #map0], iterator_types = ["parallel", "parallel"]}> ({
^bb0(%arg2: tensor<2x4x!tt.tile<32 x 32, f32>>, %arg3: tensor<2x4x!tt.tile<32 x 32, f32>>, %arg4: tensor<2x4x!tt.tile<32 x 32, f32>>):
  %8 = linalg.generic {indexing_maps = [#map0, #map0, #map0], iterator_types = ["parallel", "parallel"]} ins(%arg2, %arg3 : tensor<2x4x!tt.tile<32 x 32, f32>>, tensor<2x4x!tt.tile<32 x 32, f32>>) outs(%arg4 
: tensor<2x4x!tt.tile<32 x 32, f32>>) {
  ^bb0(%in: !tt.tile<32 x 32, f32>, %in_0: !tt.tile<32 x 32, f32>, %out: !tt.tile<32 x 32, f32>):
    %9 = tensix.unpack %out : <32 x 32, f32> -> f32
    %10 = tensix.unpack %in_0 : <32 x 32, f32> -> f32
    %11 = tensix.unpack %in : <32 x 32, f32> -> f32
    %12 = arith.mulf %11, %10 : f32
    %13 = tensix.pack %12 : f32 -> <32 x 32, f32>
    linalg.yield %13 : !tt.tile<32 x 32, f32>
  } -> tensor<2x4x!tt.tile<32 x 32, f32>>
  "tt.yield"(%8) : (tensor<2x4x!tt.tile<32 x 32, f32>>) -> ()
}) : (tensor<1x1x2x4x!tt.tile<32 x 32, f32>>, tensor<1x1x2x4x!tt.tile<32 x 32, f32>>, tensor<1x1x2x4x!tt.tile<32 x 32, f32>>) -> tensor<1x1x2x4x!tt.tile<32 x 32, f32>>
