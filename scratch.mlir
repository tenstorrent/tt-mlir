#map0 = affine_map<(c0, c1) -> (c0, c1)>

#map1 = affine_map<(d0, d1, c0, c1) -> (c1, c0)>

#map2 = affine_map<(c0, c1, k0) -> (c0, k0)>
#map3 = affine_map<(c0, c1, k0) -> (k0, c1)>
#map4 = affine_map<(c0, c1, k0) -> (c0, c1)>

#map_explicit = affine_map<(c0, c1) -> ()>





iterator_types = ["parallel", "systolic", "multicast", "explicit"]

%2 = tt.tensor_tilize %arg0 into %1 : tensor<64x128xf32> -> tensor<1x1x2x4x!tt.tile<32 x 32, f32>>
%4 = tt.tensor_tilize %arg1 into %3 : tensor<64x128xf32> -> tensor<1x1x2x4x!tt.tile<32 x 32, f32>>
%5 = tensor.empty() : tensor<1x1x2x4x!tt.tile<32 x 32, f32>>

%2_rm = !tt.tensor<shape<64x128>, strides<128x1>, pad_val<undef>, grid<1x1>, block<64x128xf32>, memory<"dram">>

%2 = !tt.tensor<shape<64x128>, strides<128x1>, pad_val<undef>, grid<[1x1],[1x1]>, block<2x4x!tt.tile<32 x 32, f32>>, memory<"l1">>


%2 = !tt.tensor<shape<16x4x64x128>, strides<0x0x0x0>, pad_val<undef>, grid<1x2>, slice<16x4x2x2x!tt.tile<32 x 32, f32>>, memory<"l1">>

%2 = !tt.tensor<shape<16x4x64x128>, strides<0x0x0x0>, pad_val<undef>, block<2x1x1x2, 8x4x2x2x!tt.tile<32 x 32, f32>>, memory<"l1">>


%2 = !tt.tensor<shape<16x4x128>, strides<0x0x0x0>, pad_val<undef>, block<2x2, 2x2x!tt.tile<32 x 32, f32>>, memory<"l1">>

%2 = !tt.tensor<shape<16x16x4x128>, strides<0x0x0x0>, pad_val<undef>, block<2x2x2, 8x2x2x!tt.tile<32 x 32, f32>>, memory<"l1">>





#map2 = affine_map<(d0, d1, d2) -> (d0, d1, d2)>

// strides = affine_map<(d0, d1, d2, d3) -> (512*d0 + 128*(d1 + d2) + d3)>
// slice = affine_map<(d0, d1, d2, d3) -> (d0, d1 + d2, d3)
%2 = !tt.tensor<shape<16x16x4x128>, strides<0x0x0x0>, pad_val<undef>, slice<2x0x1x2>, grid<1x2>, memref<8x1x2x2x!tt.tile<32 x 32, f32>, #l1_memspace>>
%2 = !tt.tensor<shape<16x16x4x128>, strides<0x0x0x0>, pad_val<undef>, slice<2x0x2x2>, memref<8x1x1x2x!tt.tile<32 x 32, f32>, #l1_memspace>>

// Layout flavors
%post = "tt.layout" %2_rm into %a : !tt.tensor<shape<64x128>, strides<128x1>, pad_val<undef>, grid<1x1>, block<64x128xf32>> -> !tt.tensor<shape<64x128>, strides<128x1>, pad_val<undef>, grid<1x1>, block<64x128xf32>>
%post = "tt.layout" %2_rm into %a <reinterpret> : !tt.tensor<shape<64x128>, strides<128x1>, pad_val<undef>, grid<1x1>, block<64x128xf32>> -> !tt.tensor<shape<64x128>, strides<128x1>, pad_val<undef>, grid<1x1>, block<64x128xf32>>






#b0 = buffer_access<alias, order = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, buffer<shape<...>, memory<"l1">>
#b1 = buffer_access<stream, order = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, buffer<shape<2x2x!tt.tile<32 x 32, f32>>, memory<"l1">>>
#b2 = buffer_access<alias, order = affine_map<(d0) -> (0)>, buffer<shape<1x!tt.tile<32 x 32, f32>>, memory<"l1">>>

// Eltwise
%6 = "tt.dispatch"(%2, %4, %5) <{grid<1x1>, indexing_maps = [#map0, #map0, #map0], iterator_types = ["parallel", "parallel"], buffer_accesses = [#b0, #b0, #b0]}> ({
^bb0(%arg2: tensor<2x4x!tt.tile<32 x 32, f32>>, %arg3: tensor<2x4x!tt.tile<32 x 32, f32>>, %arg4: tensor<2x4x!tt.tile<32 x 32, f32>>):
  %7 = memref.load %2[i, j] : memref<8x1x2x2x!tt.tile<32 x 32, f32>>
  %8 = tt.builtin "add"(%7, %arg3) : (tensor<2x4x!tt.tile<32 x 32, f32>, tensor<2x4x!tt.tile<32 x 32, f32>>) -> tensor<2x4x!tt.tile<32 x 32, f32>
  "tt.yield"(%8) : (tensor<2x4x!tt.tile<32 x 32, f32>>) -> ()
}) : (tensor<1x1x2x4x!tt.tile<32 x 32, f32>>, tensor<1x1x2x4x!tt.tile<32 x 32, f32>>, tensor<1x1x2x4x!tt.tile<32 x 32, f32>>) -> tensor<1x1x2x4x!tt.tile<32 x 32, f32>>









// Eltwise Transposed
%1 = !tt.tensor<shape<64x128>, strides<128x1>, pad_val<undef>, grid<rx1x4>, block<2x1x!tt.tile<32 x 32, f32>>>
%2 = !tt.tensor<shape<128x64>, strides<64x1>, pad_val<undef>, grid<cx4x1>, block<1x2x!tt.tile<32 x 32, f32>>>
%6 = "tt.dispatch"(%1, %2, %5) <{grid<rx1x4>, indexing_maps = [#map0, #map1, #map0], iterator_types = ["parallel", "parallel"]}> ({
^bb0(%arg2: tensor<2x4x!tt.tile<32 x 32, f32>>, %arg3: tensor<2x4x!tt.tile<32 x 32, f32>>, %arg4: tensor<2x4x!tt.tile<32 x 32, f32>>):
  %8 = tt.builtin "add" ins(%arg2, %arg3) outs(%arg4) : (tensor<2x4x!tt.tile<32 x 32, f32>, tensor<2x4x!tt.tile<32 x 32, f32>>) -> tensor<2x4x!tt.tile<32 x 32, f32>
  "tt.yield"(%8) : (tensor<2x4x!tt.tile<32 x 32, f32>>) -> ()
}) : (tensor<1x1x2x4x!tt.tile<32 x 32, f32>>, tensor<1x1x2x4x!tt.tile<32 x 32, f32>>, tensor<1x1x2x4x!tt.tile<32 x 32, f32>>) -> tensor<1x1x2x4x!tt.tile<32 x 32, f32>>

// Matmul
%6 = "tt.dispatch"(%2, %4, %5) <{grid<1x1>, indexing_maps = [#map2, #map3, #map4], iterator_types = ["parallel", "parallel", "systolic"]}> ({
^bb0(%arg2: tensor<2x4x!tt.tile<32 x 32, f32>>, %arg3: tensor<4x2x!tt.tile<32 x 32, f32>>, %arg4: tensor<2x2x!tt.tile<32 x 32, f32>>):
  %8 = tt.builtin "matmul" ins(%arg2, %arg3) intermediates() outs(%arg4): (tensor<2x4x!tt.tile<32 x 32, f32>, tensor<2x4x!tt.tile<32 x 32, f32>>) -> tensor<2x4x!tt.tile<32 x 32, f32>
  "tt.yield"(%8) : (tensor<2x4x!tt.tile<32 x 32, f32>>) -> ()
}) : (tensor<1x1x2x4x!tt.tile<32 x 32, f32>>, tensor<1x1x2x4x!tt.tile<32 x 32, f32>>, tensor<1x1x2x4x!tt.tile<32 x 32, f32>>) -> tensor<1x1x2x4x!tt.tile<32 x 32, f32>>

// Conv
%6 = "tt.dispatch"(%2, %4, %5) <{grid<1x1>, indexing_maps = [#map_explicit, #map_explicit], iterator_types = ["explicit", "explicit"]}> ({
^bb0(%arg2: tensor<2x4x!tt.tile<32 x 32, f32>>, %arg3: tensor<4x2x!tt.tile<32 x 32, f32>>, %arg4: tensor<2x2x!tt.tile<32 x 32, f32>>):
  %8 = tt.builtin "conv2d_gather" ins(%arg2) outs(%arg4): (tensor<2x4x!tt.tile<32 x 32, f32>, tensor<2x4x!tt.tile<32 x 32, f32>>) -> tensor<2x4x!tt.tile<32 x 32, f32>
  "tt.yield"(%8) : (tensor<2x4x!tt.tile<32 x 32, f32>>) -> ()
}) : (tensor<1x1x2x4x!tt.tile<32 x 32, f32>>, tensor<1x1x2x4x!tt.tile<32 x 32, f32>>, tensor<1x1x2x4x!tt.tile<32 x 32, f32>>) -> tensor<1x1x2x4x!tt.tile<32 x 32, f32>>

%6 = "tt.dispatch"(%2, %4, %5) <{grid<1x1>, indexing_maps = [#map2, #map3, #map4], iterator_types = ["parallel", "parallel", "systolic"]}> ({
^bb0(%arg2: tensor<2x4x!tt.tile<32 x 32, f32>>, %arg3: tensor<4x2x!tt.tile<32 x 32, f32>>, %arg4: tensor<2x2x!tt.tile<32 x 32, f32>>):
  %9 = tt.builtin "conv2d_matmul" ins(%arg2, %arg3) intermediates() outs(%arg4): (tensor<2x4x!tt.tile<32 x 32, f32>, tensor<2x4x!tt.tile<32 x 32, f32>>) -> tensor<2x4x!tt.tile<32 x 32, f32>
  "tt.yield"(%9) : (tensor<2x4x!tt.tile<32 x 32, f32>>) -> ()
}) : (tensor<1x1x2x4x!tt.tile<32 x 32, f32>>, tensor<1x1x2x4x!tt.tile<32 x 32, f32>>, tensor<1x1x2x4x!tt.tile<32 x 32, f32>>) -> tensor<1x1x2x4x!tt.tile<32 x 32, f32>>













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





