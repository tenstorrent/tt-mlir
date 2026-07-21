// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// 2D im2col, single NHWC input, f32. Input <1x9x9x32>, output <64x128>,
// grid <1x1>, K=2. Single-core, row-major.
// 4 spatial-shift views (K*K=4) -> 4 flatten views -> composite stack on
// logical dim 1 (C) -> <64, 128> where 128 = K*K*C.

#l_in   = #ttcore.metal_layout<logical_shape = 1x9x9x32, dim_alignments = 1x32x32x32, collapsed_intervals = dense<> : tensor<0x2xi64>, l1, sharded>
#l_view = #ttcore.metal_layout<logical_shape = 1x8x8x32, dim_alignments = 1x32x32x32, collapsed_intervals = dense<> : tensor<0x2xi64>, l1, sharded>
#l_flat = #ttcore.metal_layout<logical_shape = 64x32,    dim_alignments = 32x32,      collapsed_intervals = dense<> : tensor<0x2xi64>, l1, sharded>
#l_out  = #ttcore.metal_layout<logical_shape = 64x128,   dim_alignments = 32x32,      collapsed_intervals = dense<> : tensor<0x2xi64>, l1, sharded>

// Step 1: 8D -> 8D spatial-shift. Identity on grid (d0..d3), N (d4), C (d7).
#sh_00 = affine_map<(d0, d1, d2, d3, d4, d5, d6, d7) -> (d0, d1, d2, d3, d4, d5,     d6,     d7)>
#sh_01 = affine_map<(d0, d1, d2, d3, d4, d5, d6, d7) -> (d0, d1, d2, d3, d4, d5,     d6 + 1, d7)>
#sh_10 = affine_map<(d0, d1, d2, d3, d4, d5, d6, d7) -> (d0, d1, d2, d3, d4, d5 + 1, d6,     d7)>
#sh_11 = affine_map<(d0, d1, d2, d3, d4, d5, d6, d7) -> (d0, d1, d2, d3, d4, d5 + 1, d6 + 1, d7)>

// Step 2: 4D -> 8D flatten. NHW reshape; identity on C.
#flatten = affine_map<(d0, d1, d2, d3) -> (0, 0, 0, 0, 0, d2 floordiv 8, d2 mod 8, d3)>

func.func @composite_im2col_nhwc(%arg0: tensor<1x9x9x32xf32>) -> tensor<64x128xf32> {
  %d_init = d2m.empty() : tensor<1x1x1x1x1x32x32x32xf32, #l_in>
  %dev = d2m.to_layout %arg0, %d_init : tensor<1x9x9x32xf32> into tensor<1x1x1x1x1x32x32x32xf32, #l_in>
    -> tensor<1x1x1x1x1x32x32x32xf32, #l_in>

  %s00 = d2m.view_layout %dev remapping = #sh_00 : tensor<1x1x1x1x1x32x32x32xf32, #l_in> -> tensor<1x1x1x1x1x32x32x32xf32, #l_view>
  %s01 = d2m.view_layout %dev remapping = #sh_01 : tensor<1x1x1x1x1x32x32x32xf32, #l_in> -> tensor<1x1x1x1x1x32x32x32xf32, #l_view>
  %s10 = d2m.view_layout %dev remapping = #sh_10 : tensor<1x1x1x1x1x32x32x32xf32, #l_in> -> tensor<1x1x1x1x1x32x32x32xf32, #l_view>
  %s11 = d2m.view_layout %dev remapping = #sh_11 : tensor<1x1x1x1x1x32x32x32xf32, #l_in> -> tensor<1x1x1x1x1x32x32x32xf32, #l_view>

  %f00 = d2m.view_layout %s00 remapping = #flatten : tensor<1x1x1x1x1x32x32x32xf32, #l_view> -> tensor<1x1x64x32xf32, #l_flat>
  %f01 = d2m.view_layout %s01 remapping = #flatten : tensor<1x1x1x1x1x32x32x32xf32, #l_view> -> tensor<1x1x64x32xf32, #l_flat>
  %f10 = d2m.view_layout %s10 remapping = #flatten : tensor<1x1x1x1x1x32x32x32xf32, #l_view> -> tensor<1x1x64x32xf32, #l_flat>
  %f11 = d2m.view_layout %s11 remapping = #flatten : tensor<1x1x1x1x1x32x32x32xf32, #l_view> -> tensor<1x1x64x32xf32, #l_flat>

  %cv = "d2m.composite_view"(%f00, %f01, %f10, %f11)
        <{dim = 1 : si32}> :
        (tensor<1x1x64x32xf32, #l_flat>, tensor<1x1x64x32xf32, #l_flat>,
         tensor<1x1x64x32xf32, #l_flat>, tensor<1x1x64x32xf32, #l_flat>)
     -> tensor<1x1x64x128xf32, #l_out>

  %h_empty = d2m.empty() : tensor<64x128xf32>
  %result = d2m.to_layout %cv, %h_empty : tensor<1x1x64x128xf32, #l_out> into tensor<64x128xf32>
    -> tensor<64x128xf32>
  return %result : tensor<64x128xf32>
}
