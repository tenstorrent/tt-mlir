// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// 2D im2col, single NHWC input, rowmajor input <1x66x66x32>, output <4096x288>,
// input grid <1x64x1x1>, output grid <64x1>, K=3.
// 9 spatial-shift views (K*K=9) -> 9 flatten views (4D grid<64x1>) ->
// composite stack on logical dim 1 (KKC) -> <4096, 288> where 288 = K*K*C.

#l_in    = #ttcore.metal_layout<logical_shape = 1x66x66x32, dim_alignments = 1x64x32x32, collapsed_intervals = dense<> : tensor<0x2xi64>, l1, sharded>
#l_view  = #ttcore.metal_layout<logical_shape = 1x64x64x32, dim_alignments = 1x64x32x32, collapsed_intervals = dense<> : tensor<0x2xi64>, l1, sharded>
#l_flat  = #ttcore.metal_layout<logical_shape = 4096x32,    dim_alignments = 1x32,       collapsed_intervals = dense<> : tensor<0x2xi64>, l1, sharded>
#l_out   = #ttcore.metal_layout<logical_shape = 4096x288,   dim_alignments = 1x32,       collapsed_intervals = dense<> : tensor<0x2xi64>, l1, sharded>

// Step 1: 8D -> 8D spatial-shift. Input shard_H = 2, shard_W = 96 (W not
// sharded, W_g=1). H shift wraps across grid_h via floordiv 2 / mod 2.
#sh_00 = affine_map<(d0, d1, d2, d3, d4, d5, d6, d7) -> (d0, (d1 * 2 + d5    ) floordiv 2, d2, d3, d4, (d1 * 2 + d5    ) mod 2, d6    , d7)>
#sh_01 = affine_map<(d0, d1, d2, d3, d4, d5, d6, d7) -> (d0, (d1 * 2 + d5    ) floordiv 2, d2, d3, d4, (d1 * 2 + d5    ) mod 2, d6 + 1, d7)>
#sh_02 = affine_map<(d0, d1, d2, d3, d4, d5, d6, d7) -> (d0, (d1 * 2 + d5    ) floordiv 2, d2, d3, d4, (d1 * 2 + d5    ) mod 2, d6 + 2, d7)>
#sh_10 = affine_map<(d0, d1, d2, d3, d4, d5, d6, d7) -> (d0, (d1 * 2 + d5 + 1) floordiv 2, d2, d3, d4, (d1 * 2 + d5 + 1) mod 2, d6    , d7)>
#sh_11 = affine_map<(d0, d1, d2, d3, d4, d5, d6, d7) -> (d0, (d1 * 2 + d5 + 1) floordiv 2, d2, d3, d4, (d1 * 2 + d5 + 1) mod 2, d6 + 1, d7)>
#sh_12 = affine_map<(d0, d1, d2, d3, d4, d5, d6, d7) -> (d0, (d1 * 2 + d5 + 1) floordiv 2, d2, d3, d4, (d1 * 2 + d5 + 1) mod 2, d6 + 2, d7)>
#sh_20 = affine_map<(d0, d1, d2, d3, d4, d5, d6, d7) -> (d0, (d1 * 2 + d5 + 2) floordiv 2, d2, d3, d4, (d1 * 2 + d5 + 2) mod 2, d6    , d7)>
#sh_21 = affine_map<(d0, d1, d2, d3, d4, d5, d6, d7) -> (d0, (d1 * 2 + d5 + 2) floordiv 2, d2, d3, d4, (d1 * 2 + d5 + 2) mod 2, d6 + 1, d7)>
#sh_22 = affine_map<(d0, d1, d2, d3, d4, d5, d6, d7) -> (d0, (d1 * 2 + d5 + 2) floordiv 2, d2, d3, d4, (d1 * 2 + d5 + 2) mod 2, d6 + 2, d7)>

// Step 2: 4D grid<64x1> -> 8D grid<1x64x1x1>.
#flatten_g64x1 = affine_map<(d0, d1, d2, d3) -> (0, ((d0 * 64 + d2) floordiv 64) floordiv 2, 0, 0, 0, ((d0 * 64 + d2) floordiv 64) mod 2, (d0 * 64 + d2) mod 64, d3)>

#vgm_fwd = affine_map<(d0, d1, d2, d3, d4, d5, d6, d7) -> ((d0 * 64 + d1 + d2 + d3) floordiv 8 mod 8, (d0 * 64 + d1 + d2 + d3) mod 8, d4, d5, d6, d7)>
#vgm_inv = affine_map<(d0, d1) -> (0, 0, (d0 * 8 + d1) mod 64, 0, 0)>

func.func @composite_im2col_nhwc_grid64x1(%arg0: tensor<1x66x66x32xf32>) -> tensor<4096x288xf32> {
  %d_init = d2m.empty() {
    virtualGridForwardMapping = #vgm_fwd,
    virtualGridInverseMapping = #vgm_inv
  } : tensor<1x64x1x1x1x2x96x32xf32, #l_in>
  %dev = d2m.to_layout %arg0, %d_init : tensor<1x66x66x32xf32> into tensor<1x64x1x1x1x2x96x32xf32, #l_in>
    -> tensor<1x64x1x1x1x2x96x32xf32, #l_in>

  %s00 = d2m.view_layout %dev remapping = #sh_00 : tensor<1x64x1x1x1x2x96x32xf32, #l_in> -> tensor<1x64x1x1x1x2x96x32xf32, #l_view>
  %s01 = d2m.view_layout %dev remapping = #sh_01 : tensor<1x64x1x1x1x2x96x32xf32, #l_in> -> tensor<1x64x1x1x1x2x96x32xf32, #l_view>
  %s02 = d2m.view_layout %dev remapping = #sh_02 : tensor<1x64x1x1x1x2x96x32xf32, #l_in> -> tensor<1x64x1x1x1x2x96x32xf32, #l_view>
  %s10 = d2m.view_layout %dev remapping = #sh_10 : tensor<1x64x1x1x1x2x96x32xf32, #l_in> -> tensor<1x64x1x1x1x2x96x32xf32, #l_view>
  %s11 = d2m.view_layout %dev remapping = #sh_11 : tensor<1x64x1x1x1x2x96x32xf32, #l_in> -> tensor<1x64x1x1x1x2x96x32xf32, #l_view>
  %s12 = d2m.view_layout %dev remapping = #sh_12 : tensor<1x64x1x1x1x2x96x32xf32, #l_in> -> tensor<1x64x1x1x1x2x96x32xf32, #l_view>
  %s20 = d2m.view_layout %dev remapping = #sh_20 : tensor<1x64x1x1x1x2x96x32xf32, #l_in> -> tensor<1x64x1x1x1x2x96x32xf32, #l_view>
  %s21 = d2m.view_layout %dev remapping = #sh_21 : tensor<1x64x1x1x1x2x96x32xf32, #l_in> -> tensor<1x64x1x1x1x2x96x32xf32, #l_view>
  %s22 = d2m.view_layout %dev remapping = #sh_22 : tensor<1x64x1x1x1x2x96x32xf32, #l_in> -> tensor<1x64x1x1x1x2x96x32xf32, #l_view>

  %f00 = d2m.view_layout %s00 remapping = #flatten_g64x1 : tensor<1x64x1x1x1x2x96x32xf32, #l_view> -> tensor<64x1x64x32xf32, #l_flat>
  %f01 = d2m.view_layout %s01 remapping = #flatten_g64x1 : tensor<1x64x1x1x1x2x96x32xf32, #l_view> -> tensor<64x1x64x32xf32, #l_flat>
  %f02 = d2m.view_layout %s02 remapping = #flatten_g64x1 : tensor<1x64x1x1x1x2x96x32xf32, #l_view> -> tensor<64x1x64x32xf32, #l_flat>
  %f10 = d2m.view_layout %s10 remapping = #flatten_g64x1 : tensor<1x64x1x1x1x2x96x32xf32, #l_view> -> tensor<64x1x64x32xf32, #l_flat>
  %f11 = d2m.view_layout %s11 remapping = #flatten_g64x1 : tensor<1x64x1x1x1x2x96x32xf32, #l_view> -> tensor<64x1x64x32xf32, #l_flat>
  %f12 = d2m.view_layout %s12 remapping = #flatten_g64x1 : tensor<1x64x1x1x1x2x96x32xf32, #l_view> -> tensor<64x1x64x32xf32, #l_flat>
  %f20 = d2m.view_layout %s20 remapping = #flatten_g64x1 : tensor<1x64x1x1x1x2x96x32xf32, #l_view> -> tensor<64x1x64x32xf32, #l_flat>
  %f21 = d2m.view_layout %s21 remapping = #flatten_g64x1 : tensor<1x64x1x1x1x2x96x32xf32, #l_view> -> tensor<64x1x64x32xf32, #l_flat>
  %f22 = d2m.view_layout %s22 remapping = #flatten_g64x1 : tensor<1x64x1x1x1x2x96x32xf32, #l_view> -> tensor<64x1x64x32xf32, #l_flat>

  %cv = "d2m.composite_view"(%f00, %f01, %f02, %f10, %f11, %f12, %f20, %f21, %f22)
        <{dim = 1 : si32}> :
        (tensor<64x1x64x32xf32, #l_flat>, tensor<64x1x64x32xf32, #l_flat>, tensor<64x1x64x32xf32, #l_flat>,
         tensor<64x1x64x32xf32, #l_flat>, tensor<64x1x64x32xf32, #l_flat>, tensor<64x1x64x32xf32, #l_flat>,
         tensor<64x1x64x32xf32, #l_flat>, tensor<64x1x64x32xf32, #l_flat>, tensor<64x1x64x32xf32, #l_flat>)
     -> tensor<64x1x64x288xf32, #l_out>

  %h_empty = d2m.empty() : tensor<4096x288xf32>
  %result = d2m.to_layout %cv, %h_empty : tensor<64x1x64x288xf32, #l_out> into tensor<4096x288xf32>
    -> tensor<4096x288xf32>
  return %result : tensor<4096x288xf32>
}
