// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// 2D im2col, single NHWC input, f32. Input <1x17x17x64>, output <256x256>,
// input grid <1x8x8x1>, output grid <8x8>, K=2. Row-major.
// 4 spatial-shift views (K*K=4) -> 4 flatten views (4D grid<8x8>) ->
// composite stack on logical dim 1 (KKC) -> <256, 256> where 256 = K*K*C.

#l_in    = #ttcore.metal_layout<logical_shape = 1x17x17x64, dim_alignments = 1x32x32x32, collapsed_intervals = dense<> : tensor<0x2xi64>, l1, sharded>
#l_view  = #ttcore.metal_layout<logical_shape = 1x16x16x64, dim_alignments = 1x32x32x32, collapsed_intervals = dense<> : tensor<0x2xi64>, l1, sharded>
#l_flat  = #ttcore.metal_layout<logical_shape = 256x64,     dim_alignments = 1x32,       collapsed_intervals = dense<> : tensor<0x2xi64>, l1, sharded>
#l_out   = #ttcore.metal_layout<logical_shape = 256x256,    dim_alignments = 1x32,       collapsed_intervals = dense<> : tensor<0x2xi64>, l1, sharded>

// Step 1: 8D -> 8D spatial-shift. Input shard_H = shard_W = 4. H, W shifts
// wrap across grid via floordiv 4 / mod 4.
#sh_00 = affine_map<(d0, d1, d2, d3, d4, d5, d6, d7) -> (d0, (d1 * 4 + d5)     floordiv 4, (d2 * 4 + d6)     floordiv 4, d3, d4, (d1 * 4 + d5)     mod 4, (d2 * 4 + d6)     mod 4, d7)>
#sh_01 = affine_map<(d0, d1, d2, d3, d4, d5, d6, d7) -> (d0, (d1 * 4 + d5)     floordiv 4, (d2 * 4 + d6 + 1) floordiv 4, d3, d4, (d1 * 4 + d5)     mod 4, (d2 * 4 + d6 + 1) mod 4, d7)>
#sh_10 = affine_map<(d0, d1, d2, d3, d4, d5, d6, d7) -> (d0, (d1 * 4 + d5 + 1) floordiv 4, (d2 * 4 + d6)     floordiv 4, d3, d4, (d1 * 4 + d5 + 1) mod 4, (d2 * 4 + d6)     mod 4, d7)>
#sh_11 = affine_map<(d0, d1, d2, d3, d4, d5, d6, d7) -> (d0, (d1 * 4 + d5 + 1) floordiv 4, (d2 * 4 + d6 + 1) floordiv 4, d3, d4, (d1 * 4 + d5 + 1) mod 4, (d2 * 4 + d6 + 1) mod 4, d7)>

// Step 2: 4D grid<8x8> -> 8D grid<1x8x8x1>.
#flatten_g8x8 = affine_map<(d0, d1, d2, d3) -> (0, ((d0 * 32 + d2) floordiv 16) floordiv 4, ((d0 * 32 + d2) mod 16) floordiv 4, 0, 0, ((d0 * 32 + d2) floordiv 16) mod 4, ((d0 * 32 + d2) mod 16) mod 4, d1 * 8 + d3)>

// VGM for virt grid [1, 8, 8, 1] placed on physical grid 8x8 (volume=64).
#vgm_fwd = affine_map<(d0, d1, d2, d3, d4, d5, d6, d7) -> ((d0 * 64 + d1 * 8 + d2 + d3) floordiv 8 mod 8, (d0 * 64 + d1 * 8 + d2 + d3) mod 8, d4, d5, d6, d7)>
#vgm_inv = affine_map<(d0, d1) -> (0, 0, (d0 * 8 + d1) floordiv 8 mod 8, (d0 * 8 + d1) mod 8, 0)>

func.func @composite_im2col_nhwc_grid8x8(%arg0: tensor<1x17x17x64xf32>) -> tensor<256x256xf32> {
  %d_init = d2m.empty() {
    virtualGridForwardMapping = #vgm_fwd,
    virtualGridInverseMapping = #vgm_inv
  } : tensor<1x8x8x1x1x4x4x64xf32, #l_in>
  %dev = d2m.to_layout %arg0, %d_init : tensor<1x17x17x64xf32> into tensor<1x8x8x1x1x4x4x64xf32, #l_in>
    -> tensor<1x8x8x1x1x4x4x64xf32, #l_in>

  %s00 = d2m.view_layout %dev remapping = #sh_00 : tensor<1x8x8x1x1x4x4x64xf32, #l_in> -> tensor<1x8x8x1x1x4x4x64xf32, #l_view>
  %s01 = d2m.view_layout %dev remapping = #sh_01 : tensor<1x8x8x1x1x4x4x64xf32, #l_in> -> tensor<1x8x8x1x1x4x4x64xf32, #l_view>
  %s10 = d2m.view_layout %dev remapping = #sh_10 : tensor<1x8x8x1x1x4x4x64xf32, #l_in> -> tensor<1x8x8x1x1x4x4x64xf32, #l_view>
  %s11 = d2m.view_layout %dev remapping = #sh_11 : tensor<1x8x8x1x1x4x4x64xf32, #l_in> -> tensor<1x8x8x1x1x4x4x64xf32, #l_view>

  %f00 = d2m.view_layout %s00 remapping = #flatten_g8x8 : tensor<1x8x8x1x1x4x4x64xf32, #l_view> -> tensor<8x8x32x8xf32, #l_flat>
  %f01 = d2m.view_layout %s01 remapping = #flatten_g8x8 : tensor<1x8x8x1x1x4x4x64xf32, #l_view> -> tensor<8x8x32x8xf32, #l_flat>
  %f10 = d2m.view_layout %s10 remapping = #flatten_g8x8 : tensor<1x8x8x1x1x4x4x64xf32, #l_view> -> tensor<8x8x32x8xf32, #l_flat>
  %f11 = d2m.view_layout %s11 remapping = #flatten_g8x8 : tensor<1x8x8x1x1x4x4x64xf32, #l_view> -> tensor<8x8x32x8xf32, #l_flat>

  %cv = "d2m.composite_view"(%f00, %f01, %f10, %f11)
        <{dim = 1 : si32}> :
        (tensor<8x8x32x8xf32, #l_flat>, tensor<8x8x32x8xf32, #l_flat>,
         tensor<8x8x32x8xf32, #l_flat>, tensor<8x8x32x8xf32, #l_flat>)
     -> tensor<8x8x32x32xf32, #l_out>

  %h_empty = d2m.empty() : tensor<256x256xf32>
  %result = d2m.to_layout %cv, %h_empty : tensor<8x8x32x32xf32, #l_out> into tensor<256x256xf32>
    -> tensor<256x256xf32>
  return %result : tensor<256x256xf32>
}
