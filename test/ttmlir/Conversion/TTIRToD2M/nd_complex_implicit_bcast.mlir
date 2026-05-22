// RUN: ttmlir-opt --ttcore-register-device --ttir-to-d2m="ttnn-mode=true" --mlir-print-local-scope -o %t %s
// RUN: FileCheck %s --input-file=%t

// Verify that TTIRToD2M handles "complex" implicit broadcasts whose physical
// shard dim is neither equal to the output's nor 1: an input with N x C > 1
// outer logical dims gets one of them broadcast, and the TTNN layout collapse
// folds the broadcast dim into the same physical row as the non-broadcast
// outer dim, producing a shard whose row dim is a *divisor* of the output's
// (not 1, not equal).
//
// The fix materializes the cross-tile broadcast via a `d2m.view_layout` that
// uses `floordiv`/`mod` in its `remapping`, leaving the downstream d2m.generic
// indexing maps as simple identity/constant-0 broadcast-projected
// permutations (which is required by the d2m.generic verifier and its
// `inverseAndBroadcastProjectedPermutation`-based shape analyses).
//
// These cases are derived from the DeepSeek-V3 subgraphs in issue #8541
// (5D RoPE-style chains).

#dram = #ttnn.buffer_type<dram>

// 5D operand layouts that all collapse to a 2D physical memref:
//   32x16x1x32x1  -> physical (16384, 1) -> tiles (512, 1)  "_full_d2"
//    1x16x1x32x1  -> physical (   512, 1) -> tiles ( 16, 1)  "_bcast_d2"  (broadcast on d0, factor 32)
//   32x16x8x32x1  -> physical (131072, 1) -> tiles (4096, 1) "_full_d8"
//    1x16x1x32x1  -> physical (   512, 1) -> tiles ( 16, 1)  "_bcast_d8"  (broadcast on d0 and d2, intermediate factor 8)
//   32x16x16x32x1 -> physical (262144, 1) -> tiles (8192, 1) "_full_d16"
//    1x16x1x32x1  -> physical (   512, 1) -> tiles ( 16, 1)  "_bcast_d16" (broadcast on d0 and d2, intermediate factor 16)
#layout_full_d2  = #ttnn.ttnn_layout<(d0, d1, d2, d3, d4) -> (d0 * 512 + d1 * 32 + d2 * 32 + d3, d4),  <1x1>, memref<512x1x!ttcore.tile<32x32, f32>,  #dram>, <interleaved>>
#layout_bcast    = #ttnn.ttnn_layout<(d0, d1, d2, d3, d4) -> (d0 * 512 + d1 * 32 + d2 * 32 + d3, d4),  <1x1>, memref<16x1x!ttcore.tile<32x32, f32>,   #dram>, <interleaved>>
#layout_full_d8  = #ttnn.ttnn_layout<(d0, d1, d2, d3, d4) -> (d0 * 4096 + d1 * 256 + d2 * 32 + d3, d4), <1x1>, memref<4096x1x!ttcore.tile<32x32, f32>, #dram>, <interleaved>>
#layout_full_d16 = #ttnn.ttnn_layout<(d0, d1, d2, d3, d4) -> (d0 * 8192 + d1 * 512 + d2 * 32 + d3, d4), <1x1>, memref<8192x1x!ttcore.tile<32x32, f32>, #dram>, <interleaved>>

module {

  // Outer broadcast on d0 only (d2=1 in output). Per-tile broadcast factor in
  // the row physical dim is 32 (= output 512 tiles / input 16 tiles), and no
  // intermediate non-1 dim sits between the broadcast d0 and the non-broadcast
  // d1, so the row remap is simply `d2 mod 16`.
  // CHECK-LABEL: func.func @nd5_outer_bcast_d2
  func.func @nd5_outer_bcast_d2(
      %arg0: tensor<32x16x1x32x1xf32, #layout_full_d2>,
      %arg1: tensor<1x16x1x32x1xf32,  #layout_bcast>) -> tensor<32x16x1x32x1xf32, #layout_full_d2> {
    // CHECK: d2m.view_layout {{.*}} remapping = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2 mod 16, d3)>
    // CHECK: d2m.generic
    // CHECK: d2m.tile_add
    %0 = "ttir.add"(%arg0, %arg1) : (tensor<32x16x1x32x1xf32, #layout_full_d2>, tensor<1x16x1x32x1xf32, #layout_bcast>) -> tensor<32x16x1x32x1xf32, #layout_full_d2>
    return %0 : tensor<32x16x1x32x1xf32, #layout_full_d2>
  }

  // Outer broadcast on d0 with a non-1 d2 between the broadcast and the
  // non-broadcast d1. The remap must use a non-trivial floordiv: the
  // intermediate dim d2 (output size 8) contributes a stride of 8 tiles
  // between successive d1 values in the output, so the row remap becomes
  // `(d2 floordiv 8) mod 16`.
  // CHECK-LABEL: func.func @nd5_outer_bcast_d8
  func.func @nd5_outer_bcast_d8(
      %arg0: tensor<32x16x8x32x1xf32, #layout_full_d8>,
      %arg1: tensor<1x16x1x32x1xf32,  #layout_bcast>) -> tensor<32x16x8x32x1xf32, #layout_full_d8> {
    // CHECK: d2m.view_layout {{.*}} remapping = affine_map<(d0, d1, d2, d3) -> (d0, d1, (d2 floordiv 8) mod 16, d3)>
    // CHECK: d2m.generic
    // CHECK: d2m.tile_add
    %0 = "ttir.add"(%arg0, %arg1) : (tensor<32x16x8x32x1xf32, #layout_full_d8>, tensor<1x16x1x32x1xf32, #layout_bcast>) -> tensor<32x16x8x32x1xf32, #layout_full_d8>
    return %0 : tensor<32x16x8x32x1xf32, #layout_full_d8>
  }

  // Same pattern as above but with d2 size = 16 in the output. The row remap
  // floordiv divisor equals the intermediate dim size (16), and the mod still
  // equals the input's d1 size (16).
  // CHECK-LABEL: func.func @nd5_outer_bcast_d16
  func.func @nd5_outer_bcast_d16(
      %arg0: tensor<32x16x16x32x1xf32, #layout_full_d16>,
      %arg1: tensor<1x16x1x32x1xf32,   #layout_bcast>) -> tensor<32x16x16x32x1xf32, #layout_full_d16> {
    // CHECK: d2m.view_layout {{.*}} remapping = affine_map<(d0, d1, d2, d3) -> (d0, d1, (d2 floordiv 16) mod 16, d3)>
    // CHECK: d2m.generic
    // CHECK: d2m.tile_add
    %0 = "ttir.add"(%arg0, %arg1) : (tensor<32x16x16x32x1xf32, #layout_full_d16>, tensor<1x16x1x32x1xf32, #layout_bcast>) -> tensor<32x16x16x32x1xf32, #layout_full_d16>
    return %0 : tensor<32x16x16x32x1xf32, #layout_full_d16>
  }

  // RoPE-style 4-input chain: (q * cos) + (q_rot * sin), broadcast on the
  // outer batch dim of cos / sin. Each multiply needs its own view_layout for
  // the broadcasted operand; the final add has shape-matched operands and only
  // needs identity indexing maps.
  // CHECK-LABEL: func.func @nd5_rope_chain_d8
  func.func @nd5_rope_chain_d8(
      %q     : tensor<32x16x8x32x1xf32, #layout_full_d8>,
      %cos   : tensor<1x16x1x32x1xf32,  #layout_bcast>,
      %q_rot : tensor<32x16x8x32x1xf32, #layout_full_d8>,
      %sin   : tensor<1x16x1x32x1xf32,  #layout_bcast>) -> tensor<32x16x8x32x1xf32, #layout_full_d8> {
    // The two multiplies each get a view_layout for their broadcasted
    // operand; the trailing add reuses their already-broadcast results so it
    // only needs identity indexing maps. Sequenced as view, generic, view,
    // generic, generic.
    // CHECK: d2m.view_layout {{.*}} mod 16
    // CHECK: d2m.generic
    // CHECK: d2m.view_layout {{.*}} mod 16
    // CHECK-COUNT-2: d2m.generic
    %0 = "ttir.multiply"(%q,     %cos) : (tensor<32x16x8x32x1xf32, #layout_full_d8>, tensor<1x16x1x32x1xf32, #layout_bcast>) -> tensor<32x16x8x32x1xf32, #layout_full_d8>
    %1 = "ttir.multiply"(%q_rot, %sin) : (tensor<32x16x8x32x1xf32, #layout_full_d8>, tensor<1x16x1x32x1xf32, #layout_bcast>) -> tensor<32x16x8x32x1xf32, #layout_full_d8>
    %2 = "ttir.add"(%0, %1)            : (tensor<32x16x8x32x1xf32, #layout_full_d8>, tensor<32x16x8x32x1xf32, #layout_full_d8>) -> tensor<32x16x8x32x1xf32, #layout_full_d8>
    return %2 : tensor<32x16x8x32x1xf32, #layout_full_d8>
  }
}
