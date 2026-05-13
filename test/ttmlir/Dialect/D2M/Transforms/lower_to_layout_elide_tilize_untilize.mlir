// RUN: ttmlir-opt --ttcore-register-device --d2m-lower-to-layout -o %t %s
// RUN: FileCheck %s --input-file=%t

// When source is untiled L1 and target is tiled L1 with a complex layout
// change (differing dim_alignments / collapsed_intervals), the canonical plan
// emits:
//     Tilize ; Untilize ; Reshard_scalar ; Tilize_back
// The minimizer cancels the leading Tilize;Untilize pair, leaving just
//     Reshard_scalar ; Tilize_back
// which saves one tile_tilize_block and one tile_untilize_block op compared
// to the pre-refactor behavior.

#src = #ttcore.metal_layout<logical_shape = 32x5120, dim_alignments = 32x256, collapsed_intervals = dense<[[0, 1], [1, 2]]> : tensor<2x2xi64>, undef, l1, sharded>
#dst = #ttcore.metal_layout<logical_shape = 32x5120, dim_alignments = 32x32, collapsed_intervals = dense<> : tensor<0x2xi64>, undef, l1, sharded>

// CHECK-LABEL: func.func @elide_redundant_tilize_untilize
// The minimizer drops the tilize-then-untilize round trip: we expect exactly
// one tile_tilize_block (the final tilize_back) and zero tile_untilize_block.
// CHECK-COUNT-1: d2m.tile_tilize_block
// CHECK-NOT: d2m.tile_tilize_block
// CHECK-NOT: d2m.tile_untilize_block
func.func @elide_redundant_tilize_untilize() -> tensor<1x1x1x160x!ttcore.tile<32x32, f32>, #dst> {
  %src = d2m.empty() : tensor<1x40x32x128xf32, #src>
  %dst = d2m.empty() : tensor<1x1x1x160x!ttcore.tile<32x32, f32>, #dst>
  %r = d2m.to_layout %src, %dst : tensor<1x40x32x128xf32, #src> into tensor<1x1x1x160x!ttcore.tile<32x32, f32>, #dst> -> tensor<1x1x1x160x!ttcore.tile<32x32, f32>, #dst>
  return %r : tensor<1x1x1x160x!ttcore.tile<32x32, f32>, #dst>
}
