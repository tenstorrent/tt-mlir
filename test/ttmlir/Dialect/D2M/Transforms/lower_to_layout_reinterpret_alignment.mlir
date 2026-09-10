// RUN: ttmlir-opt --ttcore-register-device --d2m-lower-to-layout -o %t %s
// RUN: FileCheck %s --input-file=%t

// An L1->L1 layout change that only differs in dim_alignments (256x256 vs
// 32x32) pads to the exact same device shape, so the two layouts describe the
// same physical data placement. canUseReinterpretLayoutView treats this as a
// metadata-only change and reinterprets it in place instead of lowering it
// through a scalar untilize -> reshard -> tilize round trip. Here the
// reinterpreted value is consumed by a DRAM store, so the only emitted work is
// the L1->DRAM DMA: %arg0 (the 256x256-aligned input) feeds the store directly
// and no tile_tilize_block / tile_untilize_block is produced.

#l1_align_big = #ttcore.metal_layout<logical_shape = 256x256, dim_alignments = 256x256, collapsed_intervals = dense<[[0, 1], [1, 2]]> : tensor<2x2xi64>, l1, sharded>
#l1_align_tile = #ttcore.metal_layout<logical_shape = 256x256, dim_alignments = 32x32, collapsed_intervals = dense<[[0, 1], [1, 2]]> : tensor<2x2xi64>, l1, sharded>
#dram = #ttcore.metal_layout<logical_shape = 256x256, dim_alignments = 32x32, collapsed_intervals = dense<[[0, 1], [1, 2]]> : tensor<2x2xi64>, dram, sharded>

// CHECK-LABEL: func.func @reinterpret_alignment_only(
// CHECK-SAME: %arg0: tensor<1x1x8x8x!ttcore.tile<32x32, f32>, #[[BIG:.*]]>)
func.func @reinterpret_alignment_only(%arg0: tensor<1x1x8x8x!ttcore.tile<32x32, f32>, #l1_align_big>) -> tensor<1x1x8x8x!ttcore.tile<32x32, f32>, #dram> {
  // CHECK: %[[OUT:.*]] = d2m.empty() : tensor<1x1x8x8x!ttcore.tile<32x32, f32>, #[[DRAM:.*]]>
  // CHECK: d2m.generic
  // CHECK: ins(%arg0 : tensor<1x1x8x8x!ttcore.tile<32x32, f32>, #[[BIG]]>)
  // CHECK: outs(%[[OUT]] : tensor<1x1x8x8x!ttcore.tile<32x32, f32>, #[[DRAM]]>)
  // CHECK: d2m.remote_load
  // CHECK: d2m.remote_store
  // CHECK-NOT: d2m.tile_tilize_block
  // CHECK-NOT: d2m.tile_untilize_block
  // CHECK-NOT: d2m.generic
  %dst = d2m.empty() : tensor<1x1x8x8x!ttcore.tile<32x32, f32>, #l1_align_tile>
  %reint = d2m.to_layout %arg0, %dst : tensor<1x1x8x8x!ttcore.tile<32x32, f32>, #l1_align_big> into tensor<1x1x8x8x!ttcore.tile<32x32, f32>, #l1_align_tile> -> tensor<1x1x8x8x!ttcore.tile<32x32, f32>, #l1_align_tile>
  %out = d2m.empty() : tensor<1x1x8x8x!ttcore.tile<32x32, f32>, #dram>
  %res = d2m.to_layout %reint, %out : tensor<1x1x8x8x!ttcore.tile<32x32, f32>, #l1_align_tile> into tensor<1x1x8x8x!ttcore.tile<32x32, f32>, #dram> -> tensor<1x1x8x8x!ttcore.tile<32x32, f32>, #dram>
  return %res : tensor<1x1x8x8x!ttcore.tile<32x32, f32>, #dram>
}
