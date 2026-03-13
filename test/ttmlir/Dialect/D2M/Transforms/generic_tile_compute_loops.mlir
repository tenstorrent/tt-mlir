// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// RUN: ttmlir-opt --ttcore-register-device --d2m-generic-tile-compute-loops %s | FileCheck %s

// Tests for D2MGenericTileComputeLoops pass.
// The pass inserts d2m.unpack_stall_on_pack before any linalg.generic that
// reads from a buffer written by a preceding linalg.generic (producer/consumer
// dependency), then tiles every linalg.generic into phase-1 scf.for outer
// loops and phase-2 affine.for inner loops.  The outermost phase-2 affine.for
// is tagged with d2m.scratch_space_loop.

#l1 = #ttcore.memory_space<l1>

// ---------------------------------------------------------------------------
// Test: second linalg.generic reads from the first's output buffer →
//       d2m.unpack_stall_on_pack is inserted before the consumer's tiled loop.
// ---------------------------------------------------------------------------

// CHECK-LABEL: func.func @stall_inserted_producer_consumer
// CHECK:       d2m.unpack_stall_on_pack
func.func @stall_inserted_producer_consumer(
    %in0 : memref<1x1x1x1x!ttcore.tile<32x32, bf16>, #ttcore.shard<2048x2048, 1>, #l1>,
    %out0: memref<1x1x1x1x!ttcore.tile<32x32, bf16>, #ttcore.shard<2048x2048, 1>, #l1>) {
  d2m.generic {block_factors = [], grid = #ttcore.grid<1x1>,
               indexing_maps = [], iterator_types = [],
               threads = [#d2m.thread<unified>]}
      ins(%in0 : memref<1x1x1x1x!ttcore.tile<32x32, bf16>, #ttcore.shard<2048x2048, 1>, #l1>)
      outs(%out0 : memref<1x1x1x1x!ttcore.tile<32x32, bf16>, #ttcore.shard<2048x2048, 1>, #l1>) {
  ^unified0:
    %cb0_raw = d2m.get_cb(0) : !d2m.cb<memref<1x1x!ttcore.tile<32x32, bf16>, #l1>>
    %cb1_raw = d2m.get_cb(1) : !d2m.cb<memref<1x1x!ttcore.tile<32x32, bf16>, #l1>>
    %cb0 = d2m.wait %cb0_raw
        : !d2m.cb<memref<1x1x!ttcore.tile<32x32, bf16>, #l1>>
        -> memref<1x1x!ttcore.tile<32x32, bf16>, #l1>
    %cb1 = d2m.reserve %cb1_raw
        : !d2m.cb<memref<1x1x!ttcore.tile<32x32, bf16>, #l1>>
        -> memref<1x1x!ttcore.tile<32x32, bf16>, #l1>
    // Intermediate buffer: linalg1 writes it, linalg2 reads it.
    %tmp = memref.alloc() {alignment = 64 : i64}
        : memref<1x1x!ttcore.tile<32x32, bf16>, #l1>
    // Producer: ins=%cb0, outs=%tmp.
    linalg.generic {
        indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                         affine_map<(d0, d1) -> (d0, d1)>],
        iterator_types = ["parallel", "parallel"]}
        ins(%cb0 : memref<1x1x!ttcore.tile<32x32, bf16>, #l1>)
        outs(%tmp : memref<1x1x!ttcore.tile<32x32, bf16>, #l1>) {
    ^bb0(%in: !ttcore.tile<32x32, bf16>, %out: !ttcore.tile<32x32, bf16>):
      %r = "d2m.tile_exp"(%in) : (!ttcore.tile<32x32, bf16>) -> !ttcore.tile<32x32, bf16>
      linalg.yield %r : !ttcore.tile<32x32, bf16>
    }
    // Consumer: ins=%tmp → stall required before this op.
    linalg.generic {
        indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                         affine_map<(d0, d1) -> (d0, d1)>],
        iterator_types = ["parallel", "parallel"]}
        ins(%tmp : memref<1x1x!ttcore.tile<32x32, bf16>, #l1>)
        outs(%cb1 : memref<1x1x!ttcore.tile<32x32, bf16>, #l1>) {
    ^bb0(%in: !ttcore.tile<32x32, bf16>, %out: !ttcore.tile<32x32, bf16>):
      %r = "d2m.tile_sin"(%in) : (!ttcore.tile<32x32, bf16>) -> !ttcore.tile<32x32, bf16>
      linalg.yield %r : !ttcore.tile<32x32, bf16>
    }
  }
  return
}

// ---------------------------------------------------------------------------
// Test: two independent linalg.generic ops (no output→input dependency) →
//       d2m.unpack_stall_on_pack is NOT inserted.
// ---------------------------------------------------------------------------

// CHECK-LABEL: func.func @no_stall_independent_generics
// CHECK-NOT:   d2m.unpack_stall_on_pack
func.func @no_stall_independent_generics(
    %in0 : memref<1x1x1x1x!ttcore.tile<32x32, bf16>, #ttcore.shard<2048x2048, 1>, #l1>,
    %out0: memref<1x1x1x1x!ttcore.tile<32x32, bf16>, #ttcore.shard<2048x2048, 1>, #l1>) {
  d2m.generic {block_factors = [], grid = #ttcore.grid<1x1>,
               indexing_maps = [], iterator_types = [],
               threads = [#d2m.thread<unified>]}
      ins(%in0 : memref<1x1x1x1x!ttcore.tile<32x32, bf16>, #ttcore.shard<2048x2048, 1>, #l1>)
      outs(%out0 : memref<1x1x1x1x!ttcore.tile<32x32, bf16>, #ttcore.shard<2048x2048, 1>, #l1>) {
  ^unified0:
    %cb0_raw = d2m.get_cb(0) : !d2m.cb<memref<1x1x!ttcore.tile<32x32, bf16>, #l1>>
    %cb1_raw = d2m.get_cb(1) : !d2m.cb<memref<1x1x!ttcore.tile<32x32, bf16>, #l1>>
    %cb0 = d2m.wait %cb0_raw
        : !d2m.cb<memref<1x1x!ttcore.tile<32x32, bf16>, #l1>>
        -> memref<1x1x!ttcore.tile<32x32, bf16>, #l1>
    %cb1 = d2m.reserve %cb1_raw
        : !d2m.cb<memref<1x1x!ttcore.tile<32x32, bf16>, #l1>>
        -> memref<1x1x!ttcore.tile<32x32, bf16>, #l1>
    %tmp1 = memref.alloc() {alignment = 64 : i64}
        : memref<1x1x!ttcore.tile<32x32, bf16>, #l1>
    %tmp2 = memref.alloc() {alignment = 64 : i64}
        : memref<1x1x!ttcore.tile<32x32, bf16>, #l1>
    // linalg1: reads %cb0, writes %tmp1.
    linalg.generic {
        indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                         affine_map<(d0, d1) -> (d0, d1)>],
        iterator_types = ["parallel", "parallel"]}
        ins(%cb0 : memref<1x1x!ttcore.tile<32x32, bf16>, #l1>)
        outs(%tmp1 : memref<1x1x!ttcore.tile<32x32, bf16>, #l1>) {
    ^bb0(%in: !ttcore.tile<32x32, bf16>, %out: !ttcore.tile<32x32, bf16>):
      %r = "d2m.tile_exp"(%in) : (!ttcore.tile<32x32, bf16>) -> !ttcore.tile<32x32, bf16>
      linalg.yield %r : !ttcore.tile<32x32, bf16>
    }
    // linalg2: reads %cb0 (not %tmp1) → no dependency on linalg1's output.
    linalg.generic {
        indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                         affine_map<(d0, d1) -> (d0, d1)>],
        iterator_types = ["parallel", "parallel"]}
        ins(%cb0 : memref<1x1x!ttcore.tile<32x32, bf16>, #l1>)
        outs(%tmp2 : memref<1x1x!ttcore.tile<32x32, bf16>, #l1>) {
    ^bb0(%in: !ttcore.tile<32x32, bf16>, %out: !ttcore.tile<32x32, bf16>):
      %r = "d2m.tile_sin"(%in) : (!ttcore.tile<32x32, bf16>) -> !ttcore.tile<32x32, bf16>
      linalg.yield %r : !ttcore.tile<32x32, bf16>
    }
  }
  return
}

// ---------------------------------------------------------------------------
// Test: 4×4 bf16 unary SFPU (tile_exp, DST capacity = 8 tiles) →
//       two-level tiling with specific block sizes.
//
// Phase-1 (scf.for): outer blocks are [2, 4] tiles, so:
//   dim-0: scf.for 0..4 step 2  (2 iterations covering 4 tiles)
//   dim-1: scf.for 0..4 step 4  (1 iteration covering 4 tiles)
//   → each iteration extracts a [2, 4] subview.
//
// Phase-2 (affine.for): tiles the [2, 4] inner block down to [1, 4]:
//   affine.for 0 to 2           (outermost, tagged {d2m.scratch_space_loop})
//   affine.for 0 to 4 step 4   (single 4-tile step)
//   → innermost linalg.generic processes 1×4 tiles per iteration.
// ---------------------------------------------------------------------------

// CHECK-LABEL: func.func @sfpu_bf16_4x4_two_level_tiling
// CHECK:       scf.for
// CHECK:       scf.for
// CHECK:       memref.subview {{.*}} [2, 4] [1, 1]
// CHECK:       affine.for %{{.*}} = 0 to 2 {
// CHECK-NEXT:    affine.for %{{.*}} = 0 to 4 step 4 {
// CHECK:       } {d2m.scratch_space_loop}
func.func @sfpu_bf16_4x4_two_level_tiling(
    %in0 : memref<1x1x4x4x!ttcore.tile<32x32, bf16>, #ttcore.shard<2048x2048, 1>, #l1>,
    %out0: memref<1x1x4x4x!ttcore.tile<32x32, bf16>, #ttcore.shard<2048x2048, 1>, #l1>) {
  d2m.generic {block_factors = [], grid = #ttcore.grid<1x1>,
               indexing_maps = [], iterator_types = [],
               threads = [#d2m.thread<unified>]}
      ins(%in0 : memref<1x1x4x4x!ttcore.tile<32x32, bf16>, #ttcore.shard<2048x2048, 1>, #l1>)
      outs(%out0 : memref<1x1x4x4x!ttcore.tile<32x32, bf16>, #ttcore.shard<2048x2048, 1>, #l1>) {
  ^unified0:
    %cb0_raw = d2m.get_cb(0) : !d2m.cb<memref<4x4x!ttcore.tile<32x32, bf16>, #l1>>
    %cb1_raw = d2m.get_cb(1) : !d2m.cb<memref<4x4x!ttcore.tile<32x32, bf16>, #l1>>
    %cb0 = d2m.wait %cb0_raw
        : !d2m.cb<memref<4x4x!ttcore.tile<32x32, bf16>, #l1>>
        -> memref<4x4x!ttcore.tile<32x32, bf16>, #l1>
    %cb1 = d2m.reserve %cb1_raw
        : !d2m.cb<memref<4x4x!ttcore.tile<32x32, bf16>, #l1>>
        -> memref<4x4x!ttcore.tile<32x32, bf16>, #l1>
    linalg.generic {
        indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                         affine_map<(d0, d1) -> (d0, d1)>],
        iterator_types = ["parallel", "parallel"]}
        ins(%cb0 : memref<4x4x!ttcore.tile<32x32, bf16>, #l1>)
        outs(%cb1 : memref<4x4x!ttcore.tile<32x32, bf16>, #l1>) {
    ^bb0(%in: !ttcore.tile<32x32, bf16>, %out: !ttcore.tile<32x32, bf16>):
      %r = "d2m.tile_exp"(%in) : (!ttcore.tile<32x32, bf16>) -> !ttcore.tile<32x32, bf16>
      linalg.yield %r : !ttcore.tile<32x32, bf16>
    }
  }
  return
}

// ---------------------------------------------------------------------------
// Test: 4×4 f32 unary SFPU (tile_exp, DST capacity = 4 tiles = half of bf16).
//       Tighter blocks because f32 tiles are twice as large.
//
// Phase-1 (scf.for): outer blocks are [1, 4] tiles:
//   dim-0: scf.for 0..4 step 1  (4 iterations, one tile row per iteration)
//   dim-1: scf.for 0..4 step 4  (1 iteration covering 4 tiles)
//   → each iteration extracts a [1, 4] subview.
//
// Phase-2 (affine.for): tiles [1, 4] down to [1, 2]:
//   affine.for 0 to 1           (outermost, tagged {d2m.scratch_space_loop})
//   affine.for 0 to 4 step 2   (2 steps of 2 tiles each)
//   → innermost linalg.generic processes 1×2 tiles per iteration.
// ---------------------------------------------------------------------------

// CHECK-LABEL: func.func @sfpu_f32_4x4_tighter_blocks
// CHECK:       scf.for
// CHECK:       scf.for
// CHECK:       memref.subview {{.*}} [1, 4] [1, 1]
// CHECK:       affine.for %{{.*}} = 0 to 1 {
// CHECK-NEXT:    affine.for %{{.*}} = 0 to 4 step 2 {
// CHECK:       } {d2m.scratch_space_loop}
func.func @sfpu_f32_4x4_tighter_blocks(
    %in0 : memref<1x1x4x4x!ttcore.tile<32x32, f32>, #ttcore.shard<4096x4096, 1>, #l1>,
    %out0: memref<1x1x4x4x!ttcore.tile<32x32, f32>, #ttcore.shard<4096x4096, 1>, #l1>) {
  d2m.generic {block_factors = [], grid = #ttcore.grid<1x1>,
               indexing_maps = [], iterator_types = [],
               threads = [#d2m.thread<unified>]}
      ins(%in0 : memref<1x1x4x4x!ttcore.tile<32x32, f32>, #ttcore.shard<4096x4096, 1>, #l1>)
      outs(%out0 : memref<1x1x4x4x!ttcore.tile<32x32, f32>, #ttcore.shard<4096x4096, 1>, #l1>) {
  ^unified0:
    %cb0_raw = d2m.get_cb(0) : !d2m.cb<memref<4x4x!ttcore.tile<32x32, f32>, #l1>>
    %cb1_raw = d2m.get_cb(1) : !d2m.cb<memref<4x4x!ttcore.tile<32x32, f32>, #l1>>
    %cb0 = d2m.wait %cb0_raw
        : !d2m.cb<memref<4x4x!ttcore.tile<32x32, f32>, #l1>>
        -> memref<4x4x!ttcore.tile<32x32, f32>, #l1>
    %cb1 = d2m.reserve %cb1_raw
        : !d2m.cb<memref<4x4x!ttcore.tile<32x32, f32>, #l1>>
        -> memref<4x4x!ttcore.tile<32x32, f32>, #l1>
    linalg.generic {
        indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                         affine_map<(d0, d1) -> (d0, d1)>],
        iterator_types = ["parallel", "parallel"]}
        ins(%cb0 : memref<4x4x!ttcore.tile<32x32, f32>, #l1>)
        outs(%cb1 : memref<4x4x!ttcore.tile<32x32, f32>, #l1>) {
    ^bb0(%in: !ttcore.tile<32x32, f32>, %out: !ttcore.tile<32x32, f32>):
      %r = "d2m.tile_exp"(%in) : (!ttcore.tile<32x32, f32>) -> !ttcore.tile<32x32, f32>
      linalg.yield %r : !ttcore.tile<32x32, f32>
    }
  }
  return
}

// ---------------------------------------------------------------------------
// Test: 4×4 bf16 producer→consumer (tile_exp → tile_sin).
//       The stall is inserted between the two tiled loop nests, and both
//       nests use the same two-level bf16 SFPU tiling as
//       @sfpu_bf16_4x4_two_level_tiling (phase-1 [2,4], phase-2 [1,4]).
// ---------------------------------------------------------------------------

// CHECK-LABEL: func.func @producer_consumer_4x4_stall_and_tiling
// Phase-1 and phase-2 loops for the producer (tile_exp) nest:
// CHECK:       scf.for
// CHECK:       scf.for
// CHECK:       memref.subview {{.*}} [2, 4] [1, 1]
// CHECK:       affine.for %{{.*}} = 0 to 2 {
// CHECK:       } {d2m.scratch_space_loop}
// Stall is emitted between the two tiled nests:
// CHECK:       d2m.unpack_stall_on_pack
// Phase-1 and phase-2 loops for the consumer (tile_sin) nest:
// CHECK:       scf.for
// CHECK:       scf.for
// CHECK:       memref.subview {{.*}} [2, 4] [1, 1]
// CHECK:       affine.for %{{.*}} = 0 to 2 {
// CHECK:       } {d2m.scratch_space_loop}
func.func @producer_consumer_4x4_stall_and_tiling(
    %in0 : memref<1x1x4x4x!ttcore.tile<32x32, bf16>, #ttcore.shard<2048x2048, 1>, #l1>,
    %out0: memref<1x1x4x4x!ttcore.tile<32x32, bf16>, #ttcore.shard<2048x2048, 1>, #l1>) {
  d2m.generic {block_factors = [], grid = #ttcore.grid<1x1>,
               indexing_maps = [], iterator_types = [],
               threads = [#d2m.thread<unified>]}
      ins(%in0 : memref<1x1x4x4x!ttcore.tile<32x32, bf16>, #ttcore.shard<2048x2048, 1>, #l1>)
      outs(%out0 : memref<1x1x4x4x!ttcore.tile<32x32, bf16>, #ttcore.shard<2048x2048, 1>, #l1>) {
  ^unified0:
    %cb0_raw = d2m.get_cb(0) : !d2m.cb<memref<4x4x!ttcore.tile<32x32, bf16>, #l1>>
    %cb1_raw = d2m.get_cb(1) : !d2m.cb<memref<4x4x!ttcore.tile<32x32, bf16>, #l1>>
    %cb0 = d2m.wait %cb0_raw
        : !d2m.cb<memref<4x4x!ttcore.tile<32x32, bf16>, #l1>>
        -> memref<4x4x!ttcore.tile<32x32, bf16>, #l1>
    %cb1 = d2m.reserve %cb1_raw
        : !d2m.cb<memref<4x4x!ttcore.tile<32x32, bf16>, #l1>>
        -> memref<4x4x!ttcore.tile<32x32, bf16>, #l1>
    %tmp = memref.alloc() {alignment = 64 : i64}
        : memref<4x4x!ttcore.tile<32x32, bf16>, #l1>
    // Producer: tile_exp writes intermediate buffer %tmp.
    linalg.generic {
        indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                         affine_map<(d0, d1) -> (d0, d1)>],
        iterator_types = ["parallel", "parallel"]}
        ins(%cb0 : memref<4x4x!ttcore.tile<32x32, bf16>, #l1>)
        outs(%tmp : memref<4x4x!ttcore.tile<32x32, bf16>, #l1>) {
    ^bb0(%in: !ttcore.tile<32x32, bf16>, %out: !ttcore.tile<32x32, bf16>):
      %r = "d2m.tile_exp"(%in) : (!ttcore.tile<32x32, bf16>) -> !ttcore.tile<32x32, bf16>
      linalg.yield %r : !ttcore.tile<32x32, bf16>
    }
    // Consumer: tile_sin reads %tmp → stall inserted before this tiled nest.
    linalg.generic {
        indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                         affine_map<(d0, d1) -> (d0, d1)>],
        iterator_types = ["parallel", "parallel"]}
        ins(%tmp : memref<4x4x!ttcore.tile<32x32, bf16>, #l1>)
        outs(%cb1 : memref<4x4x!ttcore.tile<32x32, bf16>, #l1>) {
    ^bb0(%in: !ttcore.tile<32x32, bf16>, %out: !ttcore.tile<32x32, bf16>):
      %r = "d2m.tile_sin"(%in) : (!ttcore.tile<32x32, bf16>) -> !ttcore.tile<32x32, bf16>
      linalg.yield %r : !ttcore.tile<32x32, bf16>
    }
  }
  return
}
