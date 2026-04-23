// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// RUN: ttmlir-opt --d2m-insert-spill-and-scratch %s | FileCheck %s

// Tests for D2MInsertSpillAndScratch pass.
// The pass replaces intermediate memref.alloc buffers that flow from one
// scratch_space_loop nest to another with d2m.scratch_allocate ops, and
// rewrites the corresponding affine.store/affine.load ops to use the scratch
// buffer.  It also fuses matching outer scf.for wrappers.

#l1 = #ttcore.memory_space<l1>

// ---------------------------------------------------------------------------
// Test: three scratch_space_loop nests in a linear A→B→C chain with two
//       intermediate allocs.  Verifies that:
//         - two scratch_allocate ops are created with slot 0 and slot 1
//         - nest A stores into slot-0 scratch
//         - nest B loads from slot-0 scratch and stores into slot-1 scratch
//         - nest C loads from slot-1 scratch
// ---------------------------------------------------------------------------

// CHECK-LABEL: func.func @three_step_chain
// CHECK-DAG:   %[[SCRATCH_A:.*]] = d2m.scratch_allocate {slot = 0 : i64} : memref<2x1x!ttcore.tile<32x32, bf16>, #l1>
// CHECK-DAG:   %[[SCRATCH_B:.*]] = d2m.scratch_allocate {slot = 1 : i64} : memref<2x1x!ttcore.tile<32x32, bf16>, #l1>
//
// Nest A – stores tile_exp result into slot-0 scratch, outer loop tagged.
// CHECK:       "d2m.tile_exp"
// CHECK:       affine.store {{.*}}, %[[SCRATCH_A]]
// CHECK:       } {d2m.scratch_inserted, d2m.scratch_space_loop}
//
// Nest B – loads from slot-0, stores tile_sin result into slot-1, outer loop tagged.
// CHECK:       affine.load %[[SCRATCH_A]]
// CHECK:       "d2m.tile_sin"
// CHECK:       affine.store {{.*}}, %[[SCRATCH_B]]
// CHECK:       } {d2m.scratch_inserted, d2m.scratch_space_loop}
//
// Nest C – loads from slot-1, outer loop tagged.
// CHECK:       affine.load %[[SCRATCH_B]]
// CHECK:       "d2m.tile_abs"
// CHECK:       } {d2m.scratch_inserted, d2m.scratch_space_loop}
//
// Original allocs must be gone.
// CHECK-NOT:   memref.alloc
func.func @three_step_chain(
    %in0  : memref<1x1x2x1x!ttcore.tile<32x32, bf16>, #ttcore.shard<2048x2048, 1>, #l1>,
    %out0 : memref<1x1x2x1x!ttcore.tile<32x32, bf16>, #ttcore.shard<2048x2048, 1>, #l1>) {
  d2m.generic {block_factors = [], grid = #ttcore.grid<1x1>,
               indexing_maps = [], iterator_types = [],
               threads = [#d2m.thread<unified>]}
      ins(%in0 : memref<1x1x2x1x!ttcore.tile<32x32, bf16>, #ttcore.shard<2048x2048, 1>, #l1>)
      outs(%out0 : memref<1x1x2x1x!ttcore.tile<32x32, bf16>, #ttcore.shard<2048x2048, 1>, #l1>) {
  ^unified0:
    %cb0_raw = d2m.get_cb(0) : !d2m.cb<memref<2x1x!ttcore.tile<32x32, bf16>, #l1>>
    %cb1_raw = d2m.get_cb(1) : !d2m.cb<memref<2x1x!ttcore.tile<32x32, bf16>, #l1>>
    %cb0 = d2m.wait %cb0_raw
        : !d2m.cb<memref<2x1x!ttcore.tile<32x32, bf16>, #l1>>
        -> memref<2x1x!ttcore.tile<32x32, bf16>, #l1>
    %cb1 = d2m.reserve %cb1_raw
        : !d2m.cb<memref<2x1x!ttcore.tile<32x32, bf16>, #l1>>
        -> memref<2x1x!ttcore.tile<32x32, bf16>, #l1>
    %tmp_a = memref.alloc() {alignment = 64 : i64}
        : memref<2x1x!ttcore.tile<32x32, bf16>, #l1>
    %tmp_b = memref.alloc() {alignment = 64 : i64}
        : memref<2x1x!ttcore.tile<32x32, bf16>, #l1>
    // Nest A: cb0 → tmp_a (tile_exp).
    affine.for %i = 0 to 2 {
      affine.for %j = 0 to 1 {
        %v = affine.load %cb0[%i, %j]  : memref<2x1x!ttcore.tile<32x32, bf16>, #l1>
        %r = "d2m.tile_exp"(%v) : (!ttcore.tile<32x32, bf16>) -> !ttcore.tile<32x32, bf16>
        affine.store %r, %tmp_a[%i, %j] : memref<2x1x!ttcore.tile<32x32, bf16>, #l1>
      } {d2m.linalg_root}
    } {d2m.scratch_space_loop}
    // Nest B: tmp_a → tmp_b (tile_sin).
    affine.for %i = 0 to 2 {
      affine.for %j = 0 to 1 {
        %v = affine.load %tmp_a[%i, %j] : memref<2x1x!ttcore.tile<32x32, bf16>, #l1>
        %r = "d2m.tile_sin"(%v) : (!ttcore.tile<32x32, bf16>) -> !ttcore.tile<32x32, bf16>
        affine.store %r, %tmp_b[%i, %j] : memref<2x1x!ttcore.tile<32x32, bf16>, #l1>
      } {d2m.linalg_root}
    } {d2m.scratch_space_loop}
    // Nest C: tmp_b → cb1 (tile_abs).
    affine.for %i = 0 to 2 {
      affine.for %j = 0 to 1 {
        %v = affine.load %tmp_b[%i, %j] : memref<2x1x!ttcore.tile<32x32, bf16>, #l1>
        %r = "d2m.tile_abs"(%v) : (!ttcore.tile<32x32, bf16>) -> !ttcore.tile<32x32, bf16>
        affine.store %r, %cb1[%i, %j]   : memref<2x1x!ttcore.tile<32x32, bf16>, #l1>
      } {d2m.linalg_root}
    } {d2m.scratch_space_loop}
  }
  return
}

// ---------------------------------------------------------------------------
// Test: two independent producer nests (A and B) each write an intermediate,
//       which are then read together by a single consumer nest (C) that
//       combines them with a binary op.  Verifies that:
//         - two scratch_allocate ops are created (slot 0 for tmp_a, slot 1
//           for tmp_b)
//         - nest C performs two loads: one from slot-0 scratch (tmp_a) and
//           one from slot-1 scratch (tmp_b)
//         - the combined result is stored to the output CB
// ---------------------------------------------------------------------------

// CHECK-LABEL: func.func @two_intermediates_one_consumer
// CHECK-DAG:   %[[SCRATCH_A:.*]] = d2m.scratch_allocate {slot = 0 : i64} : memref<2x1x!ttcore.tile<32x32, bf16>, #l1>
// CHECK-DAG:   %[[SCRATCH_B:.*]] = d2m.scratch_allocate {slot = 1 : i64} : memref<2x1x!ttcore.tile<32x32, bf16>, #l1>
//
// Nest A – tile_exp result stored to slot-0 scratch, outer loop tagged.
// CHECK:       "d2m.tile_exp"
// CHECK:       affine.store {{.*}}, %[[SCRATCH_A]]
// CHECK:       } {d2m.scratch_inserted, d2m.scratch_space_loop}
//
// Nest B – tile_sin result stored to slot-1 scratch, outer loop tagged.
// CHECK:       "d2m.tile_sin"
// CHECK:       affine.store {{.*}}, %[[SCRATCH_B]]
// CHECK:       } {d2m.scratch_inserted, d2m.scratch_space_loop}
//
// Nest C – loads from BOTH scratch slots before the binary tile_add, outer loop tagged.
// CHECK:       affine.load %[[SCRATCH_A]]
// CHECK:       affine.load %[[SCRATCH_B]]
// CHECK:       "d2m.tile_add"
// CHECK:       } {d2m.scratch_inserted, d2m.scratch_space_loop}
// CHECK-NOT:   memref.alloc
func.func @two_intermediates_one_consumer(
    %in0  : memref<1x1x2x1x!ttcore.tile<32x32, bf16>, #ttcore.shard<2048x2048, 1>, #l1>,
    %in1  : memref<1x1x2x1x!ttcore.tile<32x32, bf16>, #ttcore.shard<2048x2048, 1>, #l1>,
    %out0 : memref<1x1x2x1x!ttcore.tile<32x32, bf16>, #ttcore.shard<2048x2048, 1>, #l1>) {
  d2m.generic {block_factors = [], grid = #ttcore.grid<1x1>,
               indexing_maps = [], iterator_types = [],
               threads = [#d2m.thread<unified>]}
      ins(%in0, %in1 :
          memref<1x1x2x1x!ttcore.tile<32x32, bf16>, #ttcore.shard<2048x2048, 1>, #l1>,
          memref<1x1x2x1x!ttcore.tile<32x32, bf16>, #ttcore.shard<2048x2048, 1>, #l1>)
      outs(%out0 : memref<1x1x2x1x!ttcore.tile<32x32, bf16>, #ttcore.shard<2048x2048, 1>, #l1>) {
  ^unified0:
    %cb0_raw = d2m.get_cb(0) : !d2m.cb<memref<2x1x!ttcore.tile<32x32, bf16>, #l1>>
    %cb1_raw = d2m.get_cb(1) : !d2m.cb<memref<2x1x!ttcore.tile<32x32, bf16>, #l1>>
    %cb2_raw = d2m.get_cb(2) : !d2m.cb<memref<2x1x!ttcore.tile<32x32, bf16>, #l1>>
    %cb0 = d2m.wait %cb0_raw
        : !d2m.cb<memref<2x1x!ttcore.tile<32x32, bf16>, #l1>>
        -> memref<2x1x!ttcore.tile<32x32, bf16>, #l1>
    %cb1 = d2m.wait %cb1_raw
        : !d2m.cb<memref<2x1x!ttcore.tile<32x32, bf16>, #l1>>
        -> memref<2x1x!ttcore.tile<32x32, bf16>, #l1>
    %cb2 = d2m.reserve %cb2_raw
        : !d2m.cb<memref<2x1x!ttcore.tile<32x32, bf16>, #l1>>
        -> memref<2x1x!ttcore.tile<32x32, bf16>, #l1>
    %tmp_a = memref.alloc() {alignment = 64 : i64}
        : memref<2x1x!ttcore.tile<32x32, bf16>, #l1>
    %tmp_b = memref.alloc() {alignment = 64 : i64}
        : memref<2x1x!ttcore.tile<32x32, bf16>, #l1>
    // Nest A: cb0 → tmp_a (tile_exp).
    affine.for %i = 0 to 2 {
      affine.for %j = 0 to 1 {
        %v = affine.load %cb0[%i, %j]  : memref<2x1x!ttcore.tile<32x32, bf16>, #l1>
        %r = "d2m.tile_exp"(%v) : (!ttcore.tile<32x32, bf16>) -> !ttcore.tile<32x32, bf16>
        affine.store %r, %tmp_a[%i, %j] : memref<2x1x!ttcore.tile<32x32, bf16>, #l1>
      } {d2m.linalg_root}
    } {d2m.scratch_space_loop}
    // Nest B: cb1 → tmp_b (tile_sin).
    affine.for %i = 0 to 2 {
      affine.for %j = 0 to 1 {
        %v = affine.load %cb1[%i, %j]  : memref<2x1x!ttcore.tile<32x32, bf16>, #l1>
        %r = "d2m.tile_sin"(%v) : (!ttcore.tile<32x32, bf16>) -> !ttcore.tile<32x32, bf16>
        affine.store %r, %tmp_b[%i, %j] : memref<2x1x!ttcore.tile<32x32, bf16>, #l1>
      } {d2m.linalg_root}
    } {d2m.scratch_space_loop}
    // Nest C: reads BOTH tmp_a and tmp_b, combines with tile_add → cb2.
    affine.for %i = 0 to 2 {
      affine.for %j = 0 to 1 {
        %va = affine.load %tmp_a[%i, %j] : memref<2x1x!ttcore.tile<32x32, bf16>, #l1>
        %vb = affine.load %tmp_b[%i, %j] : memref<2x1x!ttcore.tile<32x32, bf16>, #l1>
        %r  = "d2m.tile_add"(%va, %vb)
            : (!ttcore.tile<32x32, bf16>, !ttcore.tile<32x32, bf16>) -> !ttcore.tile<32x32, bf16>
        affine.store %r, %cb2[%i, %j]   : memref<2x1x!ttcore.tile<32x32, bf16>, #l1>
      } {d2m.linalg_root}
    } {d2m.scratch_space_loop}
  }
  return
}

// ---------------------------------------------------------------------------
// Test: only one scratch_space_loop nest → pass requires ≥2 nests to identify
//       a producer/consumer pair, so no scratch is inserted.
// ---------------------------------------------------------------------------

// CHECK-LABEL: func.func @single_nest_no_scratch
// CHECK-NOT:   d2m.scratch_allocate
func.func @single_nest_no_scratch(
    %in0  : memref<1x1x2x1x!ttcore.tile<32x32, bf16>, #ttcore.shard<2048x2048, 1>, #l1>,
    %out0 : memref<1x1x2x1x!ttcore.tile<32x32, bf16>, #ttcore.shard<2048x2048, 1>, #l1>) {
  d2m.generic {block_factors = [], grid = #ttcore.grid<1x1>,
               indexing_maps = [], iterator_types = [],
               threads = [#d2m.thread<unified>]}
      ins(%in0 : memref<1x1x2x1x!ttcore.tile<32x32, bf16>, #ttcore.shard<2048x2048, 1>, #l1>)
      outs(%out0 : memref<1x1x2x1x!ttcore.tile<32x32, bf16>, #ttcore.shard<2048x2048, 1>, #l1>) {
  ^unified0:
    %cb0_raw = d2m.get_cb(0) : !d2m.cb<memref<2x1x!ttcore.tile<32x32, bf16>, #l1>>
    %cb1_raw = d2m.get_cb(1) : !d2m.cb<memref<2x1x!ttcore.tile<32x32, bf16>, #l1>>
    %cb0 = d2m.wait %cb0_raw
        : !d2m.cb<memref<2x1x!ttcore.tile<32x32, bf16>, #l1>>
        -> memref<2x1x!ttcore.tile<32x32, bf16>, #l1>
    %cb1 = d2m.reserve %cb1_raw
        : !d2m.cb<memref<2x1x!ttcore.tile<32x32, bf16>, #l1>>
        -> memref<2x1x!ttcore.tile<32x32, bf16>, #l1>
    affine.for %i = 0 to 2 {
      affine.for %j = 0 to 1 {
        %v = affine.load %cb0[%i, %j] : memref<2x1x!ttcore.tile<32x32, bf16>, #l1>
        %r = "d2m.tile_exp"(%v) : (!ttcore.tile<32x32, bf16>) -> !ttcore.tile<32x32, bf16>
        affine.store %r, %cb1[%i, %j]  : memref<2x1x!ttcore.tile<32x32, bf16>, #l1>
      } {d2m.linalg_root}
    } {d2m.scratch_space_loop}
  }
  return
}

// ---------------------------------------------------------------------------
// Test: two scratch_space_loop nests wrapped in matching inner non-blocking
//       scf.for loops, all of which sit inside a shared blocking scf.for
//       (d2m.blocking_loop = 0).  The pass should:
//         - stop climbing the parent chain at the blocking loop, so only the
//           inner (non-blocking) scf.for is collected as a fusion candidate
//         - fuse those inner non-blocking scf.for wrappers into one loop
//         - leave the outer blocking scf.for untouched
//         - insert a scratch_allocate as usual
// ---------------------------------------------------------------------------

// CHECK-LABEL: func.func @blocking_loop_inner_scf_fusion
// CHECK:       %[[SCRATCH:.*]] = d2m.scratch_allocate {slot = 0 : i64}
// Outer blocking loop is preserved.
// CHECK:       scf.for
// The two inner non-blocking scf.for wrappers are fused into one.
// CHECK:       scf.for
// CHECK-NOT:   scf.for
// Both scratch nests are now tagged d2m.scratch_inserted.
// CHECK:       } {d2m.scratch_inserted, d2m.scratch_space_loop}
// CHECK:       } {d2m.scratch_inserted, d2m.scratch_space_loop}
// Outer blocking loop attribute is preserved.
// CHECK:       } {d2m.blocking_loop = 0 : i64}
// CHECK-NOT:   memref.alloc
func.func @blocking_loop_inner_scf_fusion(
    %in0  : memref<1x1x2x1x!ttcore.tile<32x32, bf16>, #ttcore.shard<2048x2048, 1>, #l1>,
    %out0 : memref<1x1x2x1x!ttcore.tile<32x32, bf16>, #ttcore.shard<2048x2048, 1>, #l1>) {
  d2m.generic {block_factors = [], grid = #ttcore.grid<1x1>,
               indexing_maps = [], iterator_types = [],
               threads = [#d2m.thread<unified>]}
      ins(%in0 : memref<1x1x2x1x!ttcore.tile<32x32, bf16>, #ttcore.shard<2048x2048, 1>, #l1>)
      outs(%out0 : memref<1x1x2x1x!ttcore.tile<32x32, bf16>, #ttcore.shard<2048x2048, 1>, #l1>) {
  ^unified0:
    %cb0_raw = d2m.get_cb(0) : !d2m.cb<memref<2x1x!ttcore.tile<32x32, bf16>, #l1>>
    %cb1_raw = d2m.get_cb(1) : !d2m.cb<memref<2x1x!ttcore.tile<32x32, bf16>, #l1>>
    %cb0 = d2m.wait %cb0_raw
        : !d2m.cb<memref<2x1x!ttcore.tile<32x32, bf16>, #l1>>
        -> memref<2x1x!ttcore.tile<32x32, bf16>, #l1>
    %cb1 = d2m.reserve %cb1_raw
        : !d2m.cb<memref<2x1x!ttcore.tile<32x32, bf16>, #l1>>
        -> memref<2x1x!ttcore.tile<32x32, bf16>, #l1>
    %tmp = memref.alloc() {alignment = 64 : i64}
        : memref<2x1x!ttcore.tile<32x32, bf16>, #l1>
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c4 = arith.constant 4 : index
    // Blocking loop (shared context for the whole shard tile).
    scf.for %bi = %c0 to %c1 step %c1 {
      // Each scratch nest is wrapped in its own matching non-blocking scf.for.
      // Nest A: cb0 → tmp (tile_exp).
      scf.for %k = %c0 to %c4 step %c1 {
        affine.for %i = 0 to 2 {
          affine.for %j = 0 to 1 {
            %v = affine.load %cb0[%i, %j] : memref<2x1x!ttcore.tile<32x32, bf16>, #l1>
            %r = "d2m.tile_exp"(%v) : (!ttcore.tile<32x32, bf16>) -> !ttcore.tile<32x32, bf16>
            affine.store %r, %tmp[%i, %j] : memref<2x1x!ttcore.tile<32x32, bf16>, #l1>
          } {d2m.linalg_root}
        } {d2m.scratch_space_loop}
      }
      // Nest B: tmp → cb1 (tile_sin).
      scf.for %k = %c0 to %c4 step %c1 {
        affine.for %i = 0 to 2 {
          affine.for %j = 0 to 1 {
            %v = affine.load %tmp[%i, %j] : memref<2x1x!ttcore.tile<32x32, bf16>, #l1>
            %r = "d2m.tile_sin"(%v) : (!ttcore.tile<32x32, bf16>) -> !ttcore.tile<32x32, bf16>
            affine.store %r, %cb1[%i, %j] : memref<2x1x!ttcore.tile<32x32, bf16>, #l1>
          } {d2m.linalg_root}
        } {d2m.scratch_space_loop}
      }
    } {d2m.blocking_loop = 0}
  }
  return
}

// ---------------------------------------------------------------------------
// Test: loops carrying d2m.scratch_inserted are filtered out before processing
//       → pass is a no-op and no additional scratch is created.
// ---------------------------------------------------------------------------

// CHECK-LABEL: func.func @already_marked_is_skipped
// CHECK-NOT:   d2m.scratch_allocate
func.func @already_marked_is_skipped(
    %in0  : memref<1x1x2x1x!ttcore.tile<32x32, bf16>, #ttcore.shard<2048x2048, 1>, #l1>,
    %out0 : memref<1x1x2x1x!ttcore.tile<32x32, bf16>, #ttcore.shard<2048x2048, 1>, #l1>) {
  d2m.generic {block_factors = [], grid = #ttcore.grid<1x1>,
               indexing_maps = [], iterator_types = [],
               threads = [#d2m.thread<unified>]}
      ins(%in0 : memref<1x1x2x1x!ttcore.tile<32x32, bf16>, #ttcore.shard<2048x2048, 1>, #l1>)
      outs(%out0 : memref<1x1x2x1x!ttcore.tile<32x32, bf16>, #ttcore.shard<2048x2048, 1>, #l1>) {
  ^unified0:
    %cb0_raw = d2m.get_cb(0) : !d2m.cb<memref<2x1x!ttcore.tile<32x32, bf16>, #l1>>
    %cb1_raw = d2m.get_cb(1) : !d2m.cb<memref<2x1x!ttcore.tile<32x32, bf16>, #l1>>
    %cb0 = d2m.wait %cb0_raw
        : !d2m.cb<memref<2x1x!ttcore.tile<32x32, bf16>, #l1>>
        -> memref<2x1x!ttcore.tile<32x32, bf16>, #l1>
    %cb1 = d2m.reserve %cb1_raw
        : !d2m.cb<memref<2x1x!ttcore.tile<32x32, bf16>, #l1>>
        -> memref<2x1x!ttcore.tile<32x32, bf16>, #l1>
    %tmp = memref.alloc() {alignment = 64 : i64}
        : memref<2x1x!ttcore.tile<32x32, bf16>, #l1>
    affine.for %i = 0 to 2 {
      affine.for %j = 0 to 1 {
        %v = affine.load %cb0[%i, %j] : memref<2x1x!ttcore.tile<32x32, bf16>, #l1>
        affine.store %v, %tmp[%i, %j]  : memref<2x1x!ttcore.tile<32x32, bf16>, #l1>
      } {d2m.linalg_root}
    } {d2m.scratch_inserted, d2m.scratch_space_loop}
    affine.for %i = 0 to 2 {
      affine.for %j = 0 to 1 {
        %v = affine.load %tmp[%i, %j] : memref<2x1x!ttcore.tile<32x32, bf16>, #l1>
        affine.store %v, %cb1[%i, %j]  : memref<2x1x!ttcore.tile<32x32, bf16>, #l1>
      } {d2m.linalg_root}
    } {d2m.scratch_inserted, d2m.scratch_space_loop}
  }
  return
}

// ---------------------------------------------------------------------------
// Test: two scratch_space_loop nests each wrapped in matching scf.for loops →
//       the pass fuses the outer scf.for loops into one shared loop and
//       inserts a single scratch_allocate.
// ---------------------------------------------------------------------------

// CHECK-LABEL: func.func @outer_scf_loops_get_fused
// CHECK:       %[[SCRATCH:.*]] = d2m.scratch_allocate {slot = 0 : i64}
// CHECK:       scf.for
// CHECK-NOT:   scf.for
// CHECK:       } {d2m.scratch_inserted, d2m.scratch_space_loop}
func.func @outer_scf_loops_get_fused(
    %in0  : memref<1x1x2x1x!ttcore.tile<32x32, bf16>, #ttcore.shard<2048x2048, 1>, #l1>,
    %out0 : memref<1x1x2x1x!ttcore.tile<32x32, bf16>, #ttcore.shard<2048x2048, 1>, #l1>) {
  d2m.generic {block_factors = [], grid = #ttcore.grid<1x1>,
               indexing_maps = [], iterator_types = [],
               threads = [#d2m.thread<unified>]}
      ins(%in0 : memref<1x1x2x1x!ttcore.tile<32x32, bf16>, #ttcore.shard<2048x2048, 1>, #l1>)
      outs(%out0 : memref<1x1x2x1x!ttcore.tile<32x32, bf16>, #ttcore.shard<2048x2048, 1>, #l1>) {
  ^unified0:
    %cb0_raw = d2m.get_cb(0) : !d2m.cb<memref<2x1x!ttcore.tile<32x32, bf16>, #l1>>
    %cb1_raw = d2m.get_cb(1) : !d2m.cb<memref<2x1x!ttcore.tile<32x32, bf16>, #l1>>
    %cb0 = d2m.wait %cb0_raw
        : !d2m.cb<memref<2x1x!ttcore.tile<32x32, bf16>, #l1>>
        -> memref<2x1x!ttcore.tile<32x32, bf16>, #l1>
    %cb1 = d2m.reserve %cb1_raw
        : !d2m.cb<memref<2x1x!ttcore.tile<32x32, bf16>, #l1>>
        -> memref<2x1x!ttcore.tile<32x32, bf16>, #l1>
    %tmp = memref.alloc() {alignment = 64 : i64}
        : memref<2x1x!ttcore.tile<32x32, bf16>, #l1>
    %c0 = arith.constant 0 : index
    %c4 = arith.constant 4 : index
    %c1 = arith.constant 1 : index
    scf.for %k = %c0 to %c4 step %c1 {
      affine.for %i = 0 to 2 {
        affine.for %j = 0 to 1 {
          %v = affine.load %cb0[%i, %j] : memref<2x1x!ttcore.tile<32x32, bf16>, #l1>
          %r = "d2m.tile_exp"(%v) : (!ttcore.tile<32x32, bf16>) -> !ttcore.tile<32x32, bf16>
          affine.store %r, %tmp[%i, %j]  : memref<2x1x!ttcore.tile<32x32, bf16>, #l1>
        } {d2m.linalg_root}
      } {d2m.scratch_space_loop}
    }
    scf.for %k = %c0 to %c4 step %c1 {
      affine.for %i = 0 to 2 {
        affine.for %j = 0 to 1 {
          %v = affine.load %tmp[%i, %j] : memref<2x1x!ttcore.tile<32x32, bf16>, #l1>
          %r = "d2m.tile_sin"(%v) : (!ttcore.tile<32x32, bf16>) -> !ttcore.tile<32x32, bf16>
          affine.store %r, %cb1[%i, %j]  : memref<2x1x!ttcore.tile<32x32, bf16>, #l1>
        } {d2m.linalg_root}
      } {d2m.scratch_space_loop}
    }
  }
  return
}
