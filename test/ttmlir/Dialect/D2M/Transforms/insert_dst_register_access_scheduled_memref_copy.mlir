// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// Tests for the scheduled pass handling of memref.load / memref.store ops.
// Exercises:
//   - memref.load → DST (via affine.store to DST, replaces with DST load)
//   - memref.store from DST → CB (via affine.load from DST)
//   - Simple memref.load → memref.store copy pattern (DecomposeMasking-like)
//     collected by the separate walker in collectDstAccessesScheduled.
// RUN: ttmlir-opt --ttcore-register-device --d2m-insert-dst-register-access-scheduled --canonicalize -o %t %s
// RUN: FileCheck %s --input-file=%t

#l1_ = #ttcore.memory_space<l1>
#dst_ = #ttcore.memory_space<dst>

module {
  // -----------------------------------------------------------------------
  // Compute op (tile_exp via affine) + simple memref.load → memref.store copy.
  // Both patterns coexist in the same scheduled root loop.
  // The copy pattern is handled by the separate memref::StoreOp walker in
  // collectDstAccessesScheduled (not the compute op walker).
  // -----------------------------------------------------------------------
  // CHECK-LABEL: func.func @scheduled_with_memref_copy
  func.func @scheduled_with_memref_copy(
      %in0: memref<1x1x1x1x!ttcore.tile<32x32, f32>, #ttcore.shard<4096x4096, 1>, #l1_>,
      %out0: memref<1x1x1x1x!ttcore.tile<32x32, f32>, #ttcore.shard<4096x4096, 1>, #l1_>) {
    d2m.generic {block_factors = [], grid = #ttcore.grid<1x1>, indexing_maps = [], iterator_types = [], threads = [#d2m.thread<unified>]}
        ins(%in0 : memref<1x1x1x1x!ttcore.tile<32x32, f32>, #ttcore.shard<4096x4096, 1>, #l1_>)
        outs(%out0 : memref<1x1x1x1x!ttcore.tile<32x32, f32>, #ttcore.shard<4096x4096, 1>, #l1_>) {
    ^unified0:
      %cb0_raw = d2m.get_cb(0) : !d2m.cb<memref<1x1x!ttcore.tile<32x32, f32>, #l1_>>
      %cb1_raw = d2m.get_cb(1) : !d2m.cb<memref<1x1x!ttcore.tile<32x32, f32>, #l1_>>
      %cb0 = d2m.wait %cb0_raw : !d2m.cb<memref<1x1x!ttcore.tile<32x32, f32>, #l1_>> -> memref<1x1x!ttcore.tile<32x32, f32>, #l1_>
      %cb1 = d2m.reserve %cb1_raw : !d2m.cb<memref<1x1x!ttcore.tile<32x32, f32>, #l1_>> -> memref<1x1x!ttcore.tile<32x32, f32>, #l1_>
      %c0 = arith.constant 0 : index
      affine.for %i = 0 to 1 {
        affine.for %j = 0 to 1 {
          // Compute op path (affine loads/stores):
          %v = affine.load %cb0[%i, %j] : memref<1x1x!ttcore.tile<32x32, f32>, #l1_>
          %exp = "d2m.tile_exp"(%v) : (!ttcore.tile<32x32, f32>) -> !ttcore.tile<32x32, f32>
          affine.store %exp, %cb1[%i, %j] : memref<1x1x!ttcore.tile<32x32, f32>, #l1_>
          // Simple memref copy (memref.load → memref.store, no compute):
          %copy_val = memref.load %cb0[%c0, %c0] : memref<1x1x!ttcore.tile<32x32, f32>, #l1_>
          memref.store %copy_val, %cb1[%c0, %c0] : memref<1x1x!ttcore.tile<32x32, f32>, #l1_>
        }
      } {d2m.linalg_root, d2m.scheduled}
    }
    // CHECK: %[[DST:.*]] = d2m.acquire_dst() : memref<4x!ttcore.tile<32x32, f32>, #dst>
    // CHECK-NOT: d2m.scheduled
    // Compute path: CB -> DST load (in-place), compute in DST:
    // CHECK: affine.load %{{.*}} : memref<1x1x!ttcore.tile<32x32, f32>, #l1>
    // CHECK: affine.store %{{.*}}, %[[DST]]
    // CHECK: affine.load %[[DST]]
    // CHECK: "d2m.tile_exp"
    // CHECK: affine.store %{{.*}}, %[[DST]]
    // memref copy: memref.load → DST → memref.store:
    // CHECK: memref.load %{{.*}} : memref<1x1x!ttcore.tile<32x32, f32>, #l1>
    // CHECK: affine.store %{{.*}}, %[[DST]]
    // CHECK: affine.load %[[DST]]
    // CHECK: memref.store %{{.*}}, %{{.*}} : memref<1x1x!ttcore.tile<32x32, f32>, #l1>
    return
  }
}
