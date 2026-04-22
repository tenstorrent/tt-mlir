// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// Tests for L1 accumulation guard insertion in the unscheduled pass.
// When enable-l1-acc=true and the root loop contains a tile_matmul,
// the pass inserts a d2m.set_l1_accumulate guarded by an scf.if that
// checks whether the outer tiling loop has reached its second iteration.
//
// The test uses hand-crafted IR with a matmul (4x4 * 4x2 -> 4x2) tiled
// into 2x2 blocks via scf.for loops, giving the outer loop 2 iterations
// so the L1 acc guard survives canonicalization.
//
// Exercises:
//   - insertPackerL1AccGuard (L1 accumulation insertion)
//   - collectAncestorLoopIVs (finding parent scf.for IV)
//   - enableL1Acc=true skips CB->DST load copies in the load loop
// RUN: ttmlir-opt --ttcore-register-device --d2m-insert-dst-register-access-unscheduled="enable-l1-acc=true" --canonicalize -o %t %s
// RUN: FileCheck %s --input-file=%t

#l1_ = #ttcore.memory_space<l1>
#dst_ = #ttcore.memory_space<dst>

module {
  // CHECK-LABEL: func.func @matmul_l1_acc
  func.func @matmul_l1_acc(
      %in0: memref<1x1x4x4x!ttcore.tile<32x32, f16>, #ttcore.shard<8192x2048, 1>, #l1_>,
      %in1: memref<1x1x4x2x!ttcore.tile<32x32, f16>, #ttcore.shard<4096x2048, 1>, #l1_>,
      %out0: memref<1x1x4x2x!ttcore.tile<32x32, f16>, #ttcore.shard<4096x2048, 1>, #l1_>) {
    d2m.generic {block_factors = [], grid = #ttcore.grid<1x1>, indexing_maps = [], iterator_types = [], threads = [#d2m.thread<unified>]}
        ins(%in0, %in1 : memref<1x1x4x4x!ttcore.tile<32x32, f16>, #ttcore.shard<8192x2048, 1>, #l1_>, memref<1x1x4x2x!ttcore.tile<32x32, f16>, #ttcore.shard<4096x2048, 1>, #l1_>)
        outs(%out0 : memref<1x1x4x2x!ttcore.tile<32x32, f16>, #ttcore.shard<4096x2048, 1>, #l1_>) {
    ^unified0:
      %cb0_raw = d2m.get_cb(0) : !d2m.cb<memref<4x4x!ttcore.tile<32x32, f16>, #l1_>>
      %cb1_raw = d2m.get_cb(1) : !d2m.cb<memref<4x2x!ttcore.tile<32x32, f16>, #l1_>>
      %cb2_raw = d2m.get_cb(2) : !d2m.cb<memref<4x2x!ttcore.tile<32x32, f16>, #l1_>>
      %cb0 = d2m.wait %cb0_raw : !d2m.cb<memref<4x4x!ttcore.tile<32x32, f16>, #l1_>> -> memref<4x4x!ttcore.tile<32x32, f16>, #l1_>
      %cb1 = d2m.wait %cb1_raw : !d2m.cb<memref<4x2x!ttcore.tile<32x32, f16>, #l1_>> -> memref<4x2x!ttcore.tile<32x32, f16>, #l1_>
      %cb2 = d2m.reserve %cb2_raw : !d2m.cb<memref<4x2x!ttcore.tile<32x32, f16>, #l1_>> -> memref<4x2x!ttcore.tile<32x32, f16>, #l1_>
      %c0 = arith.constant 0 : index
      %c2 = arith.constant 2 : index
      %c4 = arith.constant 4 : index
      // Outer tiling loop with 2 iterations (0, 2):
      scf.for %tile_i = %c0 to %c4 step %c2 {
        scf.for %tile_j = %c0 to %c2 step %c2 {
          %sv0 = memref.subview %cb0[%tile_i, %c0] [2, 4] [1, 1] : memref<4x4x!ttcore.tile<32x32, f16>, #l1_> to memref<2x4x!ttcore.tile<32x32, f16>, strided<[4, 1], offset: ?>, #l1_>
          %sv1 = memref.subview %cb1[%c0, %tile_j] [4, 2] [1, 1] : memref<4x2x!ttcore.tile<32x32, f16>, #l1_> to memref<4x2x!ttcore.tile<32x32, f16>, strided<[2, 1], offset: ?>, #l1_>
          %sv2 = memref.subview %cb2[%tile_i, %tile_j] [2, 2] [1, 1] : memref<4x2x!ttcore.tile<32x32, f16>, #l1_> to memref<2x2x!ttcore.tile<32x32, f16>, strided<[2, 1], offset: ?>, #l1_>
          affine.for %i = 0 to 2 {
            affine.for %j = 0 to 2 {
              affine.for %k = 0 to 4 {
                %a = affine.load %sv0[%i, %k] : memref<2x4x!ttcore.tile<32x32, f16>, strided<[4, 1], offset: ?>, #l1_>
                %b = affine.load %sv1[%k, %j] : memref<4x2x!ttcore.tile<32x32, f16>, strided<[2, 1], offset: ?>, #l1_>
                %c = affine.load %sv2[%i, %j] : memref<2x2x!ttcore.tile<32x32, f16>, strided<[2, 1], offset: ?>, #l1_>
                %r = "d2m.tile_matmul"(%a, %b, %c) : (!ttcore.tile<32x32, f16>, !ttcore.tile<32x32, f16>, !ttcore.tile<32x32, f16>) -> !ttcore.tile<32x32, f16>
                affine.store %r, %sv2[%i, %j] : memref<2x2x!ttcore.tile<32x32, f16>, strided<[2, 1], offset: ?>, #l1_>
              }
            }
          } {d2m.linalg_root}
        }
      }
    }
    return
  }
  // acquire_dst placed inside the scf.for:
  // CHECK: scf.for %[[TILE_I:.*]] =
  // CHECK:   %[[DST:.*]] = d2m.acquire_dst() : memref<8x!ttcore.tile<32x32, f16>, #dst>
  // L1 accumulation guard: checks if %tile_i == second_iteration_value:
  // CHECK:   %[[CMP:.*]] = arith.cmpi eq, %[[TILE_I]]
  // CHECK:   scf.if %[[CMP]]
  // CHECK:     d2m.set_l1_accumulate
  // Compute loop with matmul:
  // CHECK:   affine.for
  // CHECK:     affine.for
  // CHECK:       affine.for
  // CHECK:         "d2m.tile_matmul"
}
