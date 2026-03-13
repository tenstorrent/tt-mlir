// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// RUN: ttmlir-opt --ttcore-register-device --d2m-linalg-to-affine --d2m-insert-dst-register-access --canonicalize -o %t %s
// RUN: FileCheck %s --input-file=%t

// Tests verifying that f32 tile_add / tile_sub / tile_mul are classified as
// SFPU (commit 443f7bd0d), which yields a 4-tile DST buffer (half of the
// 8-tile SFPU buffer for bf16), and that DST slot indices are correctly
// assigned in the fused case where multiple loop bodies are linalg roots.

#l1_ = #ttcore.memory_space<l1>
#dst_ = #ttcore.memory_space<dst>

// ---------------------------------------------------------------------------
// Test: f32 tile_add (SFPU) fused with tile_recip in one linalg.generic body.
//
// After d2m-linalg-to-affine + d2m-insert-dst-register-access the body is
// split into three linalg_root loop nests that share one d2m.acquire_dst:
//
//   Linalg-root 1 (load phase):
//     in0 → DST[0]   (slot %arg3 + %arg4 + 0)
//     in1 → DST[1]   (slot %arg3 + %arg4 + 1)
//
//   Linalg-root 2 (compute phase, multiple ops fused):
//     tile_add(DST[0], DST[1]) → DST[2]   (slot %arg3 + %arg4 + 2)
//     tile_recip(DST[2]) → DST[2]          (in-place on same slot)
//
//   Linalg-root 3 (store phase):
//     DST[2] → L1
//
// Using 3 of the 4 available f32 SFPU DST slots confirms that the smaller
// DST capacity does not overflow (4 tile limit, 3 slots consumed).
// ---------------------------------------------------------------------------

// CHECK-LABEL: func.func @f32_fused_binary_then_unary
func.func @f32_fused_binary_then_unary(
    %in0: memref<1x1x1x1x!ttcore.tile<32x32, f32>, #ttcore.shard<4096x4096, 1>, #l1_>,
    %in1: memref<1x1x1x1x!ttcore.tile<32x32, f32>, #ttcore.shard<4096x4096, 1>, #l1_>,
    %out0: memref<1x1x1x1x!ttcore.tile<32x32, f32>, #ttcore.shard<4096x4096, 1>, #l1_>) {
  d2m.generic {block_factors = [], grid = #ttcore.grid<1x1>, indexing_maps = [],
               iterator_types = [], threads = [#d2m.thread<unified>]}
      ins(%in0, %in1 : memref<1x1x1x1x!ttcore.tile<32x32, f32>, #ttcore.shard<4096x4096, 1>, #l1_>,
                       memref<1x1x1x1x!ttcore.tile<32x32, f32>, #ttcore.shard<4096x4096, 1>, #l1_>)
      outs(%out0 : memref<1x1x1x1x!ttcore.tile<32x32, f32>, #ttcore.shard<4096x4096, 1>, #l1_>) {
  ^unified0:
    %arg0_cb = d2m.get_cb(0) : !d2m.cb<memref<1x1x!ttcore.tile<32x32, f32>, #l1_>>
    %arg1_cb = d2m.get_cb(1) : !d2m.cb<memref<1x1x!ttcore.tile<32x32, f32>, #l1_>>
    %arg2_cb = d2m.get_cb(2) : !d2m.cb<memref<1x1x!ttcore.tile<32x32, f32>, #l1_>>
    %cb0 = d2m.wait %arg0_cb : !d2m.cb<memref<1x1x!ttcore.tile<32x32, f32>, #l1_>>
        -> memref<1x1x!ttcore.tile<32x32, f32>, #l1_>
    %cb1 = d2m.wait %arg1_cb : !d2m.cb<memref<1x1x!ttcore.tile<32x32, f32>, #l1_>>
        -> memref<1x1x!ttcore.tile<32x32, f32>, #l1_>
    %cb2 = d2m.reserve %arg2_cb : !d2m.cb<memref<1x1x!ttcore.tile<32x32, f32>, #l1_>>
        -> memref<1x1x!ttcore.tile<32x32, f32>, #l1_>
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    scf.for %i = %c0 to %c1 step %c1 {
      scf.for %j = %c0 to %c1 step %c1 {
        %sv0 = memref.subview %cb0[%i, %j] [1, 1] [1, 1]
            : memref<1x1x!ttcore.tile<32x32, f32>, #l1_>
            to memref<1x1x!ttcore.tile<32x32, f32>, strided<[1, 1], offset: ?>, #l1_>
        %sv1 = memref.subview %cb1[%i, %j] [1, 1] [1, 1]
            : memref<1x1x!ttcore.tile<32x32, f32>, #l1_>
            to memref<1x1x!ttcore.tile<32x32, f32>, strided<[1, 1], offset: ?>, #l1_>
        %sv2 = memref.subview %cb2[%i, %j] [1, 1] [1, 1]
            : memref<1x1x!ttcore.tile<32x32, f32>, #l1_>
            to memref<1x1x!ttcore.tile<32x32, f32>, strided<[1, 1], offset: ?>, #l1_>
        linalg.generic {
            indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                             affine_map<(d0, d1) -> (d0, d1)>,
                             affine_map<(d0, d1) -> (d0, d1)>],
            iterator_types = ["parallel", "parallel"]}
            ins(%sv0, %sv1 : memref<1x1x!ttcore.tile<32x32, f32>, strided<[1, 1], offset: ?>, #l1_>,
                             memref<1x1x!ttcore.tile<32x32, f32>, strided<[1, 1], offset: ?>, #l1_>)
            outs(%sv2 : memref<1x1x!ttcore.tile<32x32, f32>, strided<[1, 1], offset: ?>, #l1_>) {
        ^bb0(%a: !ttcore.tile<32x32, f32>, %b: !ttcore.tile<32x32, f32>,
             %c: !ttcore.tile<32x32, f32>):
          // f32 tile_add → SFPU → 4-tile DST.
          %add = "d2m.tile_add"(%a, %b)
              : (!ttcore.tile<32x32, f32>, !ttcore.tile<32x32, f32>) -> !ttcore.tile<32x32, f32>
          // tile_recip fused in the same linalg body.
          %recip = "d2m.tile_recip"(%add) : (!ttcore.tile<32x32, f32>) -> !ttcore.tile<32x32, f32>
          linalg.yield %recip : !ttcore.tile<32x32, f32>
        }
      }
    }
  }
  return
}
// Verify that f32 SFPU yields a 4-tile DST buffer (not 8).
// CHECK: %[[DST:.*]] = d2m.acquire_dst() : memref<4x!ttcore.tile<32x32, f32>, #dst>
//
// Linalg-root 1: in0 → DST slot 0, in1 → DST slot 1.
// CHECK: affine.store {{.*}}, %[[DST]][{{.*}}]
// CHECK: affine.store {{.*}}, %[[DST]][{{.*}} + 1]
// CHECK: } {d2m.dst_access_inserted, d2m.linalg_root}
//
// Linalg-root 2: tile_add reads slots 0,1 and writes slot 2;
//                tile_recip reads and overwrites slot 2 in-place.
// CHECK: %[[IN0_DST:.*]] = affine.load %[[DST]][{{.*}}]
// CHECK: %[[IN1_DST:.*]] = affine.load %[[DST]][{{.*}} + 1]
// CHECK: %[[ADD:.*]] = "d2m.tile_add"(%[[IN0_DST]], %[[IN1_DST]])
// CHECK: affine.store %[[ADD]], %[[DST]][{{.*}} + 2]
// CHECK: %[[ADD_DST:.*]] = affine.load %[[DST]][{{.*}} + 2]
// CHECK: %[[RECIP:.*]] = "d2m.tile_recip"(%[[ADD_DST]])
// CHECK: affine.store %[[RECIP]], %[[DST]][{{.*}} + 2]
// CHECK: } {d2m.dst_access_inserted, d2m.linalg_root}
//
// Linalg-root 3: final result loaded from DST slot 2 and stored to L1.
// CHECK: %[[RESULT:.*]] = affine.load %[[DST]][{{.*}} + 2]
// CHECK: affine.store %[[RESULT]], {{.*}} : memref<{{.*}}, #l1>
// CHECK: } {d2m.dst_access_inserted, d2m.linalg_root}
