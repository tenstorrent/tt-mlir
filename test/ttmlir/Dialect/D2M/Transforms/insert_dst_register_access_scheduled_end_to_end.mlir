// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// End-to-end integration test: a d2m.generic with 3+ inputs goes through:
//   1. --d2m-linalg-to-affine  (converts linalg.generic to affine loops, adds d2m.linalg_root)
//   2. --d2m-op-scheduler       (reorders ops for minimal DST usage, adds d2m.scheduled)
//   3. --d2m-insert-dst-register-access-unscheduled (skips the scheduled root)
//   4. --d2m-insert-dst-register-access-scheduled   (processes the scheduled root, consumes d2m.scheduled)
//
// This verifies that the full pipeline works when OpScheduler is involved.
// RUN: ttmlir-opt --ttcore-register-device --d2m-linalg-to-affine --d2m-op-scheduler --d2m-insert-dst-register-access-unscheduled --d2m-insert-dst-register-access-scheduled --canonicalize -o %t %s
// RUN: FileCheck %s --input-file=%t

#l1_ = #ttcore.memory_space<l1>
#dst_ = #ttcore.memory_space<dst>

module {
  // -----------------------------------------------------------------------
  // 3-input fused eltwise: add(in0, in1) -> div(result, in2) -> out
  // OpScheduler triggers on 3+ inputs, adds d2m.scheduled.
  // The scheduled pass inserts DST accesses and consumes d2m.scheduled.
  // -----------------------------------------------------------------------
  // CHECK-LABEL: func.func @e2e_fused_3_inputs
  func.func @e2e_fused_3_inputs(
      %in0: memref<1x1x1x1x!ttcore.tile<32x32, f32>, #ttcore.shard<4096x4096, 1>, #l1_>,
      %in1: memref<1x1x1x1x!ttcore.tile<32x32, f32>, #ttcore.shard<4096x4096, 1>, #l1_>,
      %in2: memref<1x1x1x1x!ttcore.tile<32x32, f32>, #ttcore.shard<4096x4096, 1>, #l1_>,
      %out0: memref<1x1x1x1x!ttcore.tile<32x32, f32>, #ttcore.shard<4096x4096, 1>, #l1_>) {
    d2m.generic {block_factors = [], grid = #ttcore.grid<1x1>, indexing_maps = [], iterator_types = [], threads = [#d2m.thread<unified>]}
        ins(%in0, %in1, %in2 : memref<1x1x1x1x!ttcore.tile<32x32, f32>, #ttcore.shard<4096x4096, 1>, #l1_>, memref<1x1x1x1x!ttcore.tile<32x32, f32>, #ttcore.shard<4096x4096, 1>, #l1_>, memref<1x1x1x1x!ttcore.tile<32x32, f32>, #ttcore.shard<4096x4096, 1>, #l1_>)
        outs(%out0 : memref<1x1x1x1x!ttcore.tile<32x32, f32>, #ttcore.shard<4096x4096, 1>, #l1_>) {
    ^unified0:
      %cb0_raw = d2m.get_cb(0) : !d2m.cb<memref<1x1x!ttcore.tile<32x32, f32>, #l1_>>
      %cb1_raw = d2m.get_cb(1) : !d2m.cb<memref<1x1x!ttcore.tile<32x32, f32>, #l1_>>
      %cb2_raw = d2m.get_cb(2) : !d2m.cb<memref<1x1x!ttcore.tile<32x32, f32>, #l1_>>
      %cb3_raw = d2m.get_cb(3) : !d2m.cb<memref<1x1x!ttcore.tile<32x32, f32>, #l1_>>
      %cb0 = d2m.wait %cb0_raw : !d2m.cb<memref<1x1x!ttcore.tile<32x32, f32>, #l1_>> -> memref<1x1x!ttcore.tile<32x32, f32>, #l1_>
      %cb1 = d2m.wait %cb1_raw : !d2m.cb<memref<1x1x!ttcore.tile<32x32, f32>, #l1_>> -> memref<1x1x!ttcore.tile<32x32, f32>, #l1_>
      %cb2 = d2m.wait %cb2_raw : !d2m.cb<memref<1x1x!ttcore.tile<32x32, f32>, #l1_>> -> memref<1x1x!ttcore.tile<32x32, f32>, #l1_>
      %cb3 = d2m.reserve %cb3_raw : !d2m.cb<memref<1x1x!ttcore.tile<32x32, f32>, #l1_>> -> memref<1x1x!ttcore.tile<32x32, f32>, #l1_>
      %c0 = arith.constant 0 : index
      %sv0 = memref.subview %cb0[%c0, %c0] [1, 1] [1, 1] : memref<1x1x!ttcore.tile<32x32, f32>, #l1_> to memref<1x1x!ttcore.tile<32x32, f32>, strided<[1, 1], offset: ?>, #l1_>
      %sv1 = memref.subview %cb1[%c0, %c0] [1, 1] [1, 1] : memref<1x1x!ttcore.tile<32x32, f32>, #l1_> to memref<1x1x!ttcore.tile<32x32, f32>, strided<[1, 1], offset: ?>, #l1_>
      %sv2 = memref.subview %cb2[%c0, %c0] [1, 1] [1, 1] : memref<1x1x!ttcore.tile<32x32, f32>, #l1_> to memref<1x1x!ttcore.tile<32x32, f32>, strided<[1, 1], offset: ?>, #l1_>
      %sv3 = memref.subview %cb3[%c0, %c0] [1, 1] [1, 1] : memref<1x1x!ttcore.tile<32x32, f32>, #l1_> to memref<1x1x!ttcore.tile<32x32, f32>, strided<[1, 1], offset: ?>, #l1_>
      linalg.generic {
        indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                         affine_map<(d0, d1) -> (d0, d1)>,
                         affine_map<(d0, d1) -> (d0, d1)>,
                         affine_map<(d0, d1) -> (d0, d1)>],
        iterator_types = ["parallel", "parallel"]
      } ins(%sv0, %sv1, %sv2 : memref<1x1x!ttcore.tile<32x32, f32>, strided<[1, 1], offset: ?>, #l1_>,
                                memref<1x1x!ttcore.tile<32x32, f32>, strided<[1, 1], offset: ?>, #l1_>,
                                memref<1x1x!ttcore.tile<32x32, f32>, strided<[1, 1], offset: ?>, #l1_>)
        outs(%sv3 : memref<1x1x!ttcore.tile<32x32, f32>, strided<[1, 1], offset: ?>, #l1_>) {
      ^bb0(%a: !ttcore.tile<32x32, f32>, %b: !ttcore.tile<32x32, f32>, %c: !ttcore.tile<32x32, f32>, %out: !ttcore.tile<32x32, f32>):
        %add = "d2m.tile_add"(%a, %b) : (!ttcore.tile<32x32, f32>, !ttcore.tile<32x32, f32>) -> !ttcore.tile<32x32, f32>
        %div = "d2m.tile_div"(%add, %c) : (!ttcore.tile<32x32, f32>, !ttcore.tile<32x32, f32>) -> !ttcore.tile<32x32, f32>
        linalg.yield %div : !ttcore.tile<32x32, f32>
      }
    }
    return
  }

  // d2m.acquire_dst is inserted by the scheduled pass:
  // CHECK: d2m.acquire_dst() : memref<4x!ttcore.tile<32x32, f32>, #dst>
  // d2m.scheduled is consumed (should not appear in output):
  // CHECK-NOT: d2m.scheduled
  // Compute ops execute with DST intermediates:
  // CHECK: "d2m.tile_add"
  // CHECK: "d2m.tile_div"
  // A separate store loop exists for DST -> CB:
  // CHECK: d2m.dst_access_inserted
}
