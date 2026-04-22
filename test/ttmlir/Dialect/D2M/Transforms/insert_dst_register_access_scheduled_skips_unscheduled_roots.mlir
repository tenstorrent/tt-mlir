// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// Verify that the scheduled pass does NOT insert d2m.acquire_dst for a
// d2m.linalg_root loop that lacks d2m.scheduled (i.e. an unscheduled root).
// RUN: ttmlir-opt --ttcore-register-device --d2m-linalg-to-affine --d2m-insert-dst-register-access-scheduled --canonicalize -o %t %s
// RUN: FileCheck %s --input-file=%t

#l1_ = #ttcore.memory_space<l1>

module {
  // CHECK-LABEL: func.func @matmul_unscheduled_root
  func.func @matmul_unscheduled_root(
      %in0: memref<1x1x3x3x!ttcore.tile<32x32, f32>, #ttcore.shard<4096x4096, 1>, #l1_>,
      %in1: memref<1x1x3x2x!ttcore.tile<32x32, f32>, #ttcore.shard<4096x4096, 1>, #l1_>,
      %out0: memref<1x1x3x2x!ttcore.tile<32x32, f32>, #ttcore.shard<4096x4096, 1>, #l1_>) {
    d2m.generic {block_factors = [], grid = #ttcore.grid<1x1>, indexing_maps = [], iterator_types = [], threads = [#d2m.thread<unified>]}
        ins(%in0, %in1 : memref<1x1x3x3x!ttcore.tile<32x32, f32>, #ttcore.shard<4096x4096, 1>, #l1_>, memref<1x1x3x2x!ttcore.tile<32x32, f32>, #ttcore.shard<4096x4096, 1>, #l1_>)
        outs(%out0 : memref<1x1x3x2x!ttcore.tile<32x32, f32>, #ttcore.shard<4096x4096, 1>, #l1_>) {
    ^unified0:
      %arg0_cb = d2m.get_cb(0) : !d2m.cb<memref<3x3x!ttcore.tile<32x32, f32>, #l1_>>
      %arg1_cb = d2m.get_cb(1) : !d2m.cb<memref<3x2x!ttcore.tile<32x32, f32>, #l1_>>
      %arg2_cb = d2m.get_cb(2) : !d2m.cb<memref<3x2x!ttcore.tile<32x32, f32>, #l1_>>
      %cb0 = d2m.wait %arg0_cb : !d2m.cb<memref<3x3x!ttcore.tile<32x32, f32>, #l1_>> -> memref<3x3x!ttcore.tile<32x32, f32>, #l1_>
      %cb1 = d2m.wait %arg1_cb : !d2m.cb<memref<3x2x!ttcore.tile<32x32, f32>, #l1_>> -> memref<3x2x!ttcore.tile<32x32, f32>, #l1_>
      %cb2 = d2m.reserve %arg2_cb : !d2m.cb<memref<3x2x!ttcore.tile<32x32, f32>, #l1_>> -> memref<3x2x!ttcore.tile<32x32, f32>, #l1_>
      linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d2)>,
                                       affine_map<(d0, d1, d2) -> (d2, d1)>,
                                       affine_map<(d0, d1, d2) -> (d0, d1)>],
                     iterator_types = ["parallel", "parallel", "reduction"]}
          ins(%cb0, %cb1 : memref<3x3x!ttcore.tile<32x32, f32>, #l1_>, memref<3x2x!ttcore.tile<32x32, f32>, #l1_>)
          outs(%cb2 : memref<3x2x!ttcore.tile<32x32, f32>, #l1_>) {
      ^bb0(%arg0: !ttcore.tile<32x32, f32>, %arg1: !ttcore.tile<32x32, f32>, %arg2: !ttcore.tile<32x32, f32>):
        %0 = "d2m.tile_matmul"(%arg0, %arg1, %arg2) : (!ttcore.tile<32x32, f32>, !ttcore.tile<32x32, f32>, !ttcore.tile<32x32, f32>) -> !ttcore.tile<32x32, f32>
        linalg.yield %0 : !ttcore.tile<32x32, f32>
      }
    }
    // The scheduled pass should NOT have inserted an acquire_dst here,
    // because the linalg_root loop does not carry d2m.scheduled.
    // CHECK-NOT: d2m.acquire_dst
    return
  }
}
