// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// Verify that the unscheduled pass does NOT insert d2m.acquire_dst for a
// d2m.linalg_root loop that HAS d2m.scheduled (i.e. a scheduled root).
// The test IR contains a pre-lowered affine loop with d2m.linalg_root
// and d2m.scheduled attributes. Running only the unscheduled pass should
// leave the IR untouched (no acquire_dst).
// RUN: ttmlir-opt --ttcore-register-device --d2m-insert-dst-register-access-unscheduled -o %t %s
// RUN: FileCheck %s --input-file=%t

#l1_ = #ttcore.memory_space<l1>

module {
  // CHECK-LABEL: func.func @eltwise_scheduled_root
  func.func @eltwise_scheduled_root(%arg0: memref<1x1x2x4x!ttcore.tile<32x32, f32>, #ttcore.shard<2048x2048, 1>, #l1_>,
                                    %arg1: memref<1x1x2x4x!ttcore.tile<32x32, f32>, #ttcore.shard<2048x2048, 1>, #l1_>) {
    d2m.generic {block_factors = [], grid = #ttcore.grid<1x1>,
                 indexing_maps = [], iterator_types = [],
                 threads = [#d2m.thread<unified>]}
        ins(%arg0 : memref<1x1x2x4x!ttcore.tile<32x32, f32>, #ttcore.shard<2048x2048, 1>, #l1_>)
        outs(%arg1 : memref<1x1x2x4x!ttcore.tile<32x32, f32>, #ttcore.shard<2048x2048, 1>, #l1_>) {
    ^unified0:
      %arg0_cb = d2m.get_cb(0) : !d2m.cb<memref<2x4x!ttcore.tile<32x32, f32>, #l1_>>
      %arg1_cb = d2m.get_cb(1) : !d2m.cb<memref<2x4x!ttcore.tile<32x32, f32>, #l1_>>
      %cb0 = d2m.wait %arg0_cb : !d2m.cb<memref<2x4x!ttcore.tile<32x32, f32>, #l1_>> -> memref<2x4x!ttcore.tile<32x32, f32>, #l1_>
      %cb1 = d2m.reserve %arg1_cb : !d2m.cb<memref<2x4x!ttcore.tile<32x32, f32>, #l1_>> -> memref<2x4x!ttcore.tile<32x32, f32>, #l1_>
      affine.for %i = 0 to 2 {
        affine.for %j = 0 to 4 {
          %0 = affine.load %cb0[%i, %j] : memref<2x4x!ttcore.tile<32x32, f32>, #l1_>
          %1 = "d2m.tile_exp"(%0) : (!ttcore.tile<32x32, f32>) -> !ttcore.tile<32x32, f32>
          affine.store %1, %cb1[%i, %j] : memref<2x4x!ttcore.tile<32x32, f32>, #l1_>
        }
      } {d2m.linalg_root, d2m.scheduled}
    }
    // The unscheduled pass should NOT have inserted an acquire_dst here,
    // because the linalg_root loop carries d2m.scheduled.
    // CHECK-NOT: d2m.acquire_dst
    return
  }
}
