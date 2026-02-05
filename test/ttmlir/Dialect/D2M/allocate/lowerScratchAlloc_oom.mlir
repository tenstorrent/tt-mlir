// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// RUN: not ttmlir-opt --ttcore-register-device --d2m-lower-scratch-allocate %s 2>&1 | FileCheck %s

// Test that exceeding scratch buffer capacity produces a diagnostic.
// Scratch buffer has 8 tiles, but we request 5 + 5 = 10 tiles.

#l1 = #ttcore.memory_space<l1>
#parallel = #ttcore.iterator_type<parallel>

module {

func.func @scratch_overflow() {
  %in = memref.alloc() : memref<1x1x4x4x!ttcore.tile<32x32, f32>, #ttcore.shard<16384x4096, 1>, #l1>
  %scratch_buf = memref.alloc() : memref<1x1x1x8x!ttcore.tile<32x32, f32>, #ttcore.shard<32768x4096, 1>, #l1>
  %out = memref.alloc() : memref<1x1x4x4x!ttcore.tile<32x32, f32>, #ttcore.shard<16384x4096, 1>, #l1>
  // CHECK: error: 'd2m.generic' op total scratch allocations (10 elements) exceed scratch buffer capacity (8 elements)
  d2m.generic {
    block_factors = [1, 1], grid = #ttcore.grid<1x1>,
    indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (0, 0)>, affine_map<(d0, d1) -> (d0, d1)>],
    iterator_types = [#parallel, #parallel],
    scratch_inputs = array<i64: 1>,
    threads = [#d2m.thread<unified>]
  }
  ins(%in, %scratch_buf : memref<1x1x4x4x!ttcore.tile<32x32, f32>, #ttcore.shard<16384x4096, 1>, #l1>, memref<1x1x1x8x!ttcore.tile<32x32, f32>, #ttcore.shard<32768x4096, 1>, #l1>)
  outs(%out : memref<1x1x4x4x!ttcore.tile<32x32, f32>, #ttcore.shard<16384x4096, 1>, #l1>) {
  ^bb0(%cb0: !d2m.cb<memref<4x4x!ttcore.tile<32x32, f32>, #l1>>, %cb1: !d2m.cb<memref<1x8x!ttcore.tile<32x32, f32>, #l1>>, %cb2: !d2m.cb<memref<4x4x!ttcore.tile<32x32, f32>, #l1>>):
    %s0 = d2m.scratch_allocate {slot = 0 : i64} : memref<5x!ttcore.tile<32x32, f32>, #l1>
    %s1 = d2m.scratch_allocate {slot = 1 : i64} : memref<5x!ttcore.tile<32x32, f32>, #l1>
  }
  return
}

}
