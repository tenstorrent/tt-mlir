// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// RUN: ttmlir-opt --ttcore-register-device --d2m-decompose-arange -o %t %s
// RUN: FileCheck %s --input-file=%t

// Verify that d2m-decompose-arange expands arange_block into tile loops.
// All tests operate on pre-bufferized memref IR (as seen after ttcore-one-shot-bufferize).

#l1_ = #ttcore.memory_space<l1>
#shard_1x1 = #ttcore.shard<4096x4096, 1>
#shard_1x4 = #ttcore.shard<4096x16384, 1>
#shard_4x1 = #ttcore.shard<16384x4096, 1>

// ---- Row-major single tile (1x1 grid, 1x1 shard) ----
// Verifies fill_arange_tile, two nested scf.for loops, and no tile_transpose.

// CHECK-LABEL: func.func @row_major_single_tile
func.func @row_major_single_tile(
    %scratch: memref<1x1x!ttcore.tile<32x32, f32>>,
    %out: memref<1x1x!ttcore.tile<32x32, f32>>) {
  d2m.generic {block_factors = [], grid = #ttcore.grid<1x1>,
               indexing_maps = [], iterator_types = [],
               threads = [#d2m.thread<unified>]}
      ins(%scratch : memref<1x1x!ttcore.tile<32x32, f32>>)
      outs(%out : memref<1x1x!ttcore.tile<32x32, f32>>) {
    // CHECK-NOT: d2m.arange_block
    // CHECK: d2m.fill_arange_tile
    // CHECK: scf.for
    // CHECK:   scf.for
    // CHECK-NOT: d2m.tile_transpose
    // CHECK: d2m.tile_add
    %0 = "d2m.arange_block"(%scratch, %out)
        <{colMajor = false, num_elements = 32 : i64, start = 0 : i64, step = 1 : i64}>
        : (memref<1x1x!ttcore.tile<32x32, f32>>, memref<1x1x!ttcore.tile<32x32, f32>>)
        -> memref<1x1x!ttcore.tile<32x32, f32>>
  }
  return
}

// ---- Row-major multi-tile (1x1 grid, 1x4 shard) ----
// Verifies the inner loop runs numTileCols=4 times and no tile_transpose is emitted.

// CHECK-LABEL: func.func @row_major_multi_tile
func.func @row_major_multi_tile(
    %scratch: memref<1x1x!ttcore.tile<32x32, f32>>,
    %out: memref<1x4x!ttcore.tile<32x32, f32>>) {
  d2m.generic {block_factors = [], grid = #ttcore.grid<1x1>,
               indexing_maps = [], iterator_types = [],
               threads = [#d2m.thread<unified>]}
      ins(%scratch : memref<1x1x!ttcore.tile<32x32, f32>>)
      outs(%out : memref<1x4x!ttcore.tile<32x32, f32>>) {
    // CHECK-NOT: d2m.arange_block
    // CHECK: d2m.fill_arange_tile
    // CHECK: scf.for
    // CHECK:   scf.for
    // CHECK-NOT: d2m.tile_transpose
    // CHECK: d2m.tile_add
    %0 = "d2m.arange_block"(%scratch, %out)
        <{colMajor = false, num_elements = 128 : i64, start = 0 : i64, step = 1 : i64}>
        : (memref<1x1x!ttcore.tile<32x32, f32>>, memref<1x4x!ttcore.tile<32x32, f32>>)
        -> memref<1x4x!ttcore.tile<32x32, f32>>
  }
  return
}

// ---- Col-major (1x1 grid, 4x1 shard) ----
// Verifies that colMajor=true emits tile_transpose before arithmetic.

// CHECK-LABEL: func.func @col_major
func.func @col_major(
    %scratch: memref<1x1x!ttcore.tile<32x32, f32>>,
    %out: memref<4x1x!ttcore.tile<32x32, f32>>) {
  d2m.generic {block_factors = [], grid = #ttcore.grid<1x1>,
               indexing_maps = [], iterator_types = [],
               threads = [#d2m.thread<unified>]}
      ins(%scratch : memref<1x1x!ttcore.tile<32x32, f32>>)
      outs(%out : memref<4x1x!ttcore.tile<32x32, f32>>) {
    // CHECK-NOT: d2m.arange_block
    // CHECK: d2m.fill_arange_tile
    // CHECK: scf.for
    // CHECK:   scf.for
    // CHECK: d2m.tile_transpose
    // CHECK: d2m.tile_add
    %0 = "d2m.arange_block"(%scratch, %out)
        <{colMajor = true, num_elements = 128 : i64, start = 0 : i64, step = 1 : i64}>
        : (memref<1x1x!ttcore.tile<32x32, f32>>, memref<4x1x!ttcore.tile<32x32, f32>>)
        -> memref<4x1x!ttcore.tile<32x32, f32>>
  }
  return
}

// ---- Blocking loop IV folding ----
// Verifies that a col-blocking IV (d2m.blocking_loop = 1) is folded into
// globalTileCol as colBlockIV * shardTileCols.

// CHECK-LABEL: func.func @blocking_loop_col_iv
func.func @blocking_loop_col_iv(
    %scratch: memref<1x1x!ttcore.tile<32x32, f32>>,
    %out: memref<1x4x!ttcore.tile<32x32, f32>>) {
  d2m.generic {block_factors = [], grid = #ttcore.grid<1x1>,
               indexing_maps = [], iterator_types = [],
               threads = [#d2m.thread<unified>]}
      ins(%scratch : memref<1x1x!ttcore.tile<32x32, f32>>)
      outs(%out : memref<1x4x!ttcore.tile<32x32, f32>>) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index
    // CHECK: scf.for
    // CHECK:   d2m.fill_arange_tile
    // CHECK:   scf.for
    // CHECK:     scf.for
    // The inner loop body emits 4 muli: coreX*shardTileCols,
    // colBlockIV*shardTileCols, and the three-multiply row-major rowContrib
    // (globalTileRow * totalTileCols * 32 * 32), plus globalTileCol * 32.
    // CHECK:       arith.muli
    // CHECK:       arith.muli
    // CHECK:       arith.muli
    // CHECK:       arith.muli
    // CHECK:       arith.addi
    scf.for %blk = %c0 to %c2 step %c1 {
      %0 = "d2m.arange_block"(%scratch, %out)
          <{colMajor = false, num_elements = 128 : i64, start = 0 : i64, step = 1 : i64}>
          : (memref<1x1x!ttcore.tile<32x32, f32>>, memref<1x4x!ttcore.tile<32x32, f32>>)
          -> memref<1x4x!ttcore.tile<32x32, f32>>
    } {d2m.blocking_loop = 1 : i64}
  }
  return
}
