// RUN: ttmlir-opt --arith-to-d2m-tile-ops -o %t %s
// RUN: FileCheck %s --input-file=%t

// This test verifies you can use tile types w/ the arith dialect
// which requires its types to conform to the FloatTypeInterface

!tile = !ttcore.tile<32x32, f32>

func.func @addf(%arg0: !tile, %arg1: !tile) -> !tile {
  // CHECK: d2m.tile_add
  %0 = arith.addf %arg0, %arg1 : !tile
  return %0 : !tile
}

func.func @subf(%arg0: !tile, %arg1: !tile) -> !tile {
  // CHECK: d2m.tile_sub
  %0 = arith.subf %arg0, %arg1 : !tile
  return %0 : !tile
}

func.func @mulf(%arg0: !tile, %arg1: !tile) -> !tile {
  // CHECK: d2m.tile_mul
  %0 = arith.mulf %arg0, %arg1 : !tile
  return %0 : !tile
}

func.func @divf(%arg0: !tile, %arg1: !tile) -> !tile {
  // CHECK: d2m.tile_div
  %0 = arith.divf %arg0, %arg1 : !tile
  return %0 : !tile
}

func.func @negf(%arg0: !tile) -> !tile {
  // CHECK: d2m.tile_negative
  %0 = arith.negf %arg0 : !tile
  return %0 : !tile
}
