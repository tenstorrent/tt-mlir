// RUN: ttmlir-opt --arith-to-d2m-tile-ops -o %t %s
// RUN: FileCheck %s --input-file=%t

// This test verifies you can use tile types w/ the arith dialect
// which requires its types to conform to the FloatTypeInterface

!tile = !ttcore.tile<32x32, f32>
!tile_i1 = !ttcore.tile<32x32, i1>

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

func.func @cmpf_oeq(%arg0: !tile, %arg1: !tile) -> !tile_i1 {
  // CHECK-LABEL: func.func @cmpf_oeq
  // CHECK: %[[SUB:.*]] = "d2m.tile_sub"(%arg0, %arg1)
  // CHECK: %[[CMP:.*]] = "d2m.tile_eqz"(%[[SUB]])
  // CHECK: return %[[CMP]]
  // arith.cmpf oeq (ordered and equal): lhs == rhs
  %0 = arith.cmpf oeq, %arg0, %arg1 : !tile
  return %0 : !tile_i1
}

func.func @cmpf_one(%arg0: !tile, %arg1: !tile) -> !tile_i1 {
  // CHECK-LABEL: func.func @cmpf_one
  // CHECK: %[[SUB:.*]] = "d2m.tile_sub"(%arg0, %arg1)
  // CHECK: %[[CMP:.*]] = "d2m.tile_nez"(%[[SUB]])
  // CHECK: return %[[CMP]]
  // arith.cmpf one (ordered and not equal): lhs != rhs
  %0 = arith.cmpf one, %arg0, %arg1 : !tile
  return %0 : !tile_i1
}

func.func @cmpf_ogt(%arg0: !tile, %arg1: !tile) -> !tile_i1 {
  // CHECK-LABEL: func.func @cmpf_ogt
  // CHECK: %[[SUB:.*]] = "d2m.tile_sub"(%arg0, %arg1)
  // CHECK: %[[CMP:.*]] = "d2m.tile_gtz"(%[[SUB]])
  // CHECK: return %[[CMP]]
  // arith.cmpf ogt (ordered and greater than): lhs > rhs
  %0 = arith.cmpf ogt, %arg0, %arg1 : !tile
  return %0 : !tile_i1
}

func.func @cmpf_oge(%arg0: !tile, %arg1: !tile) -> !tile_i1 {
  // CHECK-LABEL: func.func @cmpf_oge
  // CHECK: %[[SUB:.*]] = "d2m.tile_sub"(%arg0, %arg1)
  // CHECK: %[[CMP:.*]] = "d2m.tile_gez"(%[[SUB]])
  // CHECK: return %[[CMP]]
  // arith.cmpf oge (ordered and greater than or equal): lhs >= rhs
  %0 = arith.cmpf oge, %arg0, %arg1 : !tile
  return %0 : !tile_i1
}

func.func @cmpf_olt(%arg0: !tile, %arg1: !tile) -> !tile_i1 {
  // CHECK-LABEL: func.func @cmpf_olt
  // CHECK: %[[SUB:.*]] = "d2m.tile_sub"(%arg0, %arg1)
  // CHECK: %[[CMP:.*]] = "d2m.tile_ltz"(%[[SUB]])
  // CHECK: return %[[CMP]]
  // arith.cmpf olt (ordered and less than): lhs < rhs
  %0 = arith.cmpf olt, %arg0, %arg1 : !tile
  return %0 : !tile_i1
}

func.func @cmpf_ole(%arg0: !tile, %arg1: !tile) -> !tile_i1 {
  // CHECK-LABEL: func.func @cmpf_ole
  // CHECK: %[[SUB:.*]] = "d2m.tile_sub"(%arg0, %arg1)
  // CHECK: %[[CMP:.*]] = "d2m.tile_lez"(%[[SUB]])
  // CHECK: return %[[CMP]]
  // arith.cmpf ole (ordered and less than or equal): lhs <= rhs
  %0 = arith.cmpf ole, %arg0, %arg1 : !tile
  return %0 : !tile_i1
}
