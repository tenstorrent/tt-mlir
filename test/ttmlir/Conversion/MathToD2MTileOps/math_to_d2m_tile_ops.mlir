// RUN: ttmlir-opt --math-to-d2m-tile-ops -o %t %s
// RUN: FileCheck %s --input-file=%t

!tile = !ttcore.tile<32x32, f32>

func.func @absf(%arg0: !tile) -> !tile {
  // CHECK: d2m.tile_abs
  %0 = math.absf %arg0 : !tile
  return %0 : !tile
}
