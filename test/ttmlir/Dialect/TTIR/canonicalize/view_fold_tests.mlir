// RUN: ttmlir-opt -canonicalize %s | FileCheck %s

#l1 = #ttcore.memory_space<l1>

func.func @trivial(%arg0: memref<1x2x2x2x!ttcore.tile<32x32, f32>, #ttcore.shard<8192x4096>, #l1>) -> memref<1x2x2x2x!ttcore.tile<32x32, f32>, #ttcore.view<map(4)>, #l1> {
  // CHECK: %view = ttir.view_layout %arg0
  // CHECK-NEXT: return %view
  %0 = ttir.view_layout %arg0 : memref<1x2x2x2x!ttcore.tile<32x32, f32>, #ttcore.shard<8192x4096>, #l1> -> memref<1x2x2x2x!ttcore.tile<32x32, f32>, #ttcore.view<map(4)>, #l1>
  %1 = ttir.view_layout %0 : memref<1x2x2x2x!ttcore.tile<32x32, f32>, #ttcore.view<map(4)>, #l1> -> memref<1x2x2x2x!ttcore.tile<32x32, f32>, #ttcore.view<map(4)>, #l1>
  return %1 : memref<1x2x2x2x!ttcore.tile<32x32, f32>, #ttcore.view<map(4)>, #l1>
}

func.func @composed(%arg0: memref<2x2x2x2x!ttcore.tile<32x32, f32>, #ttcore.shard<8192x4096>, #l1>) -> memref<2x2x2x2x!ttcore.tile<32x32, f32>, #ttcore.view<map(4)>, #l1> {
  // CHECK: %view = ttir.view_layout %arg0
  // CHECK-NEXT: return %view : memref<2x2x2x2x!ttcore.tile<32x32, f32>, #ttcore.view<map(4)>, #l1>
  %0 = ttir.view_layout %arg0 : memref<2x2x2x2x!ttcore.tile<32x32, f32>, #ttcore.shard<8192x4096>, #l1> -> memref<2x2x2x2x!ttcore.tile<32x32, f32>, #ttcore.view<(d0, d1, d2, d3) -> (d3, d2, d1, d0)>, #l1>
  %1 = ttir.view_layout %0 : memref<2x2x2x2x!ttcore.tile<32x32, f32>, #ttcore.view<(d0, d1, d2, d3) -> (d3, d2, d1, d0)>, #l1> -> memref<2x2x2x2x!ttcore.tile<32x32, f32>, #ttcore.view<(d0, d1, d2, d3) -> (d3, d2, d1, d0)>, #l1>
  %2 = ttir.view_layout %1 : memref<2x2x2x2x!ttcore.tile<32x32, f32>, #ttcore.view<(d0, d1, d2, d3) -> (d3, d2, d1, d0)>, #l1> -> memref<2x2x2x2x!ttcore.tile<32x32, f32>, #ttcore.view<map(4)>, #l1>
  return %2 : memref<2x2x2x2x!ttcore.tile<32x32, f32>, #ttcore.view<map(4)>, #l1>
}
