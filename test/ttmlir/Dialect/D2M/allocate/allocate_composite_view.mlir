// RUN: ttmlir-opt --ttcore-register-device --d2m-allocate 2>&1 -o %t %s
// RUN: FileCheck %s --input-file=%t

// CHECK-LABEL: func.func @test_composite_view
func.func @test_composite_view() -> memref<1x2x1x1x!ttcore.tile<32x32, f32>, #ttcore.shard<4096x4096, 1>, #ttcore.memory_space<l1>> {
  // CHECK: %[[ALLOC_LHS:.+]] = memref.alloc() {address = {{[0-9]+}} : i64, alignment = {{[0-9]+}} : i64}{{.+}} #l1>
  %alloc_0 = memref.alloc() : memref<1x1x1x1x!ttcore.tile<32x32, f32>, #ttcore.shard<4096x4096, 1>, #ttcore.memory_space<l1>>
  // CHECK: %[[ALLOC_RHS:.+]] = memref.alloc() {address = {{[0-9]+}} : i64, alignment = {{[0-9]+}} : i64}{{.+}} #l1>
  %alloc_1 = memref.alloc() : memref<1x1x1x1x!ttcore.tile<32x32, f32>, #ttcore.shard<4096x4096, 1>, #ttcore.memory_space<l1>>

  %0 = "d2m.composite_view"(%alloc_0, %alloc_1) <{dim = 1 : si32}> : (memref<1x1x1x1x!ttcore.tile<32x32, f32>, #ttcore.shard<4096x4096, 1>, #ttcore.memory_space<l1>>, memref<1x1x1x1x!ttcore.tile<32x32, f32>, #ttcore.shard<4096x4096, 1>, #ttcore.memory_space<l1>>) -> memref<1x2x1x1x!ttcore.tile<32x32, f32>, #ttcore.view<4>, #ttcore.memory_space<l1>>

  // CHECK: %[[ALLOC_OUT:.+]] = memref.alloc() {address = {{[0-9]+}} : i64, alignment = {{[0-9]+}} : i64}{{.+}} #l1>
  %alloc_2 = memref.alloc() : memref<1x2x1x1x!ttcore.tile<32x32, f32>, #ttcore.shard<4096x4096, 1>, #ttcore.memory_space<l1>>

  d2m.generic {block_factors = [1, 1], grid = #ttcore.grid<1x2>, indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = [#ttcore.iterator_type<parallel>, #ttcore.iterator_type<parallel>], threads = [#d2m.thread<unified>]}
      ins(%0 : memref<1x2x1x1x!ttcore.tile<32x32, f32>, #ttcore.view<4>, #ttcore.memory_space<l1>>)
      outs(%alloc_2 : memref<1x2x1x1x!ttcore.tile<32x32, f32>, #ttcore.shard<4096x4096, 1>, #ttcore.memory_space<l1>>)
   {
    %block_factor0 = d2m.get_block_factor(0) : index
    %block_factor1 = d2m.get_block_factor(1) : index
    affine.for %arg2 = 0 to %block_factor0 {
      affine.for %arg3 = 0 to %block_factor1 {
        %alloc_3 = memref.alloc() {alignment = 64 : i64} : memref<1x1x!ttcore.tile<32x32, f32>>
        %block_offset0 = d2m.block_offset(0) : index
        %1 = affine.apply affine_map<(d0)[s0] -> (d0 + s0)>(%arg2)[%block_offset0]
        %block_offset1 = d2m.block_offset(1) : index
        %2 = affine.apply affine_map<(d0)[s0] -> (d0 + s0)>(%arg3)[%block_offset1]
        %3 = d2m.remote_load %alloc_3 %0[%1, %2] : memref<1x1x!ttcore.tile<32x32, f32>>, memref<1x2x1x1x!ttcore.tile<32x32, f32>, #ttcore.view<4>, #ttcore.memory_space<l1>> -> memref<1x1x!ttcore.tile<32x32, f32>, #ttcore.memory_space<l1>>
        %4 = d2m.remote_store %alloc_2[%1, %2] %alloc_3 : memref<1x2x1x1x!ttcore.tile<32x32, f32>, #ttcore.shard<4096x4096, 1>, #ttcore.memory_space<l1>>, memref<1x1x!ttcore.tile<32x32, f32>> -> memref<1x2x1x1x!ttcore.tile<32x32, f32>, #ttcore.shard<4096x4096, 1>, #ttcore.memory_space<l1>>
      } {d2m.blocking_loop = 1 : i64}
    } {d2m.blocking_loop = 0 : i64}
  }

  // CHECK: return %[[ALLOC_OUT]]
  return %alloc_2 : memref<1x2x1x1x!ttcore.tile<32x32, f32>, #ttcore.shard<4096x4096, 1>, #ttcore.memory_space<l1>>
}
