// RUN: ttmlir-opt --ttcore-register-device "--d2m-allocate=stream-insert-policy=infer test-buffer-size-policy=max" -o %t %s
// RUN: FileCheck %s --input-file=%t

// Verify that the allocator replaces in-generic allocs with d2m.alias_buffer
// for operands that do not require streaming (L1, all-parallel, no broadcast).

#l1 = #ttcore.memory_space<l1>

#map = affine_map<(d0, d1) -> (d0, d1)>
#parallel = #ttcore.iterator_type<parallel>

module {

  // An eltwise generic with L1 inputs and all-parallel identity maps.
  // The inputs should get alias_buffer instead of streams.
  // CHECK-LABEL: func @test_alias_buffer_eltwise
  func.func @test_alias_buffer_eltwise() {
    %a = memref.alloc() : memref<1x1x1x1x!ttcore.tile<32x32, f32>, #ttcore.shard<4096x4096, 1>, #l1>
    %b = memref.alloc() : memref<1x1x1x1x!ttcore.tile<32x32, f32>, #ttcore.shard<4096x4096, 1>, #l1>
    %r = memref.alloc() : memref<1x1x1x1x!ttcore.tile<32x32, f32>, #ttcore.shard<4096x4096, 1>, #l1>
    // CHECK: d2m.generic
    d2m.generic {block_factors = [1, 1], grid = #ttcore.grid<1x1>, indexing_maps = [#map, #map, #map], iterator_types = [#parallel, #parallel], threads = [#d2m.thread<unified>]}
        ins(%a, %b : memref<1x1x1x1x!ttcore.tile<32x32, f32>, #ttcore.shard<4096x4096, 1>, #l1>, memref<1x1x1x1x!ttcore.tile<32x32, f32>, #ttcore.shard<4096x4096, 1>, #l1>)
        outs(%r : memref<1x1x1x1x!ttcore.tile<32x32, f32>, #ttcore.shard<4096x4096, 1>, #l1>)  {
    ^unified0:
      %bf0 = d2m.get_block_factor(0) : index
      %bf1 = d2m.get_block_factor(1) : index
      affine.for %i = 0 to %bf0 {
        affine.for %j = 0 to %bf1 {
          %buf_a = memref.alloc() : memref<1x1x!ttcore.tile<32x32, f32>, #l1>
          %buf_b = memref.alloc() : memref<1x1x!ttcore.tile<32x32, f32>, #l1>
          %buf_r = memref.alloc() : memref<1x1x!ttcore.tile<32x32, f32>, #l1>
          // Use a compute op so the generic is not classified as DMA-only.
          %cst = arith.constant 0.0 : f32
          %tile = "d2m.tile_fill"(%cst) : (f32) -> !ttcore.tile<32x32, f32>
          // CHECK: d2m.alias_buffer %{{.*}} : memref<{{.*}}> -> memref<1x1x!ttcore.tile<32x32, f32>,
          // CHECK: d2m.alias_buffer %{{.*}} : memref<{{.*}}> -> memref<1x1x!ttcore.tile<32x32, f32>,
          // CHECK-NOT: d2m.alias_buffer
        } {d2m.blocking_loop = 1}
      } {d2m.blocking_loop = 0}
    }
    return
  }

} // module
