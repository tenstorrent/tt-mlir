#l1 = #ttcore.memory_space<l1>
module {
  func.func @generic0(%arg0: memref<1x1x8x24x!ttcore.tile<32x32, f32>, #ttcore.shard<98304x4096, 1>, #l1>, %arg1: memref<1x1x24x32x!ttcore.tile<32x32, f32>, #ttcore.shard<131072x4096, 1>, #l1>) -> memref<8x8x1x4x!ttcore.tile<32x32, f32>, #ttcore.shard<16384x4096, 1>, #l1> {
    %alloc = memref.alloc() {address = 4096 : i64, alignment = 64 : i64} : memref<8x8x1x4x!ttcore.tile<32x32, f32>, #ttcore.shard<16384x4096, 1>, #l1>
    %alloc_0 = memref.alloc() {address = 65536 : i64, alignment = 64 : i64} : memref<8x8x1x3x!ttcore.tile<32x32, f32>, #ttcore.shard<12288x4096, 1>, #l1>
    %stream = "d2m.stream_layout"(%arg0, %alloc_0) : (memref<1x1x8x24x!ttcore.tile<32x32, f32>, #ttcore.shard<98304x4096, 1>, #l1>, memref<8x8x1x3x!ttcore.tile<32x32, f32>, #ttcore.shard<12288x4096, 1>, #l1>) -> memref<8x8x1x3x!ttcore.tile<32x32, f32>, #ttcore.view<map(4)>, #l1>
    %alloc_1 = memref.alloc() {address = 86016 : i64, alignment = 64 : i64} : memref<8x8x3x4x!ttcore.tile<32x32, f32>, #ttcore.shard<16384x4096, 1>, #l1>
    %stream_2 = "d2m.stream_layout"(%arg1, %alloc_1) : (memref<1x1x24x32x!ttcore.tile<32x32, f32>, #ttcore.shard<131072x4096, 1>, #l1>, memref<8x8x3x4x!ttcore.tile<32x32, f32>, #ttcore.shard<16384x4096, 1>, #l1>) -> memref<8x8x3x4x!ttcore.tile<32x32, f32>, #ttcore.view<map(4)>, #l1>
    d2m.generic {block_factors = [], grid = #ttcore.grid<8x8>, indexing_maps = [], iterator_types = [], threads = [#d2m.thread<datamovement, @datamovement_kernel0>, #d2m.thread<datamovement, @datamovement_kernel1>, #d2m.thread<compute, @compute_kernel2>]}
        ins(%stream, %stream_2 : memref<8x8x1x3x!ttcore.tile<32x32, f32>, #ttcore.view<map(4)>, #l1>, memref<8x8x3x4x!ttcore.tile<32x32, f32>, #ttcore.view<map(4)>, #l1>)
        outs(%alloc : memref<8x8x1x4x!ttcore.tile<32x32, f32>, #ttcore.shard<16384x4096, 1>, #l1>)
    return %alloc : memref<8x8x1x4x!ttcore.tile<32x32, f32>, #ttcore.shard<16384x4096, 1>, #l1>
  }
  func.func private @datamovement_kernel0(%arg0: memref<1x3x!ttcore.tile<32x32, f32>, #l1>, %arg1: memref<3x4x!ttcore.tile<32x32, f32>, #l1>, %arg2: memref<1x4x!ttcore.tile<32x32, f32>, #l1>, %arg3: !d2m.semaphore, %arg4: !d2m.semaphore, %arg5: !d2m.semaphore, %arg6: !d2m.semaphore) attributes {d2m.thread = #d2m.thread<datamovement>} {
    return
  }
  func.func private @datamovement_kernel1(%arg0: memref<1x3x!ttcore.tile<32x32, f32>, #l1>, %arg1: memref<3x4x!ttcore.tile<32x32, f32>, #l1>, %arg2: memref<1x4x!ttcore.tile<32x32, f32>, #l1>, %arg3: !d2m.semaphore, %arg4: !d2m.semaphore, %arg5: !d2m.semaphore, %arg6: !d2m.semaphore) attributes {d2m.thread = #d2m.thread<datamovement>} {
    return
  }
  func.func private @compute_kernel2(%arg0: memref<1x3x!ttcore.tile<32x32, f32>, #l1>, %arg1: memref<3x4x!ttcore.tile<32x32, f32>, #l1>, %arg2: memref<1x4x!ttcore.tile<32x32, f32>, #l1>, %arg3: !d2m.semaphore, %arg4: !d2m.semaphore, %arg5: !d2m.semaphore, %arg6: !d2m.semaphore) attributes {d2m.thread = #d2m.thread<compute>} {
    return
  }
}
