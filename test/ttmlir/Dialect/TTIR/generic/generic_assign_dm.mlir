func.func @mm(%arg0: memref<32x256xf16>, %arg1: memref<256x32xf16>) -> memref<32x32xf16> {
    %alloc = memref.alloc() {address = 130016 : i64, alignment = 16 : i64} : memref<1x1x1x8x!tt.tile<32x32, f16>, #tt.shard<16384x2048>, #tt.memory_space<l1>>
    %alloc_1 = memref.alloc() {address = 113632 : i64, alignment = 16 : i64} : memref<1x1x8x1x!tt.tile<32x32, f16>, #tt.shard<2048x2048>, #tt.memory_space<l1>>
    %alloc_4 = memref.alloc() {address = 146400 : i64, alignment = 16 : i64} : memref<1x1x1x8x!tt.tile<32x32, f16>, #tt.shard<16384x2048>, #tt.memory_space<l1>>
    %alloc_5 = memref.alloc() {address = 97248 : i64, alignment = 16 : i64} : memref<1x1x8x1x!tt.tile<32x32, f16>, #tt.shard<2048x2048>, #tt.memory_space<l1>>
    %stream = "ttir.stream_layout"(%alloc, %alloc_4) : (memref<1x1x1x8x!tt.tile<32x32, f16>, #tt.shard<16384x2048>, #tt.memory_space<l1>>, memref<1x1x1x8x!tt.tile<32x32, f16>, #tt.shard<16384x2048>, #tt.memory_space<l1>>) -> memref<1x1x1x8x!tt.tile<32x32, f16>, #tt.view<map(4)>, #tt.memory_space<l1>>
    %stream_6 = "ttir.stream_layout"(%alloc_1, %alloc_5) : (memref<1x1x8x1x!tt.tile<32x32, f16>, #tt.shard<2048x2048>, #tt.memory_space<l1>>, memref<1x1x8x1x!tt.tile<32x32, f16>, #tt.shard<2048x2048>, #tt.memory_space<l1>>) -> memref<1x1x8x1x!tt.tile<32x32, f16>, #tt.view<map(4)>, #tt.memory_space<l1>>

    // create alloc and stream for third output cb
    %alloc_8 = memref.alloc() {address = 113632 : i64, alignment = 16 : i64} 
        : memref<1x1x1x1x!tt.tile<32x32, f16>, #tt.shard<2048x2048>, #tt.memory_space<l1>>
    %alloc_9 = memref.alloc() {address = 97248  : i64, alignment = 16 : i64} 
        : memref<1x1x1x1x!tt.tile<32x32, f16>, #tt.shard<2048x2048>, #tt.memory_space<l1>>
    %stream_9 = "ttir.stream_layout"(%alloc_8, %alloc_9) 
        : (memref<1x1x1x1x!tt.tile<32x32, f16>, #tt.shard<2048x2048>, #tt.memory_space<l1>>, 
           memref<1x1x1x1x!tt.tile<32x32, f16>, #tt.shard<2048x2048>, #tt.memory_space<l1>>) 
           -> memref<1x1x1x1x!tt.tile<32x32, f16>, #tt.view<map(4)>, #tt.memory_space<l1>>

    ttir.generic {
        grid = #tt.grid<1x1>, 
        indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d2)>, affine_map<(d0, d1, d2) -> (d2, d1)>, affine_map<(d0, d1, d2) -> (d0, d1)>], 
        iterator_types = [#tt.iterator_type<parallel>, #tt.iterator_type<parallel>, #tt.iterator_type<reduction>], 
        threads = [#ttir.thread<datamovement>, #ttir.thread<datamovement>, #ttir.thread<datamovement>, #ttir.thread<compute>]}
        ins(%stream, %stream_6: 
                memref<1x1x1x8x!tt.tile<32x32, f16>, #tt.view<map(4)>, #tt.memory_space<l1>>, 
                memref<1x1x8x1x!tt.tile<32x32, f16>, #tt.view<map(4)>, #tt.memory_space<l1>>)
        outs(%stream_9: 
                memref<1x1x1x1x!tt.tile<32x32, f16>, #tt.view<map(4)>, #tt.memory_space<l1>>)  {

    ^datamovement0(%cb0: memref<1x8x!tt.tile<32x32, f16>, #tt.memory_space<l1>>, %cb1: memref<8x1x!tt.tile<32x32, f16>, #tt.memory_space<l1>>, %cb2: memref<1x1x!tt.tile<32x32, f16>, #tt.memory_space<l1>>):
      %tx = ttir.dma %stream<affine_map<(d0, d1, d2) -> (d0, d2)>>, %cb0 : (memref<1x1x1x8x!tt.tile<32x32, f16>, #tt.view<map(4)>, #tt.memory_space<l1>>, memref<1x8x!tt.tile<32x32, f16>, #tt.memory_space<l1>>) -> !ttir.mem_tx
      ttir.dma_wait %tx
      ttir.yield %cb0 : (memref<1x8x!tt.tile<32x32, f16>, #tt.memory_space<l1>>)
    }, {
    ^datamovement1(%cb0: memref<1x8x!tt.tile<32x32, f16>, #tt.memory_space<l1>>, %cb1: memref<8x1x!tt.tile<32x32, f16>, #tt.memory_space<l1>>, %cb2: memref<1x1x!tt.tile<32x32, f16>, #tt.memory_space<l1>>):
      %tx = ttir.dma %stream_6<affine_map<(d0, d1, d2) -> (d2, d1)>>, %cb1 : (memref<1x1x8x1x!tt.tile<32x32, f16>, #tt.view<map(4)>, #tt.memory_space<l1>>, memref<8x1x!tt.tile<32x32, f16>, #tt.memory_space<l1>>) -> !ttir.mem_tx
      ttir.dma_wait %tx
      ttir.yield %cb1 : (memref<8x1x!tt.tile<32x32, f16>, #tt.memory_space<l1>>)
    }, {
    ^datamovement2(
        %cb0: memref<1x8x!tt.tile<32x32, f16>, #tt.memory_space<l1>>, 
        %cb1: memref<8x1x!tt.tile<32x32, f16>, #tt.memory_space<l1>>, 
        %cb2: memref<1x1x!tt.tile<32x32, f16>, #tt.memory_space<l1>>):

      ttir.await %cb2 : (memref<1x1x!tt.tile<32x32, f16>, #tt.memory_space<l1>>)
      %tx = ttir.dma %cb2, %stream_9<affine_map<(d0, d1, d2) -> (d0, d1)>> : ( memref<1x1x!tt.tile<32x32, f16>, #tt.memory_space<l1>>, memref<1x1x1x1x!tt.tile<32x32, f16>, #tt.view<map(4)>, #tt.memory_space<l1>>) -> !ttir.mem_tx
      ttir.dma_wait %tx
    }, {
    ^compute0(%cb0: memref<1x8x!tt.tile<32x32, f16>, #tt.memory_space<l1>>, %cb1: memref<8x1x!tt.tile<32x32, f16>, #tt.memory_space<l1>>, %cb2: memref<1x1x!tt.tile<32x32, f16>, #tt.memory_space<l1>>):
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %c8 = arith.constant 8 : index
    }
    memref.dealloc %alloc_4 : memref<1x1x1x8x!tt.tile<32x32, f16>, #tt.shard<16384x2048>, #tt.memory_space<l1>>
    memref.dealloc %alloc_1 : memref<1x1x8x1x!tt.tile<32x32, f16>, #tt.shard<2048x2048>, #tt.memory_space<l1>>
    memref.dealloc %alloc_5 : memref<1x1x8x1x!tt.tile<32x32, f16>, #tt.shard<2048x2048>, #tt.memory_space<l1>>
    memref.dealloc %alloc : memref<1x1x1x8x!tt.tile<32x32, f16>, #tt.shard<16384x2048>, #tt.memory_space<l1>>

    // dummy return for top level func
    %alloc_7 = memref.alloc() : memref<32x32xf16>
    return %alloc_7 : memref<32x32xf16>
}