
!inT = memref<1x1x4x4x!ttcore.tile<32x32, f32>, #ttcore.shard<16384x4096>, #ttcore.memory_space<l1>>
!sbT = !inT
!stT = memref<1x1x4x4x!ttcore.tile<32x32, f32>, #ttcore.view<map(4)>, #ttcore.memory_space<l1>>

%inA = memref.alloc() : !inT
%sbA = memref.alloc() : !sbT
%streamA = "ttir.stream_layout"(%inA, %sbA) : (!inT, !sbT) -> !stT

%inB = memref.alloc() : !inT
%sbB = memref.alloc() : !sbT
%streamB = "ttir.stream_layout"(%inB, %sbB) : (!inT, !sbT) -> !stT

%out = memref.alloc()   : !inT
%sbOut = memref.alloc() : !sbT
%streamOut = "ttir.stream_layout"(%out, %sbOut) : (!inT, !sbT) -> !stT

!cbT = memref<4x4x!ttcore.tile<32x32, f32>, #ttcore.memory_space<l1>>

ttir.generic {
    block_factors = [1, 1],
    grid = #ttcore.grid<1x1>,
    indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,affine_map<(d0, d1) -> (d0, d1)>,affine_map<(d0, d1) -> (d0, d1)>],
    iterator_types = [#ttcore.iterator_type<parallel>, #ttcore.iterator_type<parallel>],
    threads = [#ttir.thread<datamovement>, #ttir.thread<datamovement>, #ttir.thread<datamovement>, #ttir.thread<compute>]}
    ins(%streamA, %streamB : !stT, !stT)
    outs(%streamOut : !stT)
{
^datamovement0(%cb0 : !cbT, %cb1 : !cbT, %cb2 : !cbT):
  %tx = ttir.dma %streamA<affine_map<(d0, d1) -> (d0, d1)>>, %cb0 : (!stT, !cbT) -> !ttir.mem_tx
  ttir.dma_wait %tx
  ttir.yield %cb0 : (!cbT)
},
{
^datamovement1(%cb0 : !cbT, %cb1 : !cbT, %cb2 : !cbT):
  %tx = ttir.dma %streamB<affine_map<(d0, d1) -> (d0, d1)>>, %cb1 : (!stT, !cbT) -> !ttir.mem_tx
  ttir.dma_wait %tx
  ttir.yield %cb1 : (!cbT)
},
{
^datamovement2(%cb0 : !cbT, %cb1 : !cbT, %cb2 : !cbT):
  ttir.await %cb2 : (!cbT)
  %tx = ttir.dma %cb2, %streamOut<affine_map<(d0, d1) -> (d0, d1)>> : (!cbT, !stT) -> !ttir.mem_tx
  ttir.dma_wait %tx
},
{
^compute0(%cb0 : !cbT, %cb1 : !cbT, %cb2 : !cbT):
  ttir.yield %cb2 : (!cbT)
}
