
!inT = memref<1x1x4x4x!ttcore.tile<32x32, f32>, #ttcore.shard<16384x4096>, #ttcore.memory_space<l1>>
!sbT = !inT
!stT = memref<1x1x4x4x!ttcore.tile<32x32, f32>, #ttcore.view<map(4)>, #ttcore.memory_space<l1>>

%inA = memref.alloc() {address = 296352 : i64, alignment = 16 : i64} : !inT
%sbA = memref.alloc() {address = 361888 : i64, alignment = 16 : i64} : !sbT
%streamA = "ttir.stream_layout"(%inA, %sbA) : (!inT, !sbT) -> !stT

%inB = memref.alloc() {address = 427424 : i64, alignment = 16 : i64} : !inT
%sbB = memref.alloc() {address = 492960 : i64, alignment = 16 : i64} : !sbT
%streamB = "ttir.stream_layout"(%inB, %sbB) : (!inT, !sbT) -> !stT

%out = memref.alloc() {address = 558496 : i64, alignment = 16 : i64} : !inT
%sbOut = memref.alloc() {address = 624032 : i64, alignment = 16 : i64} : !sbT
%streamOut = "ttir.stream_layout"(%out, %sbOut) : (!inT, !sbT) -> !stT

!cbT = memref<4x4x!ttcore.tile<32x32, f32>, #ttcore.memory_space<l1>>

ttir.generic {
    block_factors = [1, 1],
    grid = #ttcore.grid<1x1>,
    indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,affine_map<(d0, d1) -> (d0, d1)>,affine_map<(d0, d1) -> (d0, d1)>],
    iterator_types = [#ttcore.iterator_type<parallel>, #ttcore.iterator_type<parallel>],
    threads = [#ttir.thread<compute>]}
    ins(%streamA, %streamB : !stT, !stT)
    outs(%streamOut : !stT)
{
^compute0(%cb0 : !cbT, %cb1 : !cbT, %cb2 : !cbT):
  ttir.yield %cb2 : (!cbT)
}
