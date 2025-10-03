// RUN: ttmlir-opt --ttcore-register-device --d2m-generic-generate-datamovement -o %t %s
// RUN: FileCheck %s --input-file=%t

#l1_ = #ttcore.memory_space<l1>
#dram = #ttcore.memory_space<dram>
#map = affine_map<(d0, d1) -> (d0, d1)>
#parallel = #ttcore.iterator_type<parallel>


// grid now has three parameters: grid, physGrid, and mapping
//   (1st) grid:  The primary 'virtual' grid; this is used almost everywhere
//   (2nd) physGrid: The physical grid; primarily defines active core range
//   (3rd) mapping: Maps physical grid coords to virtual grid indices (used in DMA stream indexing)
#gridAttr = #ttcore.grid<1x64, 8x8, (d0,d1) -> (0, 8*d0 + d1)>

// ShardLayoutAttr has a new optional param coreVirtualizationMap. This behaves
// like a built-in view applied *after* any views into this memref, but _before_
// the shard layout and device maps. A non-empty core virt map indicates that
// the grid of the memref is virtual.
#virtShardLayoutAttr = #ttcore.shard<16384x4096, 1, (d0, d1) -> (d1 floordiv 8, d1 mod 8)>

!virtGridT = memref<1x64x4x4x!ttcore.tile<32x32, f32>, #virtShardLayoutAttr, #ttcore.memory_space<l1>>

// NOTE: streaming a virtual grid tensor DOES NOT fold/absorb its affine map
!streamT   = memref<1x64x4x4x!ttcore.tile<32x32, f32>, #ttcore.view<(d0,d1,d2,d3) -> (d0,d1,d2,d3)>, #ttcore.memory_space<l1>>
!cbT2 = memref<4x4x!ttcore.tile<32x32, f32>, #ttcore.memory_space<l1>>

// CHECK: func.func @virtual_grid_blocked
func.func @virtual_grid_blocked(%arg0 : !virtGridT) -> () {

  // input
  %In = memref.alloc() : !virtGridT
  %Storage = memref.alloc() : !virtGridT
  %StreamIn = "d2m.stream_layout"(%In,%Storage) : (!virtGridT,!virtGridT) -> !streamT

  // output
  %Out = memref.alloc() : !virtGridT

  d2m.generic {
    block_factors = [1, 1],
    grid = #gridAttr,
    indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,affine_map<(d0, d1) -> (d0, d1)>],
    iterator_types = [#ttcore.iterator_type<parallel>, #ttcore.iterator_type<parallel>],
    threads = [#d2m.thread<compute>]
  }
    ins(%StreamIn : !streamT)
    outs(%Out     : !virtGridT)
  {
    ^compute0(%cb0 : !cbT2, %cb1 : !cbT2):
  }

  return
}
