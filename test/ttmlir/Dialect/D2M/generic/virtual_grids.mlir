// RUN: ttmlir-opt --ttcore-register-device --d2m-generic-generate-datamovement -o %t %s
// RUN: FileCheck %s --input-file=%t

#l1_ = #ttcore.memory_space<l1>
#dram = #ttcore.memory_space<dram>
#map = affine_map<(d0, d1) -> (d0, d1)>
#parallel = #ttcore.iterator_type<parallel>

!physT = memref<8x8x4x4x!ttcore.tile<32x32, f32>, #ttcore.shard<16384x4096>, #ttcore.memory_space<l1>>

!virtT = memref<1x64x4x4x!ttcore.tile<32x32, f32>, #ttcore.view<(d0, d1, d2, d3) -> (d1 floordiv 8, d1 mod 8, d2, d3)>, #ttcore.memory_space<l1>>

!cbT = memref<4x4x!ttcore.tile<32x32, f32>, #ttcore.memory_space<l1>>

func.func @virtual_grid_noblocking(%arg0 : memref<1x64x4x4x!ttcore.tile<32x32, f32>, #ttcore.shard<16384x4096>, #ttcore.memory_space<l1>>)  {

  // input
  %physInA = memref.alloc() {address = 296352 : i64, alignment = 16 : i64} : !physT
  %physSbA = memref.alloc() {address = 296352 : i64, alignment = 16 : i64} : !physT
  %virtInA = "d2m.stream_layout"(%physInA,%physSbA) : (!physT,!physT) -> !virtT

  // output
  %physOut   = memref.alloc() {address = 558496 : i64, alignment = 16 : i64} : !physT
  %virtOut   = d2m.view_layout %physOut : !physT -> !virtT

  d2m.generic {
      block_factors = [1, 1],
      grid = #ttcore.grid<8x8>,
      indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,affine_map<(d0, d1) -> (d0, d1)>],
      iterator_types = [#ttcore.iterator_type<parallel>, #ttcore.iterator_type<parallel>],
      threads = [#d2m.thread<compute>]}

      ins(%virtInA  : !virtT)
      outs(%virtOut : !virtT)
  {
  ^compute0(%cb0 : !cbT, %cb1 : !cbT):
  }

  return
}
