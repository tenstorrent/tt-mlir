// RUN: ttmlir-opt --ttcore-register-device --ttir-generic-lower-dmas --ttir-generic-hw-thread-selection --ttir-generic-generate-loops  --ttir-generic-regions-to-funcs  --canonicalize --convert-ttir-to-ttkernel %s | FileCheck %s

#l1 = #ttcore.memory_space<l1>
#dram = #ttcore.memory_space<dram>
#map = affine_map<(d0, d1) -> (d0, d1)>
#parallel = #ttcore.iterator_type<parallel>

module {

// Test 1: Local to local DMA within same core
// CHECK-LABEL: func.func @test_dram_read
func.func @test_dram_read(%arg0: memref<1x1x128x128xf32, #ttcore.shard<128x4>, #ttcore.memory_space<dram>>) {
  %alloc_1 = memref.alloc() {address = 99904 : i64, alignment = 16 : i64} : memref<1x1x128x128xf32, #ttcore.shard<128x4>, #ttcore.memory_space<l1>>
  %view = ttir.view_layout %arg0 : memref<1x1x128x128xf32, #ttcore.shard<128x4>, #ttcore.memory_space<dram>> -> memref<1x1x128x128xf32, #ttcore.view<(d0, d1, d2, d3) -> (d0 + d2 floordiv 32, d1 + d3 floordiv 32, d2 mod 32, d3 mod 32)>, #ttcore.memory_space<dram>>
  ttir.generic {block_factors = [1, 1], grid = #ttcore.grid<1x1>, indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = [#ttcore.iterator_type<parallel>, #ttcore.iterator_type<parallel>], threads = [#ttir.thread<datamovement>]}
      ins(%view : memref<1x1x128x128xf32, #ttcore.view<(d0, d1, d2, d3) -> (d0 + d2 floordiv 32, d1 + d3 floordiv 32, d2 mod 32, d3 mod 32)>, #ttcore.memory_space<dram>>)
      outs(%alloc_1 : memref<1x1x128x128xf32, #ttcore.shard<128x4>, #ttcore.memory_space<l1>>)  {
  ^datamovement0(%cb0: memref<128x128xf32, #ttcore.memory_space<dram>>, %cb1: memref<128x128xf32, #ttcore.memory_space<l1>>):
    %tx = ttir.dma %view<affine_map<(d0, d1) -> (d0, d1)>>, %cb1 : (memref<1x1x128x128xf32, #ttcore.view<(d0, d1, d2, d3) -> (d0 + d2 floordiv 32, d1 + d3 floordiv 32, d2 mod 32, d3 mod 32)>, #ttcore.memory_space<dram>>, memref<128x128xf32, #ttcore.memory_space<l1>>) -> !ttir.mem_tx
    ttir.dma_wait %tx
  }
  return
}

}