// RUN: ttmlir-opt --ttcore-register-device --d2m-generic-lower-dmas --d2m-generic-generate-loops --canonicalize --d2m-generic-regions-to-funcs --convert-d2m-to-ttkernel %s | FileCheck %s -dump-input=always

// The above passes are all necessary!

#l1 = #ttcore.memory_space<l1>
#dram = #ttcore.memory_space<dram>
#map = affine_map<(d0, d1) -> (d0, d1)>
#parallel = #ttcore.iterator_type<parallel>

module {

// Test 1: Small DRAM read, only accessing bank 0
// CHECK-LABEL: func.func private @datamovement_kernel0
func.func @test_dram_read_onebank(%arg0: memref<1x1x32x32xf32, #ttcore.shard<128x4>, #ttcore.memory_space<dram>>) {

  // Should be lowered to a single noc_async_read() call; NO loops or guards should be generated here
  //   CHECK-NOT: scf.for
  //   CHECK-NOT: scf.if
  //   CHECK: ttkernel.get_noc_addr_from_bank_id
  //   CHECK: ttkernel.get_write_ptr
  //   CHECK: ttkernel.noc_async_read

  %alloc_1 = memref.alloc() : memref<1x1x32x32xf32, #ttcore.shard<128x4>, #ttcore.memory_space<l1>>
  %view = d2m.view_layout %arg0 : memref<1x1x32x32xf32, #ttcore.shard<128x4>, #ttcore.memory_space<dram>> -> memref<1x1x32x32xf32, #ttcore.view<(d0, d1, d2, d3) -> (d0 + d2 floordiv 32, d1 + d3 floordiv 32, d2 mod 32, d3 mod 32)>, #ttcore.memory_space<dram>>
  d2m.generic {block_factors = [1, 1], grid = #ttcore.grid<1x1>, indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = [#ttcore.iterator_type<parallel>, #ttcore.iterator_type<parallel>], threads = [#d2m.thread<datamovement>]}
      ins(%view : memref<1x1x32x32xf32, #ttcore.view<(d0, d1, d2, d3) -> (d0 + d2 floordiv 32, d1 + d3 floordiv 32, d2 mod 32, d3 mod 32)>, #ttcore.memory_space<dram>>)
      outs(%alloc_1 : memref<1x1x32x32xf32, #ttcore.shard<128x4>, #ttcore.memory_space<l1>>)  {
  ^datamovement0(%cb0: !d2m.cb<memref<32x32xf32, #ttcore.memory_space<dram>>>, %cb1: !d2m.cb<memref<32x32xf32, #ttcore.memory_space<l1>>>):
    %buf = d2m.reserve %cb1 : !d2m.cb<memref<32x32xf32, #ttcore.memory_space<l1>>> -> memref<32x32xf32, #ttcore.memory_space<l1>>
    %tx = d2m.dma %view<affine_map<(d0, d1) -> (d0, d1)>>, %buf : (memref<1x1x32x32xf32, #ttcore.view<(d0, d1, d2, d3) -> (d0 + d2 floordiv 32, d1 + d3 floordiv 32, d2 mod 32, d3 mod 32)>, #ttcore.memory_space<dram>>, memref<32x32xf32, #ttcore.memory_space<l1>>) -> !d2m.mem_tx
    d2m.dma_wait %tx
  }
  return
}

// Test 2: DRAM read spanning all DRAM banks
// CHECK-LABEL: func.func private @datamovement_kernel1
func.func @test_dram_read_multibank(%arg0: memref<1x1x128x128xf32, #ttcore.shard<128x4>, #ttcore.memory_space<dram>>) {

  // Should be lowered to a guarded, coalesced gather loop
  //   CHECK: scf.for
  //   CHECK: scf.for
  //   CHECK: scf.if
  //   CHECK: ttkernel.get_noc_addr_from_bank_id
  //   CHECK: ttkernel.get_write_ptr
  //   CHECK: ttkernel.noc_async_read

  %alloc_1 = memref.alloc() : memref<1x1x128x128xf32, #ttcore.shard<128x4>, #ttcore.memory_space<l1>>
  %view = d2m.view_layout %arg0 : memref<1x1x128x128xf32, #ttcore.shard<128x4>, #ttcore.memory_space<dram>> -> memref<1x1x128x128xf32, #ttcore.view<(d0, d1, d2, d3) -> (d0 + d2 floordiv 32, d1 + d3 floordiv 32, d2 mod 32, d3 mod 32)>, #ttcore.memory_space<dram>>
  d2m.generic {block_factors = [1, 1], grid = #ttcore.grid<1x1>, indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = [#ttcore.iterator_type<parallel>, #ttcore.iterator_type<parallel>], threads = [#d2m.thread<datamovement>]}
      ins(%view : memref<1x1x128x128xf32, #ttcore.view<(d0, d1, d2, d3) -> (d0 + d2 floordiv 32, d1 + d3 floordiv 32, d2 mod 32, d3 mod 32)>, #ttcore.memory_space<dram>>)
      outs(%alloc_1 : memref<1x1x128x128xf32, #ttcore.shard<128x4>, #ttcore.memory_space<l1>>)  {
  ^datamovement0(%cb0: !d2m.cb<memref<128x128xf32, #ttcore.memory_space<dram>>>, %cb1: !d2m.cb<memref<128x128xf32, #ttcore.memory_space<l1>>>):
    %buf = d2m.reserve %cb1 : !d2m.cb<memref<128x128xf32, #ttcore.memory_space<l1>>> -> memref<128x128xf32, #ttcore.memory_space<l1>>
    %tx = d2m.dma %view<affine_map<(d0, d1) -> (d0, d1)>>, %buf : (memref<1x1x128x128xf32, #ttcore.view<(d0, d1, d2, d3) -> (d0 + d2 floordiv 32, d1 + d3 floordiv 32, d2 mod 32, d3 mod 32)>, #ttcore.memory_space<dram>>, memref<128x128xf32, #ttcore.memory_space<l1>>) -> !d2m.mem_tx
    d2m.dma_wait %tx
  }
  return
}

}
