// RUN: ttmlir-opt --ttcore-register-device --ttir-generic-lower-dmas --ttir-generic-generate-loops --canonicalize --ttir-generic-regions-to-funcs --convert-ttir-to-ttkernel %s | FileCheck %s -dump-input=always

// The above passes are all necessary!

#l1 = #ttcore.memory_space<l1>
#dram = #ttcore.memory_space<dram>
#map = affine_map<(d0, d1) -> (d0, d1)>
#parallel = #ttcore.iterator_type<parallel>

module {

// Test 1: Small DRAM read, only accessing bank 0
func.func @test_dram_read_onebank(%arg0: memref<1x1x32x32xf32, #ttcore.shard<128x4>, #ttcore.memory_space<dram>>) {
  %alloc_1 = memref.alloc() : memref<1x1x32x32xf32, #ttcore.shard<128x4>, #ttcore.memory_space<l1>>
  %view = ttir.view_layout %arg0 : memref<1x1x32x32xf32, #ttcore.shard<128x4>, #ttcore.memory_space<dram>> -> memref<1x1x32x32xf32, #ttcore.view<(d0, d1, d2, d3) -> (d0 + d2 floordiv 32, d1 + d3 floordiv 32, d2 mod 32, d3 mod 32)>, #ttcore.memory_space<dram>>
  ttir.generic {block_factors = [1, 1], grid = #ttcore.grid<1x1>, indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = [#ttcore.iterator_type<parallel>, #ttcore.iterator_type<parallel>], threads = [#ttir.thread<datamovement>]}
      ins(%view : memref<1x1x32x32xf32, #ttcore.view<(d0, d1, d2, d3) -> (d0 + d2 floordiv 32, d1 + d3 floordiv 32, d2 mod 32, d3 mod 32)>, #ttcore.memory_space<dram>>)
      outs(%alloc_1 : memref<1x1x32x32xf32, #ttcore.shard<128x4>, #ttcore.memory_space<l1>>)  {
  ^datamovement0(%cb0: memref<32x32xf32, #ttcore.memory_space<l1>>, %cb1: memref<32x32xf32, #ttcore.memory_space<l1>>):
    ttir.await %cb0 : (memref<32x32xf32, #ttcore.memory_space<l1>>)
    %tx = ttir.dma %view<affine_map<(d0, d1) -> (d0, d1)>>, %cb1 : (memref<1x1x32x32xf32, #ttcore.view<(d0, d1, d2, d3) -> (d0 + d2 floordiv 32, d1 + d3 floordiv 32, d2 mod 32, d3 mod 32)>, #ttcore.memory_space<dram>>, memref<32x32xf32, #ttcore.memory_space<l1>>) -> !ttir.mem_tx
    ttir.dma_wait %tx
    ttir.yield %cb1 : (memref<32x32xf32, #ttcore.memory_space<l1>>)
  }
  return
}

// Should be lowered to a single noc_async_read() call; NO loops or guards should be generated here
// CHECK-LABEL: func.func private @datamovement_kernel0
// CHECK-NOT: scf.for
// CHECK-NOT: scf.if
// CHECK: ttkernel.get_noc_addr_from_bank_id
// CHECK: ttkernel.get_write_ptr
// CHECK: ttkernel.noc_async_read

// Test 2: DRAM read spanning all DRAM banks
func.func @test_dram_read_multibank(%arg0: memref<1x1x128x128xf32, #ttcore.shard<128x4>, #ttcore.memory_space<dram>>) {
  %alloc_1 = memref.alloc() : memref<1x1x128x128xf32, #ttcore.shard<128x4>, #ttcore.memory_space<l1>>
  %view = ttir.view_layout %arg0 : memref<1x1x128x128xf32, #ttcore.shard<128x4>, #ttcore.memory_space<dram>> -> memref<1x1x128x128xf32, #ttcore.view<(d0, d1, d2, d3) -> (d0 + d2 floordiv 32, d1 + d3 floordiv 32, d2 mod 32, d3 mod 32)>, #ttcore.memory_space<dram>>
  ttir.generic {block_factors = [1, 1], grid = #ttcore.grid<1x1>, indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = [#ttcore.iterator_type<parallel>, #ttcore.iterator_type<parallel>], threads = [#ttir.thread<datamovement>]}
      ins(%view : memref<1x1x128x128xf32, #ttcore.view<(d0, d1, d2, d3) -> (d0 + d2 floordiv 32, d1 + d3 floordiv 32, d2 mod 32, d3 mod 32)>, #ttcore.memory_space<dram>>)
      outs(%alloc_1 : memref<1x1x128x128xf32, #ttcore.shard<128x4>, #ttcore.memory_space<l1>>)  {
  ^datamovement0(%cb0: memref<128x128xf32, #ttcore.memory_space<l1>>, %cb1: memref<128x128xf32, #ttcore.memory_space<l1>>):
    ttir.await %cb0 : (memref<128x128xf32, #ttcore.memory_space<l1>>)
    %tx = ttir.dma %view<affine_map<(d0, d1) -> (d0, d1)>>, %cb1 : (memref<1x1x128x128xf32, #ttcore.view<(d0, d1, d2, d3) -> (d0 + d2 floordiv 32, d1 + d3 floordiv 32, d2 mod 32, d3 mod 32)>, #ttcore.memory_space<dram>>, memref<128x128xf32, #ttcore.memory_space<l1>>) -> !ttir.mem_tx
    ttir.dma_wait %tx
    ttir.yield %cb1 : (memref<128x128xf32, #ttcore.memory_space<l1>>)
  }
  return
}

// Should be lowered to a guarded, coalesced gather loop
// CHECK-LABEL: func.func private @datamovement_kernel1
// CHECK: scf.for
// CHECK: scf.for
// CHECK: scf.if
// CHECK: ttkernel.get_noc_addr_from_bank_id
// CHECK: ttkernel.get_write_ptr
// CHECK: ttkernel.noc_async_read

}
