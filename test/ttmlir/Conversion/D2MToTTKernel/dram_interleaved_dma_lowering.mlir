// RUN: ttmlir-opt --ttcore-register-device --d2m-generic-lower-dmas --d2m-generic-generate-loops --canonicalize --d2m-generic-regions-to-funcs --convert-d2m-to-ttkernel %s | FileCheck %s -dump-input=always

#l1 = #ttcore.memory_space<l1>
#dram = #ttcore.memory_space<dram>
#map = affine_map<(d0, d1) -> (d0, d1)>
#parallel = #ttcore.iterator_type<parallel>

// DRAM interleaved layout: tiles are striped round-robin across DRAM banks
#dram_interleaved = #ttcore.metal_layout<logical_shape = 32x32, dim_alignments = 32x32, collapsed_intervals = dense<[[0, 1], [1, 2]]> : tensor<2x2xi64>, undef, dram, interleaved, index_map = map(0)>
#l1_sharded = #ttcore.metal_layout<logical_shape = 32x32, dim_alignments = 32x32, collapsed_intervals = dense<[[0, 1], [1, 2]]> : tensor<2x2xi64>, undef, l1, sharded, index_map = map(0)>

module {

// Test: DRAM interleaved read should use InterleavedAddrGenFast + NocAsyncReadTile
// CHECK-LABEL: func.func private @datamovement_kernel0
func.func @test_dram_interleaved_read(%arg0: tensor<1x1x1x1x!ttcore.tile<32x32, f32>, #dram_interleaved>) {

  // Should generate interleaved address gen and tile read ops
  //   CHECK: ttkernel.get_dataformat
  //   CHECK: ttkernel.get_interleaved_addr_gen_fast
  //   CHECK: ttkernel.noc_async_read_tile

  %alloc_l1 = d2m.empty() : tensor<1x1x1x1x!ttcore.tile<32x32, f32>, #l1_sharded>
  %view = d2m.view_layout %arg0 : tensor<1x1x1x1x!ttcore.tile<32x32, f32>, #dram_interleaved> -> tensor<1x1x1x1x!ttcore.tile<32x32, f32>, #dram_interleaved>
  d2m.generic {block_factors = [1, 1], grid = #ttcore.grid<1x1>, indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = [#ttcore.iterator_type<parallel>, #ttcore.iterator_type<parallel>], threads = [#d2m.thread<datamovement>]}
      ins(%view : tensor<1x1x1x1x!ttcore.tile<32x32, f32>, #dram_interleaved>)
      outs(%alloc_l1 : tensor<1x1x1x1x!ttcore.tile<32x32, f32>, #l1_sharded>)  {
  ^datamovement0(%cb0: !d2m.cb<memref<1x1x!ttcore.tile<32x32, f32>, #dram>>, %cb1: !d2m.cb<memref<1x1x!ttcore.tile<32x32, f32>, #l1>>):
    %buf = d2m.reserve %cb1 : !d2m.cb<memref<1x1x!ttcore.tile<32x32, f32>, #l1>> -> memref<1x1x!ttcore.tile<32x32, f32>, #l1>
    %tx = d2m.dma %view<affine_map<(d0, d1) -> (d0, d1)>>, %buf : (tensor<1x1x1x1x!ttcore.tile<32x32, f32>, #dram_interleaved>, memref<1x1x!ttcore.tile<32x32, f32>, #l1>) -> !d2m.mem_tx
    d2m.dma_wait %tx
  }
  return
}

}
