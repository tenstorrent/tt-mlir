// RUN: ttmlir-opt --d2m-expand-dma-read-composite-view %s | FileCheck %s

// Input is post-LowerLoadStoreOpsToDMA IR: a DM-thread generic that reads from
// a composite_view whose inputs are a memref.alloc + a d2m.fill_buffer.
//
// ExpandDMAReadCompositeView lifts every fill_buffer input into a
// reserve / d2m.fill_pad_cb / push triple at the top of the datamovement
// region. Subsequent per-stick pad-branch dma_reads source from the per-core
// local CB (local-form dma_read, no grid coords on src).

#l1 = #ttcore.memory_space<l1>
#layout = #ttcore.metal_layout<logical_shape = 224x224, dim_alignments = 32x32, collapsed_intervals = dense<> : tensor<0x2xi64>, l1, sharded>
#system_desc = #ttcore.system_desc<[{role = host, target_triple = "x86_64-pc-linux-gnu"}], [{arch = <wormhole_b0>, grid = 8x8, coord_translation_offsets = 18x18, l1_size = 1499136, num_dram_channels = 12, dram_channel_size = 1073741824, noc_l1_address_align_bytes = 16, pcie_address_align_bytes = 32, noc_dram_address_align_bytes = 32, l1_unreserved_base = 1024, erisc_l1_unreserved_base = 1024, dram_unreserved_base = 1024, dram_unreserved_end = 1073741824, supported_data_types = [<f32>, <f16>, <bf16>, <bfp_f8>, <bfp_bf8>, <bfp_f4>, <bfp_bf4>, <bfp_f2>, <bfp_bf2>, <u32>, <u16>, <u8>, <si32>], supported_tile_sizes = [ 4x16,  16x16,  32x16,  4x32,  16x32,  32x32], dst_physical_size_tiles = 16, num_cbs = 32, num_compute_threads = 1, num_datamovement_threads = 2, dram_grid = 1x12, dram_bank_to_logical_worker_noc0 = [(7, 3), (0, 0), (3, 0), (4, 0), (0, 4), (7, 7), (1, 4), (6, 4), (5, 4), (2, 6), (3, 4), (4, 4)], dram_bank_to_logical_worker_noc1 = [(7, 3), (0, 0), (3, 0), (4, 0), (0, 4), (7, 7), (1, 4), (6, 4), (5, 4), (2, 6), (3, 4), (4, 4)]}], [0], [1 : i32], [ 0x0x0x0]>

module attributes {ttcore.system_desc = #system_desc} {
  ttcore.device @default_device = <workerGrid = #ttcore.grid<8x8, virt_to_physical_map = (d0, d1) -> (0, d0, d1), physical_to_virt_map = (d0, d1, d2) -> (d1, d2)>, dramGrid = #ttcore.grid<1x12>, l1Map = (d0, d1, d2)[s0] -> (0, d0, d1, d2 + s0), dramMap = (d0, d1, d2)[s0, s1, s2, s3, s4, s5, s6] -> (0, 0, (((d0 * s1) * (s2 * (s3 * s6)) + d1 * (s2 * (s3 * s6)) + d2) floordiv s4) mod 12, ((((d0 * s1) * (s2 * (s3 * s6)) + d1 * (s2 * (s3 * s6)) + d2) floordiv s4) floordiv 12) * s4 + ((d0 * s1) * (s2 * (s3 * s6)) + d1 * (s2 * (s3 * s6)) + d2) mod s4 + s5), meshShape = , chipIds = [0]>

  // CHECK-LABEL: func.func @pad_dm_generic
  func.func @pad_dm_generic(%arg0: memref<224x224xbf16>) -> memref<7x8x1x1x!ttcore.tile<32x32, bf16>, #ttcore.shard<2048x2048, 1>, #l1> {
    %alloc = memref.alloc() {address = 5120 : i64, alignment = 16 : i64} : memref<7x7x32x32xbf16, #ttcore.shard<64x2, 1>, #l1>
    d2m.to_device %arg0, %alloc layout = #layout : memref<224x224xbf16> into memref<7x7x32x32xbf16, #ttcore.shard<64x2, 1>, #l1>
    %0 = d2m.fill_buffer() {address = 9216 : i64, alignment = 16 : i64, fixed_shard = array<i64: 32, 32>, value = 1.000000e+00 : bf16} : memref<7x8x32x32xbf16, #ttcore.shard<64x2, 1>, #l1>
    %1 = "d2m.composite_view"(%alloc, %0) <{dim = 1 : si32, logicalSizes = array<i64: 224, 32>}> : (memref<7x7x32x32xbf16, #ttcore.shard<64x2, 1>, #l1>, memref<7x8x32x32xbf16, #ttcore.shard<64x2, 1>, #l1>) -> memref<7x8x32x32xbf16, #ttcore.view<4>, #l1>
    %alloc_0 = memref.alloc() {address = 7168 : i64, alignment = 16 : i64} : memref<7x8x1x1x!ttcore.tile<32x32, bf16>, #ttcore.shard<2048x2048, 1>, #l1>
    %alloc_1 = memref.alloc() {address = 1024 : i64, alignment = 16 : i64, d2m.synchronized_buffer = 2 : i32} : memref<32x32xbf16, #ttcore.cb_layout<64x2, 2>, #l1>
    %2 = d2m.operand_alias %alloc_0 : memref<7x8x1x1x!ttcore.tile<32x32, bf16>, #ttcore.shard<2048x2048, 1>, #l1> -> memref<1x1x!ttcore.tile<32x32, bf16>, #l1>
    // CHECK: d2m.fill_pad_cb
    // bf16(1.0) packed twice into u32 = 0x3F803F80 = 1065369472
    // CHECK-SAME: packed_value = 1065369472 : i32
    // Per-stick pad-branch is a local-form dma_read whose src is a plain L1
    // memref with no shard/device layout.
    // CHECK: d2m.dma_read %{{[^[]+}}[%{{[^,]+}}], %{{[^[]+}}[%{{[^]]+}}], <{{[0-9]+}}> : (memref<{{.*}}, #l1>, memref<{{.*}}, #l1>) -> !d2m.mem_tx<read>
    d2m.generic {block_factors = [], grid = #ttcore.grid<7x8>, indexing_maps = [], iterator_types = [], threads = [#d2m.thread<datamovement, processor = 1>, #d2m.thread<compute>]}
        ins(%1 : memref<7x8x32x32xbf16, #ttcore.view<4>, #l1>)
        outs(%alloc_0 : memref<7x8x1x1x!ttcore.tile<32x32, bf16>, #ttcore.shard<2048x2048, 1>, #l1>)
        additionalArgs(%alloc_1, %2 : memref<32x32xbf16, #ttcore.cb_layout<64x2, 2>, #l1>, memref<1x1x!ttcore.tile<32x32, bf16>, #l1>)
     {
      %10 = d2m.get_cb(2) resolution_stage =  compile : <memref<32x32xbf16, #ttcore.cb_layout<64x2, 2>, #l1>>
      %core0 = d2m.core_index(0) : index
      %core1 = d2m.core_index(1) : index
      %11 = d2m.reserve %10 : <memref<32x32xbf16, #ttcore.cb_layout<64x2, 2>, #l1>> -> memref<32x32xbf16, #ttcore.cb_layout<64x2, 2>, #l1>
      %tx = d2m.dma_read %1[%core0, %core1], %11, <0> : (memref<7x8x32x32xbf16, #ttcore.view<4>, #l1>, memref<32x32xbf16, #ttcore.cb_layout<64x2, 2>, #l1>) -> !d2m.mem_tx<read>
      d2m.dma_wait %tx : !d2m.mem_tx<read>
      d2m.push %10 : <memref<32x32xbf16, #ttcore.cb_layout<64x2, 2>, #l1>>
    }, {
      %10 = d2m.get_cb(3) resolution_stage =  compile : <memref<1x1x!ttcore.tile<32x32, bf16>, #l1>>
      %11 = d2m.get_cb(2) resolution_stage =  compile : <memref<32x32xbf16, #ttcore.cb_layout<64x2, 2>, #l1>>
      %12 = d2m.wait %11 : <memref<32x32xbf16, #ttcore.cb_layout<64x2, 2>, #l1>> -> memref<32x32xbf16, #ttcore.cb_layout<64x2, 2>, #l1>
      %13 = d2m.reserve %10 : <memref<1x1x!ttcore.tile<32x32, bf16>, #l1>> -> memref<1x1x!ttcore.tile<32x32, bf16>, #l1>
      "d2m.tile_tilize_block"(%12, %13) : (memref<32x32xbf16, #ttcore.cb_layout<64x2, 2>, #l1>, memref<1x1x!ttcore.tile<32x32, bf16>, #l1>) -> ()
      d2m.push %10 : <memref<1x1x!ttcore.tile<32x32, bf16>, #l1>>
      %14 = d2m.wait %10 : <memref<1x1x!ttcore.tile<32x32, bf16>, #l1>> -> memref<1x1x!ttcore.tile<32x32, bf16>, #l1>
      d2m.pop %10 : <memref<1x1x!ttcore.tile<32x32, bf16>, #l1>>
      d2m.pop %11 : <memref<32x32xbf16, #ttcore.cb_layout<64x2, 2>, #l1>>
    }
    memref.dealloc %alloc_1 : memref<32x32xbf16, #ttcore.cb_layout<64x2, 2>, #l1>
    memref.dealloc %alloc : memref<7x7x32x32xbf16, #ttcore.shard<64x2, 1>, #l1>
    memref.dealloc %0 : memref<7x8x32x32xbf16, #ttcore.shard<64x2, 1>, #l1>
    return %alloc_0 : memref<7x8x1x1x!ttcore.tile<32x32, bf16>, #ttcore.shard<2048x2048, 1>, #l1>
  }
}
