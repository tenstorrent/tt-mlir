// RUN: ttmlir-opt --d2m-be-pipeline --convert-d2m-to-ttkernel %s | FileCheck %s

// End-to-end backend test for d2m.gather_core: takes a post-LowerToExplicitForm
// fixture in unified-thread form, runs the entire D2M backend pipeline
// (HoistCBAllocs -> SplitUnifiedThread -> PreallocateMcastSemaphores ->
//  ScheduleDMA -> LowerLoadStoreOpsToDMA -> OptimizeDMA ->
//  ExpandDMAReadCompositeView -> LowerDMAToFullyIndexedForm ->
//  NormalizeThreadArgs -> GenericRegionsToFuncs), and finally the
// D2M -> TTKernel conversion. This pins the full lowering chain for
// gather_core in V1 (option (c) for dst CB asymmetry).
//
// Fixture: 4x4 worker grid, 1x4 gather group at (0,0), collector at (0,0).
// Each source shard is 2x4 tiles. Src/dst CB scratch buffers are
// pre-allocated (post-D2MAllocate shape).

#l1 = #ttcore.memory_space<l1>
#map = affine_map<(d0, d1) -> (d0, d1)>
#parallel = #ttcore.iterator_type<parallel>
#system_desc = #ttcore.system_desc<[{role = host, target_triple = "x86_64-pc-linux-gnu"}], [{arch = <wormhole_b0>, grid = 8x8, coord_translation_offsets = 18x18, l1_size = 1499136, num_dram_channels = 12, dram_channel_size = 1073741824, noc_l1_address_align_bytes = 16, pcie_address_align_bytes = 32, noc_dram_address_align_bytes = 32, l1_unreserved_base = 1024, erisc_l1_unreserved_base = 1024, dram_unreserved_base = 1024, dram_unreserved_end = 1073741824, supported_data_types = [<f32>, <f16>, <bf16>, <bfp_f8>, <bfp_bf8>, <bfp_f4>, <bfp_bf4>, <bfp_f2>, <bfp_bf2>, <u32>, <u16>, <u8>, <si32>], supported_tile_sizes = [ 4x16,  16x16,  32x16,  4x32,  16x32,  32x32], dst_physical_size_tiles = 16, num_cbs = 32, num_compute_threads = 1, num_datamovement_threads = 2, dram_grid = 1x12, dram_bank_to_logical_worker_noc0 = [(7, 3), (0, 0), (3, 0), (4, 0), (0, 4), (7, 7), (1, 4), (6, 4), (5, 4), (2, 6), (3, 4), (4, 4)], dram_bank_to_logical_worker_noc1 = [(7, 3), (0, 0), (3, 0), (4, 0), (0, 4), (7, 7), (1, 4), (6, 4), (5, 4), (2, 6), (3, 4), (4, 4)]}], [0], [1 : i32], [ 0x0x0x0]>

module attributes {ttcore.system_desc = #system_desc} {
  ttcore.device @default_device = <workerGrid = #ttcore.grid<8x8, virt_to_physical_map = (d0, d1) -> (0, d0, d1), physical_to_virt_map = (d0, d1) -> (0, d0, d1)>, dramGrid = #ttcore.grid<1x12>, l1Map = (d0, d1, d2)[s0] -> (0, d0, d1, d2 + s0), dramMap = (d0, d1, d2)[s0, s1, s2, s3, s4, s5, s6] -> (0, 0, (((d0 * s1) * (s2 * (s3 * s6)) + d1 * (s2 * (s3 * s6)) + d2) floordiv s4) mod 12, ((((d0 * s1) * (s2 * (s3 * s6)) + d1 * (s2 * (s3 * s6)) + d2) floordiv s4) floordiv 12) * s4 + ((d0 * s1) * (s2 * (s3 * s6)) + d1 * (s2 * (s3 * s6)) + d2) mod s4 + s5), meshShape = , chipIds = [0]>

  // The backend pipeline lifts the d2m.generic into per-thread func.funcs
  // via D2MGenericRegionsToFuncs. We expect one DM-thread kernel followed
  // by one (empty) compute kernel.

  // CHECK-LABEL: func.func private @datamovement_kernel0
  //
  // Compile-time arg spec: two local semaphores (collectorDone, sourceReady)
  // and two CB ports (src + dst). All four surfaced via
  // get_compile_time_arg_val + get_semaphore / get_cb.
  // CHECK-SAME: ttkernel.arg_spec
  // CHECK-SAME: arg_type = local_semaphore
  // CHECK-SAME: arg_type = local_semaphore
  // CHECK-SAME: arg_type = cb_port
  // CHECK-SAME: arg_type = cb_port
  //
  // Both semaphores and both CB ports are materialized at the top of the
  // kernel body. dst CB comes first (its operand index in the generic is
  // higher); src CB second. We bind names to use them in later CHECKs.
  // CHECK: %[[SEM_DONE:.*]] = ttkernel.get_semaphore
  // CHECK: %[[SEM_READY:.*]] = ttkernel.get_semaphore
  // CHECK: %[[DST_CB:.*]] = ttkernel.get_compile_time_arg_val({{.*}}) : () -> !ttkernel.cb
  // CHECK: %[[SRC_CB:.*]] = ttkernel.get_compile_time_arg_val({{.*}}) : () -> !ttkernel.cb
  //
  // wait_front on the src CB happens outside the collector branch (every
  // core in the group consumes one src-CB element).
  // CHECK: ttkernel.cb_wait_front(%[[SRC_CB]], %{{.*}})
  //
  // isCollector test against (0, 0); these use my_logical_y/x (this core's
  // own location), distinct from the gather-loop source coords below.
  // CHECK: ttkernel.my_logical_y
  // CHECK: ttkernel.my_logical_x
  // CHECK: scf.if %{{.*}} {

  // ---- Collector branch ----
  // Reserve the dst CB (only the collector does this; option (c)).
  // CHECK: ttkernel.cb_reserve_back(%[[DST_CB]], %{{.*}})
  //
  // Wait for the other (group_volume - 1) sources, then reset the counter.
  // CHECK: ttkernel.experimental::semaphore_wait
  // CHECK: ttkernel.noc_semaphore_set
  //
  // Gather loop: each iteration issues a cross-core dma_read whose virtual
  // source coords come from the loop induction variable (via
  // convert_logical_x_to_translated(%[[SX]])), NOT from my_logical_x. This
  // proves the srcCore operand was carried end-to-end through
  // LowerDMAToFullyIndexedForm and the D2MToTTKernel cross-core path.
  // CHECK: scf.for %[[SX:.*]] = %{{.*}} to %{{.*}} step %{{.*}} {
  // CHECK:   ttkernel.experimental::convert_logical_y_to_translated
  // CHECK:   ttkernel.experimental::convert_logical_x_to_translated(%[[SX]])
  // CHECK:   ttkernel.get_noc_addr
  // CHECK:   ttkernel.noc_async_read
  // CHECK:   ttkernel.noc_async_read_barrier
  // CHECK: }
  //
  // Multicast the collectorDone signal to the entire group (group_volume - 1
  // peers receive it; the collector also writes its own copy locally first).
  // CHECK: ttkernel.experimental::get_noc_multicast_addr
  // CHECK: ttkernel.noc_semaphore_set
  // CHECK: ttkernel.noc_semaphore_set_multicast(%[[SEM_DONE]],
  //
  // Push the dst CB (only the collector).
  // CHECK: ttkernel.cb_push_back(%[[DST_CB]], %{{.*}})
  // CHECK: } else {

  // ---- Source branch ----
  // Producers signal the collector and wait for the multicast release.
  // No NoC reads, no dst-CB reserve/push (option (c)).
  // CHECK: ttkernel.experimental::convert_logical_y_to_translated
  // CHECK: ttkernel.experimental::convert_logical_x_to_translated
  // CHECK: ttkernel.get_noc_addr
  // CHECK: ttkernel.noc_semaphore_inc
  // CHECK: ttkernel.experimental::semaphore_wait
  // CHECK: ttkernel.noc_semaphore_set
  // CHECK-NOT: ttkernel.noc_async_read
  // CHECK-NOT: ttkernel.cb_reserve_back
  // CHECK-NOT: ttkernel.cb_push_back
  // CHECK: }
  //
  // pop_front on the src CB after the branch (every core pops, mirroring
  // the wait_front before the branch).
  // CHECK: ttkernel.cb_pop_front(%[[SRC_CB]], %{{.*}})

  // The compute kernel is empty: gather_core lives entirely on the DM
  // thread, so SplitUnifiedThread emits a no-op compute kernel.
  // CHECK-LABEL: func.func private @compute_kernel1
  // CHECK-NEXT: return

  func.func @gather_core_backend(
      %arg0: memref<4x4x2x4x!ttcore.tile<32x32, f32>, #ttcore.shard<16384x4096, 1>, #l1>,
      %arg1: memref<4x4x2x4x!ttcore.tile<32x32, f32>, #ttcore.shard<16384x4096, 1>, #l1>) {
    %src_buf = memref.alloc() {address = 5120 : i64, alignment = 16 : i64} : memref<2x4x!ttcore.tile<32x32, f32>, #ttcore.cb_layout<16384x4096, 2>, #l1>
    %dst_buf = memref.alloc() {address = 6144 : i64, alignment = 16 : i64} : memref<2x4x!ttcore.tile<32x32, f32>, #ttcore.cb_layout<16384x4096, 2>, #l1>

    d2m.generic {block_factors = [1, 1], grid = #ttcore.grid<4x4>,
                 indexing_maps = [#map, #map],
                 iterator_types = [#parallel, #parallel],
                 threads = [#d2m.thread<unified>]}
        ins(%arg0 : memref<4x4x2x4x!ttcore.tile<32x32, f32>, #ttcore.shard<16384x4096, 1>, #l1>)
        outs(%arg1 : memref<4x4x2x4x!ttcore.tile<32x32, f32>, #ttcore.shard<16384x4096, 1>, #l1>)
        additionalArgs(%src_buf, %dst_buf : memref<2x4x!ttcore.tile<32x32, f32>, #ttcore.cb_layout<16384x4096, 2>, #l1>, memref<2x4x!ttcore.tile<32x32, f32>, #ttcore.cb_layout<16384x4096, 2>, #l1>) {
    ^unified0:
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %c4 = arith.constant 4 : index
      d2m.gather_core %src_buf into %dst_buf
        group [%c0, %c0] shape [%c1, %c4] collector [%c0, %c0]
        : memref<2x4x!ttcore.tile<32x32, f32>, #ttcore.cb_layout<16384x4096, 2>, #l1>,
          memref<2x4x!ttcore.tile<32x32, f32>, #ttcore.cb_layout<16384x4096, 2>, #l1>
    }
    memref.dealloc %src_buf : memref<2x4x!ttcore.tile<32x32, f32>, #ttcore.cb_layout<16384x4096, 2>, #l1>
    memref.dealloc %dst_buf : memref<2x4x!ttcore.tile<32x32, f32>, #ttcore.cb_layout<16384x4096, 2>, #l1>
    return
  }
}
