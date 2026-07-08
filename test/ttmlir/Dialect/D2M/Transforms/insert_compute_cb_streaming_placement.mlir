// RUN: ttmlir-opt --ttcore-register-device --d2m-insert-compute-cb %s | FileCheck %s

// Placement regression for a single-scope streaming kernel (vector-add shape).
// The datamovement thread loads x/y and stores the output once per loop
// iteration, so the compute-side CB protocol must sit ENTIRELY INSIDE the
// streaming loop -- one reserve/wait/push/pop per iteration.
//
// The output CB's collapse_shape view is loop-invariant and therefore hoisted
// ABOVE the streaming loop. The regression this guards: the sync ops were
// anchored on that hoisted view (loop depth 0) and got bracketed OUTSIDE the
// loop -- reserve/push/pop above/below it -- which deadlocks for >1 iteration.

#dram = #ttcore.memory_space<dram>
#dst = #ttcore.memory_space<dst>
#l1 = #ttcore.memory_space<l1>
#system_desc = #ttcore.system_desc<[{role = host, target_triple = "x86_64-pc-linux-gnu"}], [{arch = <wormhole_b0>, grid = 8x8, coord_translation_offsets = 18x18, l1_size = 1499136, num_dram_channels = 12, dram_channel_size = 1073741824, noc_l1_address_align_bytes = 16, pcie_address_align_bytes = 32, noc_dram_address_align_bytes = 32, l1_unreserved_base = 1024, erisc_l1_unreserved_base = 1024, dram_unreserved_base = 1024, dram_unreserved_end = 1073741824, supported_data_types = [<f32>, <f16>, <bf16>, <bfp_f8>, <bfp_bf8>, <bfp_f4>, <bfp_bf4>, <bfp_f2>, <bfp_bf2>, <u32>, <u16>, <u8>, <si32>], supported_tile_sizes = [ 4x16,  16x16,  32x16,  4x32,  16x32,  32x32], dst_physical_size_tiles = 16, num_cbs = 32, num_compute_threads = 1, num_datamovement_threads = 2, dram_grid = 1x12, dram_bank_to_logical_worker_noc0 = [(7, 3), (0, 0), (3, 0), (4, 0), (0, 4), (7, 7), (1, 4), (6, 4), (5, 4), (2, 6), (3, 4), (4, 4)], dram_bank_to_logical_worker_noc1 = [(7, 3), (0, 0), (3, 0), (4, 0), (0, 4), (7, 7), (1, 4), (6, 4), (5, 4), (2, 6), (3, 4), (4, 4)]}], [0], [1 : i32], [ 0x0x0x0]>

module attributes {ttcore.system_desc = #system_desc} {
  ttcore.device @default_device = <workerGrid = #ttcore.grid<8x8, virt_to_physical_map = (d0, d1) -> (0, d0, d1), physical_to_virt_map = (d0, d1) -> (0, d0, d1)>, dramGrid = #ttcore.grid<1x12>, l1Map = (d0, d1, d2)[s0] -> (0, d0, d1, d2 + s0), dramMap = (d0, d1, d2)[s0, s1, s2, s3, s4, s5, s6] -> (0, 0, (((d0 * s1) * (s2 * (s3 * s6)) + d1 * (s2 * (s3 * s6)) + d2) floordiv s4) mod 12, ((((d0 * s1) * (s2 * (s3 * s6)) + d1 * (s2 * (s3 * s6)) + d2) floordiv s4) floordiv 12) * s4 + ((d0 * s1) * (s2 * (s3 * s6)) + d1 * (s2 * (s3 * s6)) + d2) mod s4 + s5), meshShape = , chipIds = [0]>

  // CHECK-LABEL: func.func @add_streaming
  func.func @add_streaming(
      %x_dram: memref<1x1x1x1x!ttcore.tile<32x32, bf16>, #ttcore.interleaved<2048x2048>, #dram>,
      %y_dram: memref<1x1x1x1x!ttcore.tile<32x32, bf16>, #ttcore.interleaved<2048x2048>, #dram>,
      %out_dram: memref<1x1x1x1x!ttcore.tile<32x32, bf16>, #ttcore.interleaved<2048x2048>, #dram>) attributes {tt.function_type = "forward_device"} {
    %x_cb = memref.alloc() {address = 1024 : i64, alignment = 16 : i64, d2m.synchronized_buffer = 2 : i32} : memref<1x1x!ttcore.tile<32x32, bf16>, #ttcore.cb_layout<2048x2048, 2>, #l1>
    %y_cb = memref.alloc() {address = 3072 : i64, alignment = 16 : i64, d2m.synchronized_buffer = 2 : i32} : memref<1x1x!ttcore.tile<32x32, bf16>, #ttcore.cb_layout<2048x2048, 2>, #l1>
    %out_cb = memref.alloc() {address = 5120 : i64, alignment = 16 : i64, d2m.synchronized_buffer = 2 : i32} : memref<1x1x!ttcore.tile<32x32, bf16>, #ttcore.cb_layout<2048x2048, 2>, #l1>
    // Operand indices (== CB ports): x_cb = 3, y_cb = 4, out_cb = 5.
    d2m.generic {block_factors = [], grid = #ttcore.grid<8x8>, indexing_maps = [], iterator_types = [], threads = [#d2m.thread<datamovement>, #d2m.thread<compute>]}
        ins(%x_dram, %y_dram : memref<1x1x1x1x!ttcore.tile<32x32, bf16>, #ttcore.interleaved<2048x2048>, #dram>, memref<1x1x1x1x!ttcore.tile<32x32, bf16>, #ttcore.interleaved<2048x2048>, #dram>)
        outs(%out_dram : memref<1x1x1x1x!ttcore.tile<32x32, bf16>, #ttcore.interleaved<2048x2048>, #dram>)
        additionalArgs(%x_cb, %y_cb, %out_cb : memref<1x1x!ttcore.tile<32x32, bf16>, #ttcore.cb_layout<2048x2048, 2>, #l1>, memref<1x1x!ttcore.tile<32x32, bf16>, #ttcore.cb_layout<2048x2048, 2>, #l1>, memref<1x1x!ttcore.tile<32x32, bf16>, #ttcore.cb_layout<2048x2048, 2>, #l1>)
        {
      // Datamovement thread: stream x/y in and the output out, once per iter.
      %xcb = d2m.get_cb(3) resolution_stage = compile : <memref<1x1x!ttcore.tile<32x32, bf16>, #ttcore.cb_layout<2048x2048, 2>, #l1>>
      %ycb = d2m.get_cb(4) resolution_stage = compile : <memref<1x1x!ttcore.tile<32x32, bf16>, #ttcore.cb_layout<2048x2048, 2>, #l1>>
      %ocb = d2m.get_cb(5) resolution_stage = compile : <memref<1x1x!ttcore.tile<32x32, bf16>, #ttcore.cb_layout<2048x2048, 2>, #l1>>
      %c0 = arith.constant 0 : index
      %c0_i32 = arith.constant 0 : i32
      %c1_i32 = arith.constant 1 : i32
      %c4_i32 = arith.constant 4 : i32
      scf.for %i = %c0_i32 to %c4_i32 step %c1_i32 : i32 {
        d2m.remote_load %x_dram[%c0, %c0] into %xcb : memref<1x1x1x1x!ttcore.tile<32x32, bf16>, #ttcore.interleaved<2048x2048>, #dram> into !d2m.cb<memref<1x1x!ttcore.tile<32x32, bf16>, #ttcore.cb_layout<2048x2048, 2>, #l1>>
        d2m.remote_load %y_dram[%c0, %c0] into %ycb : memref<1x1x1x1x!ttcore.tile<32x32, bf16>, #ttcore.interleaved<2048x2048>, #dram> into !d2m.cb<memref<1x1x!ttcore.tile<32x32, bf16>, #ttcore.cb_layout<2048x2048, 2>, #l1>>
        d2m.remote_store %out_dram[%c0, %c0] from %ocb : memref<1x1x1x1x!ttcore.tile<32x32, bf16>, #ttcore.interleaved<2048x2048>, #dram> from !d2m.cb<memref<1x1x!ttcore.tile<32x32, bf16>, #ttcore.cb_layout<2048x2048, 2>, #l1>>
      }
    }, {
      // Compute thread. The output view is hoisted above the loop; the sync must
      // still land inside it.
      // CHECK: }, {
      // CHECK: d2m.get_cb
      // COM: The killer discriminator: NO sync op precedes the streaming loop.
      // COM: The bug hoisted reserve (and the output view) above the loop.
      // CHECK-NOT: d2m.reserve
      // CHECK-NOT: d2m.wait
      // CHECK: scf.for
      // COM: All six sync ops land inside the loop body (order is not pinned:
      // COM: wait/pop bracket each load tightly, reserve/push bracket the store).
      // CHECK-DAG: d2m.reserve
      // CHECK-DAG: d2m.wait
      // CHECK-DAG: d2m.wait
      // CHECK-DAG: d2m.push
      // CHECK-DAG: d2m.pop
      // CHECK-DAG: d2m.pop
      %c0 = arith.constant 0 : index
      %c0_i32 = arith.constant 0 : i32
      %c1_i32 = arith.constant 1 : i32
      %c4_i32 = arith.constant 4 : i32
      %out_view = memref.collapse_shape %out_cb [[0, 1]] : memref<1x1x!ttcore.tile<32x32, bf16>, #ttcore.cb_layout<2048x2048, 2>, #l1> into memref<1x!ttcore.tile<32x32, bf16>, #l1>
      scf.for %i = %c0_i32 to %c4_i32 step %c1_i32 : i32 {
        %x_view = memref.collapse_shape %x_cb [[0, 1]] : memref<1x1x!ttcore.tile<32x32, bf16>, #ttcore.cb_layout<2048x2048, 2>, #l1> into memref<1x!ttcore.tile<32x32, bf16>, #l1>
        %y_view = memref.collapse_shape %y_cb [[0, 1]] : memref<1x1x!ttcore.tile<32x32, bf16>, #ttcore.cb_layout<2048x2048, 2>, #l1> into memref<1x!ttcore.tile<32x32, bf16>, #l1>
        %dst = d2m.acquire_dst() : memref<8x!ttcore.tile<32x32, bf16>, #dst>
        %vx = memref.load %x_view[%c0] : memref<1x!ttcore.tile<32x32, bf16>, #l1>
        %vy = memref.load %y_view[%c0] : memref<1x!ttcore.tile<32x32, bf16>, #l1>
        %r = "d2m.tile_add"(%vx, %vy) : (!ttcore.tile<32x32, bf16>, !ttcore.tile<32x32, bf16>) -> !ttcore.tile<32x32, bf16>
        memref.store %r, %dst[%c0] : memref<8x!ttcore.tile<32x32, bf16>, #dst>
        %rd = memref.load %dst[%c0] : memref<8x!ttcore.tile<32x32, bf16>, #dst>
        memref.store %rd, %out_view[%c0] : memref<1x!ttcore.tile<32x32, bf16>, #l1>
      }
    }
    memref.dealloc %x_cb : memref<1x1x!ttcore.tile<32x32, bf16>, #ttcore.cb_layout<2048x2048, 2>, #l1>
    memref.dealloc %y_cb : memref<1x1x!ttcore.tile<32x32, bf16>, #ttcore.cb_layout<2048x2048, 2>, #l1>
    memref.dealloc %out_cb : memref<1x1x!ttcore.tile<32x32, bf16>, #ttcore.cb_layout<2048x2048, 2>, #l1>
    return
  }
}
