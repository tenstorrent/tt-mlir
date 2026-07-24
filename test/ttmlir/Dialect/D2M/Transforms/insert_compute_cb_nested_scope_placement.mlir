// RUN: ttmlir-opt --ttcore-register-device --d2m-insert-compute-cb %s | FileCheck %s

// Placement regression for a kernel whose CBs live in DIFFERENT loop scopes
// (persistent matmul shape). The A/B operands stream in once per K-iteration,
// but the accumulator/output is transferred once per output tile (by the outer
// persistent loop's remote_store). So each CB's compute-side sync must follow
// ITS OWN datamovement cadence, not one region-wide bracket:
//
//   * A, B  (remote_load in the K-loop)      -> wait/pop INSIDE the K-loop.
//   * accumulator (remote_store in the outer -> reserve BEFORE the K-loop and
//     persistent loop)                          push AFTER it, once per tile.
//
// The regression this guards: the accumulator was anchored on its in-K-loop
// store and got reserved/pushed per K-iteration (K pushes vs one DMA pop ->
// deadlock), and/or A/B waits were hoisted to the persistent-loop scope.

#dram = #ttcore.memory_space<dram>
#dst = #ttcore.memory_space<dst>
#l1 = #ttcore.memory_space<l1>
#system_desc = #ttcore.system_desc<[{role = host, target_triple = "x86_64-pc-linux-gnu"}], [{arch = <wormhole_b0>, grid = 8x8, coord_translation_offsets = 18x18, l1_size = 1499136, num_dram_channels = 12, dram_channel_size = 1073741824, noc_l1_address_align_bytes = 16, pcie_address_align_bytes = 32, noc_dram_address_align_bytes = 32, l1_unreserved_base = 1024, erisc_l1_unreserved_base = 1024, dram_unreserved_base = 1024, dram_unreserved_end = 1073741824, supported_data_types = [<f32>, <f16>, <bf16>, <bfp_f8>, <bfp_bf8>, <bfp_f4>, <bfp_bf4>, <bfp_f2>, <bfp_bf2>, <u32>, <u16>, <u8>, <si32>], supported_tile_sizes = [ 4x16,  16x16,  32x16,  4x32,  16x32,  32x32], dst_physical_size_tiles = 16, num_cbs = 32, num_compute_threads = 1, num_datamovement_threads = 2, dram_grid = 1x12, dram_bank_to_logical_worker_noc0 = [(7, 3), (0, 0), (3, 0), (4, 0), (0, 4), (7, 7), (1, 4), (6, 4), (5, 4), (2, 6), (3, 4), (4, 4)], dram_bank_to_logical_worker_noc1 = [(7, 3), (0, 0), (3, 0), (4, 0), (0, 4), (7, 7), (1, 4), (6, 4), (5, 4), (2, 6), (3, 4), (4, 4)]}], [0], [1 : i32], [ 0x0x0x0]>

module attributes {ttcore.system_desc = #system_desc} {
  ttcore.device @default_device = <workerGrid = #ttcore.grid<8x8, virt_to_physical_map = (d0, d1) -> (0, d0, d1), physical_to_virt_map = (d0, d1) -> (0, d0, d1)>, dramGrid = #ttcore.grid<1x12>, l1Map = (d0, d1, d2)[s0] -> (0, d0, d1, d2 + s0), dramMap = (d0, d1, d2)[s0, s1, s2, s3, s4, s5, s6] -> (0, 0, (((d0 * s1) * (s2 * (s3 * s6)) + d1 * (s2 * (s3 * s6)) + d2) floordiv s4) mod 12, ((((d0 * s1) * (s2 * (s3 * s6)) + d1 * (s2 * (s3 * s6)) + d2) floordiv s4) floordiv 12) * s4 + ((d0 * s1) * (s2 * (s3 * s6)) + d1 * (s2 * (s3 * s6)) + d2) mod s4 + s5), meshShape = , chipIds = [0]>

  // CHECK-LABEL: func.func @matmul_nested_scope
  func.func @matmul_nested_scope(
      %a_dram: memref<1x1x1x1x!ttcore.tile<32x32, f32>, #ttcore.interleaved<4096x4096>, #dram>,
      %b_dram: memref<1x1x1x1x!ttcore.tile<32x32, f32>, #ttcore.interleaved<4096x4096>, #dram>,
      %c_dram: memref<1x1x1x1x!ttcore.tile<32x32, f32>, #ttcore.interleaved<4096x4096>, #dram>) attributes {tt.function_type = "forward_device"} {
    %acc_cb = memref.alloc() {address = 1024 : i64, alignment = 16 : i64, d2m.synchronized_buffer = 2 : i32} : memref<1x1x!ttcore.tile<32x32, f32>, #ttcore.cb_layout<4096x4096, 2>, #l1>
    %a_cb = memref.alloc() {address = 5120 : i64, alignment = 16 : i64, d2m.synchronized_buffer = 2 : i32} : memref<1x1x!ttcore.tile<32x32, f32>, #ttcore.cb_layout<4096x4096, 2>, #l1>
    %b_cb = memref.alloc() {address = 9216 : i64, alignment = 16 : i64, d2m.synchronized_buffer = 2 : i32} : memref<1x1x!ttcore.tile<32x32, f32>, #ttcore.cb_layout<4096x4096, 2>, #l1>
    // Operand indices (== CB ports): acc_cb = 3, a_cb = 4, b_cb = 5.
    d2m.generic {block_factors = [], grid = #ttcore.grid<1x1>, indexing_maps = [], iterator_types = [], threads = [#d2m.thread<datamovement>, #d2m.thread<compute>]}
        ins(%a_dram, %b_dram : memref<1x1x1x1x!ttcore.tile<32x32, f32>, #ttcore.interleaved<4096x4096>, #dram>, memref<1x1x1x1x!ttcore.tile<32x32, f32>, #ttcore.interleaved<4096x4096>, #dram>)
        outs(%c_dram : memref<1x1x1x1x!ttcore.tile<32x32, f32>, #ttcore.interleaved<4096x4096>, #dram>)
        additionalArgs(%acc_cb, %a_cb, %b_cb : memref<1x1x!ttcore.tile<32x32, f32>, #ttcore.cb_layout<4096x4096, 2>, #l1>, memref<1x1x!ttcore.tile<32x32, f32>, #ttcore.cb_layout<4096x4096, 2>, #l1>, memref<1x1x!ttcore.tile<32x32, f32>, #ttcore.cb_layout<4096x4096, 2>, #l1>)
        {
      // Datamovement thread: A/B stream in the K-loop; C stores in the outer loop.
      %acccb = d2m.get_cb(3) resolution_stage = compile : <memref<1x1x!ttcore.tile<32x32, f32>, #ttcore.cb_layout<4096x4096, 2>, #l1>>
      %acb = d2m.get_cb(4) resolution_stage = compile : <memref<1x1x!ttcore.tile<32x32, f32>, #ttcore.cb_layout<4096x4096, 2>, #l1>>
      %bcb = d2m.get_cb(5) resolution_stage = compile : <memref<1x1x!ttcore.tile<32x32, f32>, #ttcore.cb_layout<4096x4096, 2>, #l1>>
      %c0 = arith.constant 0 : index
      %c0_i32 = arith.constant 0 : i32
      %c1_i32 = arith.constant 1 : i32
      %c2_i32 = arith.constant 2 : i32
      %c4_i32 = arith.constant 4 : i32
      scf.for %p = %c0_i32 to %c4_i32 step %c1_i32 : i32 {
        scf.for %k = %c0_i32 to %c2_i32 step %c1_i32 : i32 {
          d2m.remote_load %a_dram[%c0, %c0] into %acb : memref<1x1x1x1x!ttcore.tile<32x32, f32>, #ttcore.interleaved<4096x4096>, #dram> into !d2m.cb<memref<1x1x!ttcore.tile<32x32, f32>, #ttcore.cb_layout<4096x4096, 2>, #l1>>
          d2m.remote_load %b_dram[%c0, %c0] into %bcb : memref<1x1x1x1x!ttcore.tile<32x32, f32>, #ttcore.interleaved<4096x4096>, #dram> into !d2m.cb<memref<1x1x!ttcore.tile<32x32, f32>, #ttcore.cb_layout<4096x4096, 2>, #l1>>
        }
        d2m.remote_store %c_dram[%c0, %c0] from %acccb : memref<1x1x1x1x!ttcore.tile<32x32, f32>, #ttcore.interleaved<4096x4096>, #dram> from !d2m.cb<memref<1x1x!ttcore.tile<32x32, f32>, #ttcore.cb_layout<4096x4096, 2>, #l1>>
      }
    }, {
      // Compute thread. The accumulator view is hoisted above the loops.
      // CHECK: }, {
      // CHECK: d2m.get_cb
      // COM: nothing is synchronized before the persistent loop.
      // CHECK-NOT: d2m.reserve
      // CHECK-NOT: d2m.wait
      // CHECK: scf.for
      // COM: the accumulator is reserved between the persistent loop and the
      // COM: K-loop -- i.e. once per output tile, bracketing the K-loop.
      // CHECK: d2m.reserve
      // COM: and NO A/B wait leaks up to the persistent-loop scope.
      // CHECK-NOT: d2m.wait
      // CHECK: scf.for
      // COM: A and B are waited/popped INSIDE the K-loop (order not pinned:
      // COM: each wait/pop brackets its own load).
      // CHECK-DAG: d2m.wait
      // CHECK-DAG: d2m.wait
      // CHECK-DAG: d2m.tile_matmul
      // CHECK-DAG: d2m.pop
      // CHECK-DAG: d2m.pop
      // COM: the accumulator push follows the whole K-loop body -- once per
      // COM: output tile, not once per K-iteration.
      // CHECK: d2m.push
      %c0 = arith.constant 0 : index
      %c0_i32 = arith.constant 0 : i32
      %c1_i32 = arith.constant 1 : i32
      %c2_i32 = arith.constant 2 : i32
      %c4_i32 = arith.constant 4 : i32
      %acc_view = memref.collapse_shape %acc_cb [[0, 1]] : memref<1x1x!ttcore.tile<32x32, f32>, #ttcore.cb_layout<4096x4096, 2>, #l1> into memref<1x!ttcore.tile<32x32, f32>, #l1>
      scf.for %p = %c0_i32 to %c4_i32 step %c1_i32 : i32 {
        scf.for %k = %c0_i32 to %c2_i32 step %c1_i32 : i32 {
          %a_view = memref.collapse_shape %a_cb [[0, 1]] : memref<1x1x!ttcore.tile<32x32, f32>, #ttcore.cb_layout<4096x4096, 2>, #l1> into memref<1x!ttcore.tile<32x32, f32>, #l1>
          %b_view = memref.collapse_shape %b_cb [[0, 1]] : memref<1x1x!ttcore.tile<32x32, f32>, #ttcore.cb_layout<4096x4096, 2>, #l1> into memref<1x!ttcore.tile<32x32, f32>, #l1>
          %dst = d2m.acquire_dst() : memref<4x!ttcore.tile<32x32, f32>, #dst>
          %va = memref.load %a_view[%c0] : memref<1x!ttcore.tile<32x32, f32>, #l1>
          %vb = memref.load %b_view[%c0] : memref<1x!ttcore.tile<32x32, f32>, #l1>
          %vd = memref.load %dst[%c0] : memref<4x!ttcore.tile<32x32, f32>, #dst>
          %r = "d2m.tile_matmul"(%va, %vb, %vd) : (!ttcore.tile<32x32, f32>, !ttcore.tile<32x32, f32>, !ttcore.tile<32x32, f32>) -> !ttcore.tile<32x32, f32>
          memref.store %r, %dst[%c0] : memref<4x!ttcore.tile<32x32, f32>, #dst>
          %rd = memref.load %dst[%c0] : memref<4x!ttcore.tile<32x32, f32>, #dst>
          memref.store %rd, %acc_view[%c0] : memref<1x!ttcore.tile<32x32, f32>, #l1>
        }
      }
    }
    memref.dealloc %acc_cb : memref<1x1x!ttcore.tile<32x32, f32>, #ttcore.cb_layout<4096x4096, 2>, #l1>
    memref.dealloc %a_cb : memref<1x1x!ttcore.tile<32x32, f32>, #ttcore.cb_layout<4096x4096, 2>, #l1>
    memref.dealloc %b_cb : memref<1x1x!ttcore.tile<32x32, f32>, #ttcore.cb_layout<4096x4096, 2>, #l1>
    return
  }
}
