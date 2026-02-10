// RUN: ttmlir-opt --d2m-allocate %s 2>&1 | FileCheck %s

// This test verifies that d2m-allocate correctly updates remote_load ops
// when input view operands are replaced with streams. The remote ops should
// reference the new stream operand, not the original view.
//
// Before the fix, this test would fail with:
//   "'d2m.remote_load' op memref operand must reference one of the parent
//    generic op's operands directly"
//
// This happens because when a stream is inserted to replace a view operand,
// the remote_load still referenced the old view value. The fix ensures the
// remote_load is updated to reference the new stream.

#l1 = #ttcore.memory_space<l1>
#remap4 = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#remap_rhs = affine_map<(d0, d1, d2, d3) -> ((d0 * 128 + d2 * 64 + d1) floordiv 1024, (d1 floordiv 8) mod 8, (d0 * 2 + d2 + d1 floordiv 64) mod 16, d1 mod 8)>
#system_desc = #ttcore.system_desc<[{role = host, target_triple = "x86_64-pc-linux"}], [{arch = <wormhole_b0>, grid = 8x8, coord_translation_offsets = 18x18, l1_size = 1499136, num_dram_channels = 12, dram_channel_size = 1073741824, noc_l1_address_align_bytes = 16, pcie_address_align_bytes = 32, noc_dram_address_align_bytes = 32, l1_unreserved_base = 102656, erisc_l1_unreserved_base = 98304, dram_unreserved_base = 32, dram_unreserved_end = 1073125888, supported_data_types = [<f32>, <f16>, <bf16>, <bfp_f8>, <bfp_bf8>, <bfp_f4>, <bfp_bf4>, <bfp_f2>, <bfp_bf2>, <u32>, <u16>, <u8>, <si32>], supported_tile_sizes = [ 4x16,  16x16,  32x16,  4x32,  16x32,  32x32], dst_physical_size_tiles = 16, num_cbs = 32, num_compute_threads = 1, num_datamovement_threads = 2}], [0], [1 : i32], [ 0x0x0x0]>

module attributes {ttcore.system_desc = #system_desc} {
  ttcore.device @default_device = <workerGrid = #ttcore.grid<8x8, (d0, d1) -> (0, d0, d1)>, l1Map = (d0, d1, d2)[s0] -> (0, d0, d1, d2 + s0), dramMap = (d0, d1, d2)[s0, s1, s2, s3, s4, s5, s6] -> (0, 0, (((d0 * s1) * (s2 * (s3 * s6)) + d1 * (s2 * (s3 * s6)) + d2) floordiv s4) mod 12, ((((d0 * s1) * (s2 * (s3 * s6)) + d1 * (s2 * (s3 * s6)) + d2) floordiv s4) floordiv 12) * s4 + ((d0 * s1) * (s2 * (s3 * s6)) + d1 * (s2 * (s3 * s6)) + d2) mod s4 + s5), meshShape = 1x1, chipIds = [0]>

  // CHECK-LABEL: func.func @test_view_to_stream_remote_load_update
  func.func @test_view_to_stream_remote_load_update() -> memref<1x64x1x1x!ttcore.tile<32x32, f32>, #ttcore.shard<4096x4096, 1>, #l1> {
    // Allocations for LHS (A matrix) and RHS (B matrix) for matmul
    %alloc = memref.alloc() : memref<1x64x1x2x!ttcore.tile<32x32, f32>, #ttcore.shard<8192x4096, 1>, #l1>
    %alloc_2 = memref.alloc() : memref<8x8x16x8x!ttcore.tile<32x32, f32>, #ttcore.shard<32768x4096, 1>, #l1>
    %alloc_4 = memref.alloc() : memref<1x64x1x1x!ttcore.tile<32x32, f32>, #ttcore.shard<4096x4096, 1>, #l1>

    // Create views of the allocations - these will be replaced with streams
    %view_5 = d2m.view_layout %alloc remapping = #remap4 : memref<1x64x1x2x!ttcore.tile<32x32, f32>, #ttcore.shard<8192x4096, 1>, #l1> -> memref<1x64x1x2x!ttcore.tile<32x32, f32>, #ttcore.view<4>, #l1>
    %view_6 = d2m.view_layout %alloc_2 remapping = #remap_rhs : memref<8x8x16x8x!ttcore.tile<32x32, f32>, #ttcore.shard<32768x4096, 1>, #l1> -> memref<64x64x2x1x!ttcore.tile<32x32, f32>, #ttcore.view<4>, #l1>

    // Matmul generic with reduction dimension - triggers stream insertion
    // After d2m-allocate:
    // 1. Streams are inserted: stream_layout(%view_5, ...) and stream_layout(%view_6, ...)
    // 2. Generic op inputs are updated to reference streams instead of views
    // 3. remote_load ops are updated to reference streams (this is what this test verifies)
    //
    // CHECK: %[[LHS_STREAM:.*]] = "d2m.stream_layout"
    // CHECK: %[[RHS_STREAM:.*]] = "d2m.stream_layout"
    // CHECK: d2m.generic
    // CHECK: ins(%[[LHS_STREAM]], %[[RHS_STREAM]] :
    %view_out = d2m.view_layout %alloc_4 remapping = #remap4 : memref<1x64x1x1x!ttcore.tile<32x32, f32>, #ttcore.shard<4096x4096, 1>, #l1> -> memref<1x64x1x1x!ttcore.tile<32x32, f32>, #ttcore.view<4>, #l1>
    d2m.generic {block_factors = [1, 1, 64], grid = #ttcore.grid<1x64, (d0, d1) -> (0, 0, d0 * 8 + d1)>, indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d2)>, affine_map<(d0, d1, d2) -> (d2, d1)>, affine_map<(d0, d1, d2) -> (d0, d1)>], iterator_types = [#ttcore.iterator_type<parallel>, #ttcore.iterator_type<parallel>, #ttcore.iterator_type<reduction>], threads = [#d2m.thread<unified>]}
        ins(%view_5, %view_6 : memref<1x64x1x2x!ttcore.tile<32x32, f32>, #ttcore.view<4>, #l1>, memref<64x64x2x1x!ttcore.tile<32x32, f32>, #ttcore.view<4>, #l1>)
        outs(%view_out : memref<1x64x1x1x!ttcore.tile<32x32, f32>, #ttcore.view<4>, #l1>) {
    ^unified0(%cb0: !d2m.cb<memref<1x2x!ttcore.tile<32x32, f32>, #l1>>, %cb1: !d2m.cb<memref<2x1x!ttcore.tile<32x32, f32>, #l1>>, %cb2: !d2m.cb<memref<1x1x!ttcore.tile<32x32, f32>, #l1>>):
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %bf0 = d2m.get_block_factor(0) : index
      %bf1 = d2m.get_block_factor(1) : index
      %bf2 = d2m.get_block_factor(2) : index
      affine.for %iter0 = 0 to %bf0 {
        affine.for %iter1 = 0 to %bf1 {
          affine.for %iter2 = 0 to %bf2 {
            // Before fix: remote_load still references %view_5, causing verifier error
            // After fix: remote_load is updated to reference %stream
            %buffer_lhs = memref.alloc() : memref<1x2x!ttcore.tile<32x32, f32>, #l1>
            // CHECK: d2m.remote_load %{{.*}} %[[LHS_STREAM]]
            %0 = d2m.remote_load %buffer_lhs %view_5[%iter0, %iter2] mcast[%c0] : memref<1x2x!ttcore.tile<32x32, f32>, #l1>, memref<1x64x1x2x!ttcore.tile<32x32, f32>, #ttcore.view<4>, #l1> -> memref<1x2x!ttcore.tile<32x32, f32>, #l1>
            %buffer_rhs = memref.alloc() : memref<2x1x!ttcore.tile<32x32, f32>, #l1>
            // CHECK: d2m.remote_load %{{.*}} %[[RHS_STREAM]]
            %1 = d2m.remote_load %buffer_rhs %view_6[%iter2, %iter1] mcast[%c1] : memref<2x1x!ttcore.tile<32x32, f32>, #l1>, memref<64x64x2x1x!ttcore.tile<32x32, f32>, #ttcore.view<4>, #l1> -> memref<2x1x!ttcore.tile<32x32, f32>, #l1>
            %buffer_out = memref.alloc() : memref<1x1x!ttcore.tile<32x32, f32>, #l1>
            linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d2)>, affine_map<(d0, d1, d2) -> (d2, d1)>, affine_map<(d0, d1, d2) -> (d0, d1)>], iterator_types = ["parallel", "parallel", "reduction"]} ins(%0, %1 : memref<1x2x!ttcore.tile<32x32, f32>, #l1>, memref<2x1x!ttcore.tile<32x32, f32>, #l1>) outs(%buffer_out : memref<1x1x!ttcore.tile<32x32, f32>, #l1>) {
            ^bb0(%in: !ttcore.tile<32x32, f32>, %in_14: !ttcore.tile<32x32, f32>, %out: !ttcore.tile<32x32, f32>):
              %3 = "d2m.tile_matmul"(%in, %in_14, %out) : (!ttcore.tile<32x32, f32>, !ttcore.tile<32x32, f32>, !ttcore.tile<32x32, f32>) -> !ttcore.tile<32x32, f32>
              linalg.yield %3 : !ttcore.tile<32x32, f32>
            }
            %2 = d2m.remote_store %view_out[%iter0, %iter1] %buffer_out : memref<1x64x1x1x!ttcore.tile<32x32, f32>, #ttcore.view<4>, #l1>, memref<1x1x!ttcore.tile<32x32, f32>, #l1> -> memref<1x64x1x1x!ttcore.tile<32x32, f32>, #ttcore.view<4>, #l1>
          } {d2m.blocking_loop = 2}
        } {d2m.blocking_loop = 1}
      } {d2m.blocking_loop = 0}
    }

    memref.dealloc %alloc : memref<1x64x1x2x!ttcore.tile<32x32, f32>, #ttcore.shard<8192x4096, 1>, #l1>
    memref.dealloc %alloc_2 : memref<8x8x16x8x!ttcore.tile<32x32, f32>, #ttcore.shard<32768x4096, 1>, #l1>
    return %alloc_4 : memref<1x64x1x1x!ttcore.tile<32x32, f32>, #ttcore.shard<4096x4096, 1>, #l1>
  }
}
