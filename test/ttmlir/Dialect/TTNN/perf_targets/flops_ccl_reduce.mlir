// RUN: rm -rf %t.dir && mkdir -p %t.dir
// RUN: ttmlir-opt --ttnn-collect-perf-metrics="ttnn-perf-metrics-output-file=%t.dir/out.json ttnn-perf-metrics-verbose-output-enabled=true" %s -o /dev/null
// RUN: cat %t.dir/out.json | FileCheck %s

// FLOP accounting for the CCL reductions. Their combine step is the same
// arithmetic on the same engine as a matmul, so it counts against the same peak.
//
// A collective over D devices along its cluster_axis combines D values per
// element (D-1 adds), and a ring spreads that evenly over the D chips, so the
// per-chip cost is numel(input) * (D-1) / D. all_reduce and reduce_scatter come
// out identical, because the all_gather half of a ring all_reduce moves data
// without doing arithmetic.
//
// The mesh here is 1x4, so cluster_axis 1 gives D=4 and cluster_axis 0 gives
// D=1. Input to every collective is [1,1,256,128] = 32,768 scalars.
//
//   matmul         64x64x64                    -> 2*64*64*64      =   524,288
//   all_reduce     cluster_axis 1, D=4         -> 32768 * 3/4     =    24,576
//   reduce_scatter cluster_axis 1, D=4         -> same as above   =    24,576
//   all_reduce     cluster_axis 0, D=1         -> no-op, NOT counted
//   all_gather     no reduce_type              -> NOT counted
//   total                                                         =   573,440
//
// No op sets a compute_config, so everything is accounted at the HiFi4 default
// and peak_flops_per_sec must equal the hifi4 peak: 64 * 1e9 * 2*32^3 / 64.

#dram = #ttnn.buffer_type<dram>
#system_desc = #ttcore.system_desc<[{role = host, target_triple = "x86_64-pc-linux"}], [{arch = <wormhole_b0>, grid = 8x8, coord_translation_offsets = 18x18, l1_size = 1499136, num_dram_channels = 12, dram_channel_size = 1073741824, noc_l1_address_align_bytes = 16, pcie_address_align_bytes = 32, noc_dram_address_align_bytes = 32, l1_unreserved_base = 103712, erisc_l1_unreserved_base = 98304, dram_unreserved_base = 1920032, dram_unreserved_end = 1073119552, supported_data_types = [<f32>, <f16>, <bf16>, <bfp_f8>, <bfp_bf8>, <bfp_f4>, <bfp_bf4>, <bfp_f2>, <bfp_bf2>, <u32>, <u16>, <u8>, <si32>], supported_tile_sizes = [ 4x16, 16x16, 32x16, 4x32, 16x32, 32x32], dst_physical_size_tiles = 16, num_cbs = 64, num_compute_threads = 1, num_datamovement_threads = 2, dram_grid = 1x12, dram_bank_to_logical_worker_noc0 = [(7, 3), (0, 0), (3, 0), (4, 0), (0, 4), (7, 7), (1, 4), (6, 4), (5, 4), (2, 6), (3, 4), (4, 4)], dram_bank_to_logical_worker_noc1 = [(7, 3), (0, 0), (3, 0), (4, 0), (0, 4), (7, 7), (1, 4), (6, 4), (5, 4), (2, 6), (3, 4), (4, 4)]}, {arch = <wormhole_b0>, grid = 8x8, coord_translation_offsets = 18x18, l1_size = 1499136, num_dram_channels = 12, dram_channel_size = 1073741824, noc_l1_address_align_bytes = 16, pcie_address_align_bytes = 32, noc_dram_address_align_bytes = 32, l1_unreserved_base = 103712, erisc_l1_unreserved_base = 98304, dram_unreserved_base = 1920032, dram_unreserved_end = 1073119552, supported_data_types = [<f32>, <f16>, <bf16>, <bfp_f8>, <bfp_bf8>, <bfp_f4>, <bfp_bf4>, <bfp_f2>, <bfp_bf2>, <u32>, <u16>, <u8>, <si32>], supported_tile_sizes = [ 4x16, 16x16, 32x16, 4x32, 16x32, 32x32], dst_physical_size_tiles = 16, num_cbs = 64, num_compute_threads = 1, num_datamovement_threads = 2, dram_grid = 1x12, dram_bank_to_logical_worker_noc0 = [(7, 3), (0, 0), (3, 0), (4, 0), (0, 4), (7, 7), (1, 4), (6, 4), (5, 4), (2, 6), (3, 4), (4, 4)], dram_bank_to_logical_worker_noc1 = [(7, 3), (0, 0), (3, 0), (4, 0), (0, 4), (7, 7), (1, 4), (6, 4), (5, 4), (2, 6), (3, 4), (4, 4)]}, {arch = <wormhole_b0>, grid = 8x8, coord_translation_offsets = 18x18, l1_size = 1499136, num_dram_channels = 12, dram_channel_size = 1073741824, noc_l1_address_align_bytes = 16, pcie_address_align_bytes = 32, noc_dram_address_align_bytes = 32, l1_unreserved_base = 103712, erisc_l1_unreserved_base = 98304, dram_unreserved_base = 1920032, dram_unreserved_end = 1073119552, supported_data_types = [<f32>, <f16>, <bf16>, <bfp_f8>, <bfp_bf8>, <bfp_f4>, <bfp_bf4>, <bfp_f2>, <bfp_bf2>, <u32>, <u16>, <u8>, <si32>], supported_tile_sizes = [ 4x16, 16x16, 32x16, 4x32, 16x32, 32x32], dst_physical_size_tiles = 16, num_cbs = 64, num_compute_threads = 1, num_datamovement_threads = 2, dram_grid = 1x12, dram_bank_to_logical_worker_noc0 = [(7, 3), (0, 0), (3, 0), (4, 0), (0, 4), (7, 7), (1, 4), (6, 4), (5, 4), (2, 6), (3, 4), (4, 4)], dram_bank_to_logical_worker_noc1 = [(7, 3), (0, 0), (3, 0), (4, 0), (0, 4), (7, 7), (1, 4), (6, 4), (5, 4), (2, 6), (3, 4), (4, 4)]}, {arch = <wormhole_b0>, grid = 8x8, coord_translation_offsets = 18x18, l1_size = 1499136, num_dram_channels = 12, dram_channel_size = 1073741824, noc_l1_address_align_bytes = 16, pcie_address_align_bytes = 32, noc_dram_address_align_bytes = 32, l1_unreserved_base = 103712, erisc_l1_unreserved_base = 98304, dram_unreserved_base = 1920032, dram_unreserved_end = 1073119552, supported_data_types = [<f32>, <f16>, <bf16>, <bfp_f8>, <bfp_bf8>, <bfp_f4>, <bfp_bf4>, <bfp_f2>, <bfp_bf2>, <u32>, <u16>, <u8>, <si32>], supported_tile_sizes = [ 4x16, 16x16, 32x16, 4x32, 16x32, 32x32], dst_physical_size_tiles = 16, num_cbs = 64, num_compute_threads = 1, num_datamovement_threads = 2, dram_grid = 1x12, dram_bank_to_logical_worker_noc0 = [(7, 3), (0, 0), (3, 0), (4, 0), (0, 4), (7, 7), (1, 4), (6, 4), (5, 4), (2, 6), (3, 4), (4, 4)], dram_bank_to_logical_worker_noc1 = [(7, 3), (0, 0), (3, 0), (4, 0), (0, 4), (7, 7), (1, 4), (6, 4), (5, 4), (2, 6), (3, 4), (4, 4)]}], [0, 1, 2, 3], [1 : i32, 1 : i32, 1 : i32, 1 : i32], [ 0x0x0x0,  0x0x0x1,  0x0x0x2,  0x0x0x3]>

// [1,1,256,128] -> 256 rows x 128 cols = 8x4 tiles
#full = #ttnn.ttnn_layout<(d0, d1, d2, d3) -> (d0 * 256 + d1 * 256 + d2, d3), <1x1>, memref<8x4x!ttcore.tile<32x32, bf16>, #dram>, <interleaved>>
// [1,1,256,32] scattered along dim 3 -> 8x1 tiles
#scat = #ttnn.ttnn_layout<(d0, d1, d2, d3) -> (d0 * 256 + d1 * 256 + d2, d3), <1x1>, memref<8x1x!ttcore.tile<32x32, bf16>, #dram>, <interleaved>>
// [1,1,256,512] gathered along dim 3 -> 8x16 tiles
#gath = #ttnn.ttnn_layout<(d0, d1, d2, d3) -> (d0 * 256 + d1 * 256 + d2, d3), <1x1>, memref<8x16x!ttcore.tile<32x32, bf16>, #dram>, <interleaved>>
// [64,64] -> 2x2 tiles
#mm = #ttnn.ttnn_layout<(d0, d1) -> (d0, d1), <1x1>, memref<2x2x!ttcore.tile<32x32, bf16>, #dram>, <interleaved>>

module attributes {ttcore.system_desc = #system_desc} {
ttcore.device @default_device = <workerGrid = #ttcore.grid<8x8, virt_to_physical_map = (d0, d1) -> (0, d0, d1), physical_to_virt_map = (d0, d1, d2) -> (d1, d2)>, dramGrid = #ttcore.grid<1x12>, l1Map = (d0, d1, d2)[s0] -> (0, d0, d1, d2 + s0), dramMap = (d0, d1, d2)[s0, s1, s2, s3, s4, s5, s6] -> (0, 0, (((d0 * s1) * (s2 * (s3 * s6)) + d1 * (s2 * (s3 * s6)) + d2) floordiv s4) mod 12, ((((d0 * s1) * (s2 * (s3 * s6)) + d1 * (s2 * (s3 * s6)) + d2) floordiv s4) floordiv 12) * s4 + ((d0 * s1) * (s2 * (s3 * s6)) + d1 * (s2 * (s3 * s6)) + d2) mod s4 + s5), meshShape = 1x4, chipIds = [0, 1, 2, 3]>
  func.func @forward(
      %a: tensor<64x64xbf16, #mm>            {ttcore.argument_type = #ttcore.argument_type<input>},
      %b: tensor<64x64xbf16, #mm>            {ttcore.argument_type = #ttcore.argument_type<parameter>},
      %x: tensor<1x1x256x128xbf16, #full>    {ttcore.argument_type = #ttcore.argument_type<input>}
  ) -> tensor<1x1x256x512xbf16, #gath>
      attributes {tt.function_type = "forward_device"} {
    // Ordinary matrix-engine work, so the CCL flops below fold into the same total.
    %0 = "ttnn.matmul"(%a, %b) <{transpose_a = false, transpose_b = false}>
        : (tensor<64x64xbf16, #mm>, tensor<64x64xbf16, #mm>) -> tensor<64x64xbf16, #mm>

    // D=4 along mesh dim 1: 32768 - 32768/4 = 24,576
    %1 = "ttnn.all_reduce"(%x) <{cluster_axis = 1 : ui32, reduce_type = #ttcore.reduce_type<sum>}>
        : (tensor<1x1x256x128xbf16, #full>) -> tensor<1x1x256x128xbf16, #full>

    // Same per-chip cost as the all_reduce despite the smaller result.
    %2 = "ttnn.reduce_scatter"(%1) <{cluster_axis = 1 : ui32, scatter_dim = 3 : si32, reduce_type = #ttcore.reduce_type<sum>}>
        : (tensor<1x1x256x128xbf16, #full>) -> tensor<1x1x256x32xbf16, #scat>

    // Mesh dim 0 has extent 1, so D=1 and the collective reduces nothing.
    %3 = "ttnn.all_reduce"(%x) <{cluster_axis = 0 : ui32, reduce_type = #ttcore.reduce_type<sum>}>
        : (tensor<1x1x256x128xbf16, #full>) -> tensor<1x1x256x128xbf16, #full>

    // Pure data movement: no reduce_type, so no arithmetic.
    %4 = "ttnn.all_gather"(%3) <{cluster_axis = 1 : ui32, all_gather_dim = 3 : si32}>
        : (tensor<1x1x256x128xbf16, #full>) -> tensor<1x1x256x512xbf16, #gath>

    return %4 : tensor<1x1x256x512xbf16, #gath>
  }
}

// JSON keys are emitted in alphabetical order.
// CHECK:      "flops": {
// CHECK:        "peak_flops_per_sec": 65536000000000
// CHECK:        "per_op": [
// CHECK:          "flops": 524288
// CHECK:          "math_fidelity": "hifi4"
// CHECK:          "operation": "ttnn.matmul"
// CHECK:          "flops": 24576
// CHECK:          "operation": "ttnn.all_reduce"
// CHECK:          "flops": 24576
// CHECK:          "operation": "ttnn.reduce_scatter"
// The D=1 all_reduce and the all_gather are both dropped, so the array ends here.
// CHECK-NOT:      "ttnn.all_gather"
// CHECK:        "total_flops": 573440
