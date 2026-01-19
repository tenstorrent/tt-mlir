#l1 = #ttnn.buffer_type<l1>
#system_memory = #ttnn.buffer_type<system_memory>
#system_desc = #ttcore.system_desc<[{role = host, target_triple = "x86_64-pc-linux"}], [{arch = <wormhole_b0>, grid = 8x8, coord_translation_offsets = 18x18, l1_size = 1499136, num_dram_channels = 12, dram_channel_size = 1073741824, noc_l1_address_align_bytes = 16, pcie_address_align_bytes = 32, noc_dram_address_align_bytes = 32, l1_unreserved_base = 102656, erisc_l1_unreserved_base = 98304, dram_unreserved_base = 1920032, dram_unreserved_end = 1073142976, supported_data_types = [<f32>, <f16>, <bf16>, <bfp_f8>, <bfp_bf8>, <bfp_f4>, <bfp_bf4>, <bfp_f2>, <bfp_bf2>, <u32>, <u16>, <u8>, <si32>], supported_tile_sizes = [ 4x16,  16x16,  32x16,  4x32,  16x32,  32x32], dst_physical_size_tiles = 16, num_cbs = 32, num_compute_threads = 1, num_datamovement_threads = 2}, {arch = <wormhole_b0>, grid = 8x8, coord_translation_offsets = 18x18, l1_size = 1499136, num_dram_channels = 12, dram_channel_size = 1073741824, noc_l1_address_align_bytes = 16, pcie_address_align_bytes = 32, noc_dram_address_align_bytes = 32, l1_unreserved_base = 102656, erisc_l1_unreserved_base = 98304, dram_unreserved_base = 1920032, dram_unreserved_end = 1073142976, supported_data_types = [<f32>, <f16>, <bf16>, <bfp_f8>, <bfp_bf8>, <bfp_f4>, <bfp_bf4>, <bfp_f2>, <bfp_bf2>, <u32>, <u16>, <u8>, <si32>], supported_tile_sizes = [ 4x16,  16x16,  32x16,  4x32,  16x32,  32x32], dst_physical_size_tiles = 16, num_cbs = 32, num_compute_threads = 1, num_datamovement_threads = 2}, {arch = <wormhole_b0>, grid = 8x8, coord_translation_offsets = 18x18, l1_size = 1499136, num_dram_channels = 12, dram_channel_size = 1073741824, noc_l1_address_align_bytes = 16, pcie_address_align_bytes = 32, noc_dram_address_align_bytes = 32, l1_unreserved_base = 102656, erisc_l1_unreserved_base = 98304, dram_unreserved_base = 1920032, dram_unreserved_end = 1073142976, supported_data_types = [<f32>, <f16>, <bf16>, <bfp_f8>, <bfp_bf8>, <bfp_f4>, <bfp_bf4>, <bfp_f2>, <bfp_bf2>, <u32>, <u16>, <u8>, <si32>], supported_tile_sizes = [ 4x16,  16x16,  32x16,  4x32,  16x32,  32x32], dst_physical_size_tiles = 16, num_cbs = 32, num_compute_threads = 1, num_datamovement_threads = 2}, {arch = <wormhole_b0>, grid = 8x8, coord_translation_offsets = 18x18, l1_size = 1499136, num_dram_channels = 12, dram_channel_size = 1073741824, noc_l1_address_align_bytes = 16, pcie_address_align_bytes = 32, noc_dram_address_align_bytes = 32, l1_unreserved_base = 102656, erisc_l1_unreserved_base = 98304, dram_unreserved_base = 1920032, dram_unreserved_end = 1073142976, supported_data_types = [<f32>, <f16>, <bf16>, <bfp_f8>, <bfp_bf8>, <bfp_f4>, <bfp_bf4>, <bfp_f2>, <bfp_bf2>, <u32>, <u16>, <u8>, <si32>], supported_tile_sizes = [ 4x16,  16x16,  32x16,  4x32,  16x32,  32x32], dst_physical_size_tiles = 16, num_cbs = 32, num_compute_threads = 1, num_datamovement_threads = 2}, {arch = <wormhole_b0>, grid = 8x8, coord_translation_offsets = 18x18, l1_size = 1499136, num_dram_channels = 12, dram_channel_size = 1073741824, noc_l1_address_align_bytes = 16, pcie_address_align_bytes = 32, noc_dram_address_align_bytes = 32, l1_unreserved_base = 102656, erisc_l1_unreserved_base = 98304, dram_unreserved_base = 1920032, dram_unreserved_end = 1073160096, supported_data_types = [<f32>, <f16>, <bf16>, <bfp_f8>, <bfp_bf8>, <bfp_f4>, <bfp_bf4>, <bfp_f2>, <bfp_bf2>, <u32>, <u16>, <u8>, <si32>], supported_tile_sizes = [ 4x16,  16x16,  32x16,  4x32,  16x32,  32x32], dst_physical_size_tiles = 16, num_cbs = 32, num_compute_threads = 1, num_datamovement_threads = 2}, {arch = <wormhole_b0>, grid = 8x8, coord_translation_offsets = 18x18, l1_size = 1499136, num_dram_channels = 12, dram_channel_size = 1073741824, noc_l1_address_align_bytes = 16, pcie_address_align_bytes = 32, noc_dram_address_align_bytes = 32, l1_unreserved_base = 102656, erisc_l1_unreserved_base = 98304, dram_unreserved_base = 1920032, dram_unreserved_end = 1073160096, supported_data_types = [<f32>, <f16>, <bf16>, <bfp_f8>, <bfp_bf8>, <bfp_f4>, <bfp_bf4>, <bfp_f2>, <bfp_bf2>, <u32>, <u16>, <u8>, <si32>], supported_tile_sizes = [ 4x16,  16x16,  32x16,  4x32,  16x32,  32x32], dst_physical_size_tiles = 16, num_cbs = 32, num_compute_threads = 1, num_datamovement_threads = 2}, {arch = <wormhole_b0>, grid = 8x8, coord_translation_offsets = 18x18, l1_size = 1499136, num_dram_channels = 12, dram_channel_size = 1073741824, noc_l1_address_align_bytes = 16, pcie_address_align_bytes = 32, noc_dram_address_align_bytes = 32, l1_unreserved_base = 102656, erisc_l1_unreserved_base = 98304, dram_unreserved_base = 1920032, dram_unreserved_end = 1073160096, supported_data_types = [<f32>, <f16>, <bf16>, <bfp_f8>, <bfp_bf8>, <bfp_f4>, <bfp_bf4>, <bfp_f2>, <bfp_bf2>, <u32>, <u16>, <u8>, <si32>], supported_tile_sizes = [ 4x16,  16x16,  32x16,  4x32,  16x32,  32x32], dst_physical_size_tiles = 16, num_cbs = 32, num_compute_threads = 1, num_datamovement_threads = 2}, {arch = <wormhole_b0>, grid = 8x8, coord_translation_offsets = 18x18, l1_size = 1499136, num_dram_channels = 12, dram_channel_size = 1073741824, noc_l1_address_align_bytes = 16, pcie_address_align_bytes = 32, noc_dram_address_align_bytes = 32, l1_unreserved_base = 102656, erisc_l1_unreserved_base = 98304, dram_unreserved_base = 1920032, dram_unreserved_end = 1073160096, supported_data_types = [<f32>, <f16>, <bf16>, <bfp_f8>, <bfp_bf8>, <bfp_f4>, <bfp_bf4>, <bfp_f2>, <bfp_bf2>, <u32>, <u16>, <u8>, <si32>], supported_tile_sizes = [ 4x16,  16x16,  32x16,  4x32,  16x32,  32x32], dst_physical_size_tiles = 16, num_cbs = 32, num_compute_threads = 1, num_datamovement_threads = 2}], [0, 1, 2, 3, 4, 5, 6, 7], [1 : i32, 1 : i32, 1 : i32, 1 : i32, 0 : i32, 0 : i32, 0 : i32, 0 : i32], [ 0x0x0x0]>

// Layouts matching metal_p2p.mlir:
// - Full tensor: 256x768 bf16
// - Mesh shard: 128x192 bf16 (after 2x4 device sharding)
// - L1 shard: 4x6 tiles = 24 tiles = 49152 bytes (height_sharded on single core)
#ttnn_layout_host = #ttnn.ttnn_layout<(d0, d1) -> (d0, d1), <1x1>, memref<256x768xbf16, #system_memory>>
#ttnn_layout_host_shard = #ttnn.ttnn_layout<(d0, d1) -> (d0, d1), <1x1>, memref<128x192xbf16, #system_memory>>
#ttnn_layout_l1_rm = #ttnn.ttnn_layout<(d0, d1) -> (d0, d1), <1x1>, memref<128x192xbf16, #l1>, <height_sharded>>
#ttnn_layout_l1_tile = #ttnn.ttnn_layout<(d0, d1) -> (d0, d1), <1x1>, memref<4x6x!ttcore.tile<32x32, bf16>, #l1>, <height_sharded>>

module attributes {ttcore.system_desc = #system_desc} {
  ttcore.device @default_device = <workerGrid = #ttcore.grid<8x8, (d0, d1) -> (0, d0, d1)>, l1Map = (d0, d1, d2)[s0] -> (0, d0, d1, d2 + s0), dramMap = (d0, d1, d2)[s0, s1, s2, s3, s4, s5, s6] -> (0, 0, (((d0 * s1) * (s2 * (s3 * s6)) + d1 * (s2 * (s3 * s6)) + d2) floordiv s4) mod 12, ((((d0 * s1) * (s2 * (s3 * s6)) + d1 * (s2 * (s3 * s6)) + d2) floordiv s4) floordiv 12) * s4 + ((d0 * s1) * (s2 * (s3 * s6)) + d1 * (s2 * (s3 * s6)) + d2) mod s4 + s5), meshShape = 2x4, chipIds = [0, 1, 2, 3, 4, 5, 6, 7]>

  func.func @test_fabric_p2p(%arg0: tensor<256x768xbf16, #ttnn_layout_host>) -> tensor<256x768xbf16, #ttnn_layout_host> {
    %0 = "ttnn.get_device"() <{mesh_offset = #ttnn<mesh_offset 0x0>, mesh_shape = #ttnn<mesh_shape 2x4>}> : () -> !ttnn.device

    // distribute_tensor shards host tensor across 2x4 mesh, output stays on HOST
    %1 = "ttnn.distribute_tensor"(%arg0, %0) <{mapper_config = #ttnn.mesh_mapper_config<placements = [<shard, 0 : i64>, <shard, 1 : i64>]>}> : (tensor<256x768xbf16, #ttnn_layout_host>, !ttnn.device) -> tensor<128x192xbf16, #ttnn_layout_host_shard>

    // Move from host to device L1 with height_sharded layout (full tensor on one core)
    %2 = "ttnn.to_device"(%1, %0) <{memory_config = #ttnn.memory_config<#l1, <height_sharded>>}> : (tensor<128x192xbf16, #ttnn_layout_host_shard>, !ttnn.device) -> tensor<128x192xbf16, #ttnn_layout_l1_rm>

    // Convert from row-major to tile layout
    %3 = "ttnn.to_layout"(%2) <{layout = #ttnn.layout<tile>, dtype = #ttcore.supportedDataTypes<bf16>, memory_config = #ttnn.memory_config<#l1, <height_sharded>>}> : (tensor<128x192xbf16, #ttnn_layout_l1_rm>) -> tensor<128x192xbf16, #ttnn_layout_l1_tile>

    %4 = "ttnn.empty"(%0) <{dtype = #ttcore.supportedDataTypes<bf16>, layout = #ttnn.layout<tile>, memory_config = #ttnn.memory_config<#l1, <height_sharded>>, shape = #ttnn.shape<128x192>}> : (!ttnn.device) -> tensor<128x192xbf16, #ttnn_layout_l1_tile>

    // Generic op with CB size = 49152 = full 4x6 tiles on single core with sharded layout
    "ttnn.generic"(%3, %4) <{program = #ttnn.program<kernels = [#ttnn.read_kernel<symbol_ref = @datamovement_kernel0, core_ranges = <[#ttnn.core_range<(0,0), (0,0)>]>, ct_args = [#ttnn.kernel_arg_cb_buffer_index<0>], common_rt_args = [], rt_args = []>], cbs = [<total_size = 49152, core_ranges = <[#ttnn.core_range<(0,0), (0,0)>]>, formats = [<buffer_index = 0, dtype = bf16, page_size = 2048>], buffer = #ttnn.kernel_cb_global_buffer_address_of_tensor<0>>], semaphores = []>}> : (tensor<128x192xbf16, #ttnn_layout_l1_tile>, tensor<128x192xbf16, #ttnn_layout_l1_tile>) -> ()

    // Move from device back to host, then aggregate
    %5 = "ttnn.from_device"(%4) : (tensor<128x192xbf16, #ttnn_layout_l1_tile>) -> tensor<128x192xbf16, #ttnn_layout_host_shard>
    %6 = "ttnn.aggregate_tensor"(%5, %0) <{composer_config = #ttnn.mesh_composer_config<dims = [0 : i64, 1 : i64]>}> : (tensor<128x192xbf16, #ttnn_layout_host_shard>, !ttnn.device) -> tensor<256x768xbf16, #ttnn_layout_host>

    return %6 : tensor<256x768xbf16, #ttnn_layout_host>
  }

  func.func private @datamovement_kernel0() attributes {ttkernel.arg_spec = #ttkernel.arg_spec< ct_args = [<arg_type = cb_port, operand_index = 0>]>, ttkernel.thread = #ttkernel.thread<noc>} {
    %0 = "emitc.constant"() <{value = 49152 : i32}> : () -> i32
    %1 = "emitc.constant"() <{value = 0 : i16}> : () -> i16
    %2 = "emitc.constant"() <{value = 1 : i16}> : () -> i16
    %3 = "emitc.constant"() <{value = #emitc.opaque<"get_absolute_logical_x()">}> : () -> !emitc.size_t
    %4 = "emitc.constant"() <{value = #emitc.opaque<"get_absolute_logical_y()">}> : () -> !emitc.size_t
    %5 = emitc.call_opaque "experimental::convert_logical_x_to_translated"(%3) : (!emitc.size_t) -> !emitc.size_t
    %6 = emitc.call_opaque "experimental::convert_logical_y_to_translated"(%4) : (!emitc.size_t) -> !emitc.size_t
    %7 = emitc.literal "get_compile_time_arg_val(0)" : !emitc.opaque<"::tt::CB">
    %8 = emitc.call_opaque "get_write_ptr"(%7) : (!emitc.opaque<"::tt::CB">) -> i32
    %9 = emitc.call_opaque "get_noc_addr"(%5, %6, %8) : (!emitc.size_t, !emitc.size_t, i32) -> i64
    %10 = emitc.call_opaque "experimental::get_my_device_id"() : () -> i16
    %11 = emitc.cmp eq, %10, %1 : (i16, i16) -> i1
    emitc.if %11 {
      emitc.call_opaque "experimental::fabric_fast_write_any_len"(%1, %2, %9, %8, %0) : (i16, i16, i64, i32, i32) -> ()
    }
    emitc.call_opaque "experimental::close_fabric_connections"() : () -> ()
    return
  }
}
