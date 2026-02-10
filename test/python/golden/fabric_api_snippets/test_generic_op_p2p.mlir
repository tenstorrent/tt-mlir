#l1 = #ttnn.buffer_type<l1>
#system_memory = #ttnn.buffer_type<system_memory>
#ttnn_layout_host_row_major = #ttnn.ttnn_layout<(d0, d1) -> (d0, d1), <1x1>, memref<64x128xbf16, #system_memory>>
#ttnn_layout_host_tile = #ttnn.ttnn_layout<(d0, d1) -> (d0, d1), <1x1>, memref<2x4x!ttcore.tile<32x32, bf16>, #system_memory>>
#ttnn_layout_host_tile_shard = #ttnn.ttnn_layout<(d0, d1) -> (d0, d1), <1x1>, memref<1x1x!ttcore.tile<32x32, bf16>, #system_memory>>
#ttnn_layout_device_tile_sharded = #ttnn.ttnn_layout<(d0, d1) -> (d0, d1), <1x1, (d0, d1) -> (0, d0, d1)>, memref<1x1x!ttcore.tile<32x32, bf16>, #l1>, <block_sharded>>
#ttnn_layout_device_tile_interleaved = #ttnn.ttnn_layout<(d0, d1) -> (d0, d1), <1x1>, memref<1x1x!ttcore.tile<32x32, bf16>, #l1>, <interleaved>>
module attributes {} {
  ttcore.device @default_device = <workerGrid = #ttcore.grid<8x8, (d0, d1) -> (0, d0, d1)>, l1Map = (d0, d1, d2)[s0] -> (0, d0, d1, d2 + s0), dramMap = (d0, d1, d2)[s0, s1, s2, s3, s4, s5, s6] -> (0, 0, (((d0 * s1) * (s2 * (s3 * s6)) + d1 * (s2 * (s3 * s6)) + d2) floordiv s4) mod 12, ((((d0 * s1) * (s2 * (s3 * s6)) + d1 * (s2 * (s3 * s6)) + d2) floordiv s4) floordiv 12) * s4 + ((d0 * s1) * (s2 * (s3 * s6)) + d1 * (s2 * (s3 * s6)) + d2) mod s4 + s5), meshShape = 2x4, chipIds = [0, 1, 2, 3, 4, 5, 6, 7]>
  func.func @test_fabric_p2p(%arg0: tensor<64x128xbf16, #ttnn_layout_host_row_major>) -> tensor<64x128xbf16, #ttnn_layout_host_row_major> attributes {tt.function_type = "forward_device"} {
    %0 = "ttnn.get_device"() <{mesh_offset = #ttnn<mesh_offset 0x0>, mesh_shape = #ttnn<mesh_shape 2x4>}> : () -> !ttnn.device

    // distribute input tensor
    %1 = "ttnn.to_layout"(%arg0) <{layout = #ttnn.layout<tile>}> : (tensor<64x128xbf16, #ttnn_layout_host_row_major>) -> tensor<64x128xbf16, #ttnn_layout_host_tile>
    %2 = "ttnn.distribute_tensor"(%1, %0) <{mapper_config = #ttnn.mesh_mapper_config<placements = [<shard, 0 : i64>, <shard, 1 : i64>]>}> : (tensor<64x128xbf16, #ttnn_layout_host_tile>, !ttnn.device) -> tensor<32x32xbf16, #ttnn_layout_host_tile_shard>
    %3 = "ttnn.to_device"(%2, %0) <{memory_config = #ttnn.memory_config<#l1, <block_sharded>, #ttnn.shard_spec<<[#ttnn.core_range<(0,0), (0,0)>]>, <32x32>, <row_major>>>}> : (tensor<32x32xbf16, #ttnn_layout_host_tile_shard>, !ttnn.device) -> tensor<32x32xbf16, #ttnn_layout_device_tile_sharded>

    // preallocate and distribute output tensor
    %out1 = "ttnn.to_layout"(%arg0) <{layout = #ttnn.layout<tile>}> : (tensor<64x128xbf16, #ttnn_layout_host_row_major>) -> tensor<64x128xbf16, #ttnn_layout_host_tile>
    %out2 = "ttnn.distribute_tensor"(%out1, %0) <{mapper_config = #ttnn.mesh_mapper_config<placements = [<shard, 0 : i64>, <shard, 1 : i64>]>}> : (tensor<64x128xbf16, #ttnn_layout_host_tile>, !ttnn.device) -> tensor<32x32xbf16, #ttnn_layout_host_tile_shard>
    %out3 = "ttnn.to_device"(%out2, %0) <{memory_config = #ttnn.memory_config<#l1, <block_sharded>, #ttnn.shard_spec<<[#ttnn.core_range<(0,0), (0,0)>]>, <32x32>, <row_major>>>}> : (tensor<32x32xbf16, #ttnn_layout_host_tile_shard>, !ttnn.device) -> tensor<32x32xbf16, #ttnn_layout_device_tile_sharded>

    "ttnn.generic"(%3, %out3) <{
      program = #ttnn.mesh_program_descriptor<[
        #ttnn.mesh_program<
          range = #ttnn.mesh_range<(0,0), (1,3)>,
          program = #ttnn.program<
            kernels = [
              #ttnn.read_kernel<
                symbol_ref = @datamovement_kernel0,
                core_ranges = <[#ttnn.core_range<(0,0), (0,0)>]>,
                ct_args = [#ttnn.kernel_arg_cb_buffer_index<0>, #ttnn.kernel_arg_cb_buffer_index<1>],
                common_rt_args = [],
                rt_args = []>
            ],
            cbs = [
              <total_size = 2048,
               core_ranges = <[#ttnn.core_range<(0,0), (0,0)>]>,
               formats = [<buffer_index = 0, dtype = bf16, page_size = 2048>],
               buffer = #ttnn.kernel_cb_global_buffer_address_of_tensor<0>>,
              <total_size = 2048,
               core_ranges = <[#ttnn.core_range<(0,0), (0,0)>]>,
               formats = [<buffer_index = 1, dtype = bf16, page_size = 2048>],
               buffer = #ttnn.kernel_cb_global_buffer_address_of_tensor<1>>
            ],
            semaphores = []>
        >
      ], fabric_connection_config = #ttnn.fabric_connection_config<noc_index = noc0, topology = ring, cluster_axis = 1, num_links = 1>>
    }> : (tensor<32x32xbf16, #ttnn_layout_device_tile_sharded>, tensor<32x32xbf16, #ttnn_layout_device_tile_sharded>) -> ()

    // convert from block_sharded to interleaved before from_device or else aggregate_tensor will crash
    %out4 = "ttnn.to_memory_config"(%out3) <{memory_config = #ttnn.memory_config<#l1, <interleaved>>}> : (tensor<32x32xbf16, #ttnn_layout_device_tile_sharded>) -> tensor<32x32xbf16, #ttnn_layout_device_tile_interleaved>
    %4 = "ttnn.from_device"(%out4) : (tensor<32x32xbf16, #ttnn_layout_device_tile_interleaved>) -> tensor<32x32xbf16, #ttnn_layout_host_tile_shard>
    %5 = "ttnn.aggregate_tensor"(%4, %0) <{composer_config = #ttnn.mesh_composer_config<dims = [0 : i64, 1 : i64]>}> : (tensor<32x32xbf16, #ttnn_layout_host_tile_shard>, !ttnn.device) -> tensor<64x128xbf16, #ttnn_layout_host_tile>
    %6 = "ttnn.to_layout"(%5) <{layout = #ttnn.layout<row_major>}> : (tensor<64x128xbf16, #ttnn_layout_host_tile>) -> tensor<64x128xbf16, #ttnn_layout_host_row_major>

    return %6 : tensor<64x128xbf16, #ttnn_layout_host_row_major>
  }
  func.func private @datamovement_kernel0() attributes {tt.function_type = "kernel", ttkernel.arg_spec = #ttkernel.arg_spec< ct_args = [<arg_type = cb_port, operand_index = 0>, <arg_type = cb_port, operand_index = 1>]>, ttkernel.thread = #ttkernel.thread<noc>} {
    // Setup fabric connections
    %fabric_connection_manager_var = "emitc.variable"() <{value = #emitc.opaque<"">}> : () -> !emitc.lvalue<!emitc.opaque<"experimental::FabricConnectionManager">>
    %fabric_connection_manager = emitc.load %fabric_connection_manager_var : <!emitc.opaque<"experimental::FabricConnectionManager">>
    emitc.call_opaque "experimental::setup_fabric_connections"(%fabric_connection_manager) : (!emitc.opaque<"experimental::FabricConnectionManager">) -> ()

    // Constants
    %len_bytes = "emitc.constant"() <{value = 49152 : i32}> : () -> i32
    %device0 = "emitc.constant"() <{value = 0 : i16}> : () -> i16
    %device1 = "emitc.constant"() <{value = 1 : i16}> : () -> i16

    // Get logical coordinates of current core and convert logical coords to translated coords
    %my_x = "emitc.constant"() <{value = #emitc.opaque<"get_absolute_logical_x()">}> : () -> !emitc.size_t
    %my_y = "emitc.constant"() <{value = #emitc.opaque<"get_absolute_logical_y()">}> : () -> !emitc.size_t
    %translated_x = emitc.call_opaque "experimental::convert_logical_x_to_translated"(%my_x) : (!emitc.size_t) -> !emitc.size_t
    %translated_y = emitc.call_opaque "experimental::convert_logical_y_to_translated"(%my_y) : (!emitc.size_t) -> !emitc.size_t

    // Get CB write pointers
    %cb0 = emitc.literal "get_compile_time_arg_val(0)" : !emitc.opaque<"::tt::CB">
    %src_ptr = emitc.call_opaque "get_write_ptr"(%cb0) : (!emitc.opaque<"::tt::CB">) -> i32
    %cb1 = emitc.literal "get_compile_time_arg_val(1)" : !emitc.opaque<"::tt::CB">
    %dst_ptr = emitc.call_opaque "get_write_ptr"(%cb1) : (!emitc.opaque<"::tt::CB">) -> i32

    // Get NOC address
    %noc_addr = emitc.call_opaque "get_noc_addr"(%translated_x, %translated_y, %dst_ptr) : (!emitc.size_t, !emitc.size_t, i32) -> i64

    // Get my device id
    %my_device_id = emitc.call_opaque "experimental::get_my_device_id"() : () -> i16

    // Check if device is 0 and send to device 1 if it is
    %is_device0 = emitc.cmp eq, %my_device_id, %device0 : (i16, i16) -> i1
    emitc.if %is_device0 {
      emitc.call_opaque "experimental::fabric_fast_write_any_len"(%fabric_connection_manager, %device0, %device1, %noc_addr, %src_ptr, %len_bytes) : (!emitc.opaque<"experimental::FabricConnectionManager">, i16, i16, i64, i32, i32) -> ()
    }
    emitc.call_opaque "experimental::close_fabric_connections"(%fabric_connection_manager) : (!emitc.opaque<"experimental::FabricConnectionManager">) -> ()
    return
  }
}
