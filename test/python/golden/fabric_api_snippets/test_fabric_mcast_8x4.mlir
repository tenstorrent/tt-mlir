!full_tensor_layout = memref<256x128xbf16>
!mesh_shard_layout = memref<32x32xbf16, #ttcore.host_layout<logical_shape = 32x32, host_strides = 32x1, host_volume = 1024, <"mesh">>>
!l1_shard_layout = memref<1x1x1x1x!ttcore.tile<32x32, bf16>, #ttcore.shard<2048x2048, 1>, #ttcore.memory_space<l1>>


module attributes {} {
  ttcore.device @default_device = <workerGrid = #ttcore.grid<8x8, (d0, d1) -> (0, d0, d1)>, l1Map = (d0, d1, d2)[s0] -> (0, d0, d1, d2 + s0), dramMap = (d0, d1, d2)[s0, s1, s2, s3, s4, s5, s6] -> (0, 0, (((d0 * s1) * (s2 * (s3 * s6)) + d1 * (s2 * (s3 * s6)) + d2) floordiv s4) mod 12, ((((d0 * s1) * (s2 * (s3 * s6)) + d1 * (s2 * (s3 * s6)) + d2) floordiv s4) floordiv 12) * s4 + ((d0 * s1) * (s2 * (s3 * s6)) + d1 * (s2 * (s3 * s6)) + d2) mod s4 + s5), meshShape = 8x4, chipIds = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31]>
  func.func @test_fabric_mcast_8x4(%arg0: !full_tensor_layout) -> !full_tensor_layout {
    %0 = "ttmetal.mesh_shard"(%arg0) <{shard_dims = array<i64: 0, 1>, shard_direction = #ttcore.shard_direction<full_to_shard>, shard_shape = array<i64: 4, 8>, shard_type = #ttcore.shard_type<devices>}> : (!full_tensor_layout) -> !mesh_shard_layout
    %1 = "ttmetal.create_buffer"() <{address = 104128 : i64}> : () -> !l1_shard_layout
    "ttmetal.enqueue_write_buffer"(%0, %1) : (!mesh_shard_layout, !l1_shard_layout) -> ()
    "ttmetal.enqueue_program"(%1, %1) <{cb_ports = array<i64: 0>, kernelConfigs = [#ttmetal.noc_config<@datamovement_kernel0, #ttmetal.core_range<0x0, 1x1>, #ttmetal.kernel_args< ct_args = [<cb_port[0]>]>, noc0>], operandSegmentSizes = array<i32: 1, 1>, fabricConnectionConfig = #ttmetal.fabric_connection_config<noc_index = noc0, topology = ring, cluster_axis = 1, num_links = 1>}> : (!l1_shard_layout, !l1_shard_layout) -> ()
    %alloc_1 = memref.alloc() : !mesh_shard_layout
    "ttmetal.enqueue_read_buffer"(%1, %alloc_1) : (!l1_shard_layout, !mesh_shard_layout) -> ()
    "ttmetal.finish"() : () -> ()
    "ttmetal.deallocate_buffer"(%1) : (!l1_shard_layout) -> ()
    %8 = "ttmetal.mesh_shard"(%alloc_1) <{shard_dims = array<i64: 0, 1>, shard_direction = #ttcore.shard_direction<shard_to_full>, shard_shape = array<i64: 4, 8>, shard_type = #ttcore.shard_type<devices>}> : (!mesh_shard_layout) -> !full_tensor_layout
    return %8 : !full_tensor_layout
  }

  func.func private @datamovement_kernel0() attributes {ttkernel.arg_spec = #ttkernel.arg_spec< ct_args = [<arg_type = cb_port, operand_index = 0>]>, ttkernel.thread = #ttkernel.thread<noc>} {
    // Setup fabric connections
    %fabric_connection_manager = "ttkernel.experimental::create_fabric_connection_manager"() : () -> !ttkernel.fabric_connection_manager
    "ttkernel.experimental::setup_fabric_connections"(%fabric_connection_manager) : (!ttkernel.fabric_connection_manager) -> ()

    // Constants
    %len_bytes = arith.constant 2048 : i32
    %src_dev_id = arith.constant 0 : i16
    %dst_mesh_id = arith.constant 0 : i16
    %dst_dev_id_start = arith.constant 1 : i16
    %dst_dev_id_end = arith.constant 2 : i16

    // Get logical coordinates of current core and convert logical coords to translated coords
    %logical_x = ttkernel.my_logical_x_ : () -> index
    %logical_y = ttkernel.my_logical_y_ : () -> index
    %translated_x = "ttkernel.experimental::convert_logical_x_to_translated"(%logical_x) : (index) -> index
    %translated_y = "ttkernel.experimental::convert_logical_y_to_translated"(%logical_y) : (index) -> index

    // Get CB write pointers
    %cb0 = ttkernel.get_compile_time_arg_val(0) : () -> !ttkernel.cb<24, !ttcore.tile<32x32, bf16>>
    %write_ptr = ttkernel.get_write_ptr(%cb0) : (!ttkernel.cb<24, !ttcore.tile<32x32, bf16>>) -> i32

    // Get noc address
    %noc_addr = ttkernel.get_noc_addr(%translated_x, %translated_y, %write_ptr) : (index, index, i32) -> !ttkernel.noc_addr

    // Get my device id
    %my_device_id = "ttkernel.experimental::get_my_device_id"() : () -> i16

    // Check if device is 0 and send to device 1 if it is
    %is_device_0 = arith.cmpi eq, %my_device_id, %src_dev_id : i16
    scf.if %is_device_0 {
      // Device 0 sends to device 1
      "ttkernel.experimental::fabric_mcast_fast_write_any_len"(%fabric_connection_manager, %dst_mesh_id, %dst_dev_id_start, %dst_dev_id_end, %noc_addr, %write_ptr, %len_bytes) : (!ttkernel.fabric_connection_manager, i16, i16, i16, !ttkernel.noc_addr, i32, i32) -> ()
    }

    // Close fabric connections
    "ttkernel.experimental::close_fabric_connections"(%fabric_connection_manager) : (!ttkernel.fabric_connection_manager) -> ()

    return
  }
}
