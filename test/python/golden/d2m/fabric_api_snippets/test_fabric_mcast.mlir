!full_tensor_layout = memref<insert_full_tensor_shape_0xinsert_full_tensor_shape_1xbf16>
!mesh_shard_layout = memref<32x32xbf16, #ttcore.host_layout<logical_shape = 32x32, host_strides = 32x1, host_volume = 1024, <"mesh">>>
!src_shard_layout = memref<1x1x1x1x!ttcore.tile<32x32, bf16>, #ttcore.shard<2048x2048, 1>, #ttcore.memory_space<l1>>
!dst_shard_layout = memref<1x1x1x1x!ttcore.tile<32x32, bf16>, #ttcore.shard<2048x2048, 1>, #ttcore.memory_space<insert_dst_memory_space>>

module attributes {} {
  ttcore.device @default_device = <workerGrid = #ttcore.grid<8x8, virt_to_physical_map = (d0, d1) -> (0, d0, d1), physical_to_virt_map = (d0, d1, d2) -> (d1, d2)>, dramGrid = #ttcore.grid<1x12>, l1Map = (d0, d1, d2)[s0] -> (0, d0, d1, d2 + s0), dramMap = (d0, d1, d2)[s0, s1, s2, s3, s4, s5, s6] -> (0, 0, (((d0 * s1) * (s2 * (s3 * s6)) + d1 * (s2 * (s3 * s6)) + d2) floordiv s4) mod 12, ((((d0 * s1) * (s2 * (s3 * s6)) + d1 * (s2 * (s3 * s6)) + d2) floordiv s4) floordiv 12) * s4 + ((d0 * s1) * (s2 * (s3 * s6)) + d1 * (s2 * (s3 * s6)) + d2) mod s4 + s5), meshShape = insert_mesh_shape_0xinsert_mesh_shape_1, chipIds = insert_chip_ids>
  func.func @test_fabric_mcast(%arg0: !full_tensor_layout) -> !full_tensor_layout {
    %0 = "ttmetal.mesh_shard"(%arg0) <{shard_dims = array<i64: 0, 1>, shard_direction = #ttcore.shard_direction<full_to_shard>, shard_shape = array<i64: insert_mesh_shape_0, insert_mesh_shape_1>, shard_type = #ttcore.shard_type<devices>}> : (!full_tensor_layout) -> !mesh_shard_layout
    %src = "ttmetal.create_buffer"() <{address = insert_src_buffer_address : i64}> : () -> !src_shard_layout
    %dst = "ttmetal.create_buffer"() <{address = insert_dst_buffer_address : i64}> : () -> !dst_shard_layout
    "ttmetal.enqueue_write_buffer"(%0, %src) : (!mesh_shard_layout, !src_shard_layout) -> ()
    // Test assumes output is the same as input except for the shards that get overwritten so we copy the input host tensor to the output as well.
    "ttmetal.enqueue_write_buffer"(%0, %dst) : (!mesh_shard_layout, !dst_shard_layout) -> ()
    "ttmetal.enqueue_program"(%dst, %src) <{cb_ports = array<i64: 0>, kernelConfigs = [#ttmetal.noc_config<@datamovement_kernel0, #ttmetal.core_range<0x0, 1x1>, #ttmetal.kernel_args<common_rt_args = [<buffer_address[0]>]  ct_args = [<cb_port[0]>]>, dm_core = 1, insert_noc_name>], operandSegmentSizes = array<i32: 1, 1>, fabricConnectionConfig = #ttcore.fabric_connection_config<noc_index = insert_noc_name, topology = insert_topology, cluster_axis = insert_cluster_axis, routing_mode = insert_routing_mode, num_links = 1>}> : (!dst_shard_layout, !src_shard_layout) -> ()
    %alloc_1 = memref.alloc() : !mesh_shard_layout
    "ttmetal.enqueue_read_buffer"(%dst, %alloc_1) : (!dst_shard_layout, !mesh_shard_layout) -> ()
    "ttmetal.finish"() : () -> ()
    "ttmetal.deallocate_buffer"(%dst) : (!dst_shard_layout) -> ()
    "ttmetal.deallocate_buffer"(%src) : (!src_shard_layout) -> ()
    %8 = "ttmetal.mesh_shard"(%alloc_1) <{shard_dims = array<i64: 0, 1>, shard_direction = #ttcore.shard_direction<shard_to_full>, shard_shape = array<i64: insert_mesh_shape_0, insert_mesh_shape_1>, shard_type = #ttcore.shard_type<devices>}> : (!mesh_shard_layout) -> !full_tensor_layout
    return %8 : !full_tensor_layout
  }

  func.func private @datamovement_kernel0() attributes {ttkernel.arg_spec = #ttkernel.arg_spec< ct_args = [<arg_type = cb_port, operand_index = 0>]>, ttkernel.thread = #ttkernel.thread<noc>} {
    // Setup fabric connections
    %fabric_connection_manager = "ttkernel.experimental.create_fabric_connection_manager"() : () -> !ttkernel.fabric_connection_manager
    "ttkernel.experimental.setup_fabric_connections"(%fabric_connection_manager) : (!ttkernel.fabric_connection_manager) -> ()

    // Constants
    %len_bytes = arith.constant 2048 : i32
    %src_dev_id = arith.constant insert_src_dev_id : i16
    %dst_mesh_id = arith.constant 0 : i16
    %dst_dev_id_start = arith.constant insert_dst_dev_id_start : i16
    %dst_dev_id_end = arith.constant insert_dst_dev_id_end : i16

    // Get CB write pointer for src address
    %src_cb = ttkernel.get_compile_time_arg_val(0) : () -> !ttkernel.cb<24, !ttcore.tile<32x32, bf16>>
    %src_ptr = ttkernel.get_write_ptr(%src_cb) : (!ttkernel.cb<24, !ttcore.tile<32x32, bf16>>) -> i32

    // Destination buffer base address (common runtime arg).
    %c0_idx = arith.constant 0 : index
    %dst_addr = ttkernel.get_common_arg_val(%c0_idx) : (index) -> i32

    // Compute destination noc address
    %is_dram = arith.constant insert_is_dram : i1
    %noc_addr = scf.if %is_dram -> (!ttkernel.noc_addr) {
      // DRAM destination
      %bank_id = arith.constant 0 : i32
      %dram_noc_addr = ttkernel.experimental.get_fabric_noc_addr_from_bank_id(%bank_id, %dst_addr) : (i32, i32) -> !ttkernel.noc_addr
      scf.yield %dram_noc_addr : !ttkernel.noc_addr
    } else {
      // L1 destination
      %logical_x = ttkernel.my_logical_x_ : () -> index
      %logical_y = ttkernel.my_logical_y_ : () -> index
      %translated_x = "ttkernel.experimental.convert_logical_x_to_translated"(%logical_x) : (index) -> index
      %translated_y = "ttkernel.experimental.convert_logical_y_to_translated"(%logical_y) : (index) -> index
      %noc_idx = arith.constant insert_noc_index : i8
      %l1_noc_addr = ttkernel.get_noc_addr(%translated_x, %translated_y, %dst_addr, %noc_idx) : (index, index, i32, i8) -> !ttkernel.noc_addr
      scf.yield %l1_noc_addr : !ttkernel.noc_addr
    }

    // Get my device id
    %my_device_id = "ttkernel.experimental.get_my_device_id"() : () -> i16

    // Src device multicasts its shard to the [start, end] device range.
    %is_src = arith.cmpi eq, %my_device_id, %src_dev_id : i16
    scf.if %is_src {
      "ttkernel.experimental.fabric_mcast_fast_write_any_len"(%fabric_connection_manager, %dst_mesh_id, %dst_dev_id_start, %dst_dev_id_end, %noc_addr, %src_ptr, %len_bytes) : (!ttkernel.fabric_connection_manager, i16, i16, i16, !ttkernel.noc_addr, i32, i32) -> ()
    }

    // Close fabric connections
    "ttkernel.experimental.close_fabric_connections"(%fabric_connection_manager) : (!ttkernel.fabric_connection_manager) -> ()

    return
  }
}
