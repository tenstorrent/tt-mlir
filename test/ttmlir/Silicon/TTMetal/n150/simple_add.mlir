#l1 = #ttcore.memory_space<l1>
#system_desc = #ttcore.system_desc<[{role = host, target_triple = "x86_64-pc-linux"}], [{arch = <wormhole_b0>, grid = 8x8, coord_translation_offsets = 18x18, l1_size = 1499136, num_dram_channels = 12, dram_channel_size = 1073741824, noc_l1_address_align_bytes = 16, pcie_address_align_bytes = 32, noc_dram_address_align_bytes = 32, l1_unreserved_base = 101664, erisc_l1_unreserved_base = 98304, dram_unreserved_base = 2560032, dram_unreserved_end = 1073174176, supported_data_types = [<f32>, <f16>, <bf16>, <bfp_f8>, <bfp_bf8>, <bfp_f4>, <bfp_bf4>, <bfp_f2>, <bfp_bf2>, <u32>, <u16>, <u8>, <si32>], supported_tile_sizes = [ 4x16,  16x16,  32x16,  4x32,  16x32,  32x32], dst_physical_size_tiles = 16, num_cbs = 32, num_compute_threads = 1, num_datamovement_threads = 2}, {arch = <wormhole_b0>, grid = 8x8, coord_translation_offsets = 18x18, l1_size = 1499136, num_dram_channels = 12, dram_channel_size = 1073741824, noc_l1_address_align_bytes = 16, pcie_address_align_bytes = 32, noc_dram_address_align_bytes = 32, l1_unreserved_base = 101664, erisc_l1_unreserved_base = 98304, dram_unreserved_base = 2560032, dram_unreserved_end = 1073182656, supported_data_types = [<f32>, <f16>, <bf16>, <bfp_f8>, <bfp_bf8>, <bfp_f4>, <bfp_bf4>, <bfp_f2>, <bfp_bf2>, <u32>, <u16>, <u8>, <si32>], supported_tile_sizes = [ 4x16,  16x16,  32x16,  4x32,  16x32,  32x32], dst_physical_size_tiles = 16, num_cbs = 32, num_compute_threads = 1, num_datamovement_threads = 2}], [0, 1], [1 : i32, 0 : i32], [ 0x0x0x0]>
module {
  ttcore.device_module {
    builtin.module attributes {ttcore.system_desc = #system_desc} {
      ttcore.device @default_device = <workerGrid = #ttcore.grid<8x8, (d0, d1) -> (0, d0, d1)>, l1Map = (d0, d1, d2)[s0] -> (0, d0, d1, d2 + s0), dramMap = (d0, d1, d2)[s0, s1, s2, s3, s4, s5, s6] -> (0, 0, (((d0 * s1) * (s2 * (s3 * s6)) + d1 * (s2 * (s3 * s6)) + d2) floordiv s4) mod 12, ((((d0 * s1) * (s2 * (s3 * s6)) + d1 * (s2 * (s3 * s6)) + d2) floordiv s4) floordiv 12) * s4 + ((d0 * s1) * (s2 * (s3 * s6)) + d1 * (s2 * (s3 * s6)) + d2) mod s4 + s5), meshShape = 1x1, chipIds = [0]>
      memref.global "private" constant @__constant_32x32xf32 : memref<32x32xf32> = dense<1.000000e+00>
      func.func @reductions_constrained_inputs(%arg0: memref<512x64xf32>) -> memref<512x1xf32> {
        %0 = memref.get_global @__constant_32x32xf32 : memref<32x32xf32>
        %1 = "ttmetal.create_buffer"() <{address = 134432 : i64}> : () -> memref<8x2x2x1x!ttcore.tile<32x32, f32>, #ttcore.shard<4096x4096>, #l1>
        %2 = "ttmetal.create_buffer"() <{address = 101664 : i64}> : () -> memref<8x2x64x32xf32, #ttcore.shard<128x4>, #l1>
        "ttmetal.enqueue_write_buffer"(%arg0, %2) : (memref<512x64xf32>, memref<8x2x64x32xf32, #ttcore.shard<128x4>, #l1>) -> ()
        "ttmetal.enqueue_program"(%2, %1, %2, %1) <{cb_ports = array<i64: 0, 1>, kernelConfigs = [#ttmetal.noc_config<@datamovement_kernel0, #ttmetal.core_range<0x0, 8x2>, #ttmetal.kernel_args< ct_args = [<cb_port[0]>, <cb_port[1]>]>, noc0>, #ttmetal.noc_config<@datamovement_kernel1, #ttmetal.core_range<0x0, 8x2>, #ttmetal.kernel_args< ct_args = [<cb_port[0]>, <cb_port[1]>]>, noc1>, #ttmetal.compute_config<@compute_kernel2, #ttmetal.core_range<0x0, 8x2>, #ttmetal.kernel_args< ct_args = [<cb_port[0]>, <cb_port[1]>]>, hifi4, true, false, false, [default]>], operandSegmentSizes = array<i32: 2, 2>}> : (memref<8x2x64x32xf32, #ttcore.shard<128x4>, #l1>, memref<8x2x2x1x!ttcore.tile<32x32, f32>, #ttcore.shard<4096x4096>, #l1>, memref<8x2x64x32xf32, #ttcore.shard<128x4>, #l1>, memref<8x2x2x1x!ttcore.tile<32x32, f32>, #ttcore.shard<4096x4096>, #l1>) -> ()
        "ttmetal.deallocate_buffer"(%2) : (memref<8x2x64x32xf32, #ttcore.shard<128x4>, #l1>) -> ()
        %3 = "ttmetal.create_buffer"() <{address = 142624 : i64}> : () -> memref<1x1x1x1x!ttcore.tile<32x32, f32>, #ttcore.shard<4096x4096>, #l1>
        %4 = "ttmetal.create_buffer"() <{address = 101664 : i64}> : () -> memref<1x1x32x32xf32, #ttcore.shard<128x4>, #l1>
        "ttmetal.enqueue_write_buffer"(%0, %4) : (memref<32x32xf32>, memref<1x1x32x32xf32, #ttcore.shard<128x4>, #l1>) -> ()
        "ttmetal.enqueue_program"(%4, %3, %4, %3) <{cb_ports = array<i64: 0, 1>, kernelConfigs = [#ttmetal.noc_config<@datamovement_kernel3, #ttmetal.core_range<0x0, 1x1>, #ttmetal.kernel_args< ct_args = [<cb_port[0]>, <cb_port[1]>]>, noc0>, #ttmetal.noc_config<@datamovement_kernel4, #ttmetal.core_range<0x0, 1x1>, #ttmetal.kernel_args< ct_args = [<cb_port[0]>, <cb_port[1]>]>, noc1>, #ttmetal.compute_config<@compute_kernel5, #ttmetal.core_range<0x0, 1x1>, #ttmetal.kernel_args< ct_args = [<cb_port[0]>, <cb_port[1]>]>, hifi4, true, false, false, [default]>], operandSegmentSizes = array<i32: 2, 2>}> : (memref<1x1x32x32xf32, #ttcore.shard<128x4>, #l1>, memref<1x1x1x1x!ttcore.tile<32x32, f32>, #ttcore.shard<4096x4096>, #l1>, memref<1x1x32x32xf32, #ttcore.shard<128x4>, #l1>, memref<1x1x1x1x!ttcore.tile<32x32, f32>, #ttcore.shard<4096x4096>, #l1>) -> ()
        "ttmetal.deallocate_buffer"(%4) : (memref<1x1x32x32xf32, #ttcore.shard<128x4>, #l1>) -> ()
        %5 = "ttmetal.create_buffer"() <{address = 118048 : i64}> : () -> memref<8x1x2x1x!ttcore.tile<32x32, f32>, #ttcore.shard<4096x4096>, #l1>
        %6 = "ttmetal.create_buffer"() <{address = 101664 : i64}> : () -> memref<8x2x2x1x!ttcore.tile<32x32, f32>, #ttcore.shard<4096x4096, 2>, #l1>
        %7 = "ttmetal.create_buffer"() <{address = 126240 : i64}> : () -> memref<1x1x1x1x!ttcore.tile<32x32, f32>, #ttcore.shard<4096x4096, 2>, #l1>
        "ttmetal.enqueue_program"(%1, %3, %5, %6, %7, %5) <{cb_ports = array<i64: 0, 1, 2>, kernelConfigs = [#ttmetal.noc_config<@datamovement_kernel6, #ttmetal.core_range<0x0, 8x1>, #ttmetal.kernel_args< ct_args = [<cb_port[0]>, <cb_port[1]>, <cb_port[2]>, <buffer_address[0]>]>, noc0>, #ttmetal.noc_config<@datamovement_kernel7, #ttmetal.core_range<0x0, 8x1>, #ttmetal.kernel_args< ct_args = [<cb_port[0]>, <cb_port[1]>, <cb_port[2]>, <buffer_address[1]>]>, noc1>, #ttmetal.compute_config<@compute_kernel8, #ttmetal.core_range<0x0, 8x1>, #ttmetal.kernel_args< ct_args = [<cb_port[0]>, <cb_port[1]>, <cb_port[2]>]>, hifi4, true, false, false, [default]>], operandSegmentSizes = array<i32: 3, 3>}> : (memref<8x2x2x1x!ttcore.tile<32x32, f32>, #ttcore.shard<4096x4096>, #l1>, memref<1x1x1x1x!ttcore.tile<32x32, f32>, #ttcore.shard<4096x4096>, #l1>, memref<8x1x2x1x!ttcore.tile<32x32, f32>, #ttcore.shard<4096x4096>, #l1>, memref<8x2x2x1x!ttcore.tile<32x32, f32>, #ttcore.shard<4096x4096, 2>, #l1>, memref<1x1x1x1x!ttcore.tile<32x32, f32>, #ttcore.shard<4096x4096, 2>, #l1>, memref<8x1x2x1x!ttcore.tile<32x32, f32>, #ttcore.shard<4096x4096>, #l1>) -> ()
        "ttmetal.deallocate_buffer"(%7) : (memref<1x1x1x1x!ttcore.tile<32x32, f32>, #ttcore.shard<4096x4096, 2>, #l1>) -> ()
        "ttmetal.deallocate_buffer"(%6) : (memref<8x2x2x1x!ttcore.tile<32x32, f32>, #ttcore.shard<4096x4096, 2>, #l1>) -> ()
        "ttmetal.deallocate_buffer"(%1) : (memref<8x2x2x1x!ttcore.tile<32x32, f32>, #ttcore.shard<4096x4096>, #l1>) -> ()
        "ttmetal.deallocate_buffer"(%3) : (memref<1x1x1x1x!ttcore.tile<32x32, f32>, #ttcore.shard<4096x4096>, #l1>) -> ()
        %alloc = memref.alloc() : memref<512x1xf32>
        %8 = "ttmetal.create_buffer"() <{address = 101664 : i64}> : () -> memref<8x1x64x32xf32, #ttcore.shard<128x4>, #l1>
        "ttmetal.enqueue_program"(%5, %8, %5, %8) <{cb_ports = array<i64: 0, 1>, kernelConfigs = [#ttmetal.noc_config<@datamovement_kernel9, #ttmetal.core_range<0x0, 8x1>, #ttmetal.kernel_args< ct_args = [<cb_port[0]>, <cb_port[1]>]>, noc0>, #ttmetal.noc_config<@datamovement_kernel10, #ttmetal.core_range<0x0, 8x1>, #ttmetal.kernel_args< ct_args = [<cb_port[0]>, <cb_port[1]>]>, noc1>, #ttmetal.compute_config<@compute_kernel11, #ttmetal.core_range<0x0, 8x1>, #ttmetal.kernel_args< ct_args = [<cb_port[0]>, <cb_port[1]>]>, hifi4, true, false, false, [default]>], operandSegmentSizes = array<i32: 2, 2>}> : (memref<8x1x2x1x!ttcore.tile<32x32, f32>, #ttcore.shard<4096x4096>, #l1>, memref<8x1x64x32xf32, #ttcore.shard<128x4>, #l1>, memref<8x1x2x1x!ttcore.tile<32x32, f32>, #ttcore.shard<4096x4096>, #l1>, memref<8x1x64x32xf32, #ttcore.shard<128x4>, #l1>) -> ()
        "ttmetal.deallocate_buffer"(%5) : (memref<8x1x2x1x!ttcore.tile<32x32, f32>, #ttcore.shard<4096x4096>, #l1>) -> ()
        %alloc_0 = memref.alloc() : memref<512x1xf32, #ttcore.host_layout<logical_shape = 512x1, host_strides = 32x1, host_volume = 16384>>
        "ttmetal.enqueue_read_buffer"(%8, %alloc_0) : (memref<8x1x64x32xf32, #ttcore.shard<128x4>, #l1>, memref<512x1xf32, #ttcore.host_layout<logical_shape = 512x1, host_strides = 32x1, host_volume = 16384>>) -> ()
        "ttmetal.finish"() : () -> ()
        "ttmetal.deallocate_buffer"(%8) : (memref<8x1x64x32xf32, #ttcore.shard<128x4>, #l1>) -> ()
        memref.copy %alloc_0, %alloc : memref<512x1xf32, #ttcore.host_layout<logical_shape = 512x1, host_strides = 32x1, host_volume = 16384>> to memref<512x1xf32>
        return %alloc : memref<512x1xf32>
      }
      func.func private @datamovement_kernel0() attributes {ttkernel.arg_spec = #ttkernel.arg_spec< ct_args = [<arg_type = cb_port, operand_index = 0>, <arg_type = cb_port, operand_index = 1>]>, ttkernel.thread = #ttkernel.thread<noc>} {
        %0 = emitc.expression  : () -> i32 {
          %2 = "emitc.constant"() <{value = 2 : i32}> : () -> i32
          yield %2 : i32
        }
        %1 = emitc.literal "get_compile_time_arg_val(0)" : !emitc.opaque<"::tt::CB">
        emitc.call_opaque "cb_reserve_back"(%1, %0) : (!emitc.opaque<"::tt::CB">, i32) -> ()
        emitc.call_opaque "cb_push_back"(%1, %0) : (!emitc.opaque<"::tt::CB">, i32) -> ()
        return
      }
      func.func private @datamovement_kernel1() attributes {ttkernel.arg_spec = #ttkernel.arg_spec< ct_args = [<arg_type = cb_port, operand_index = 0>, <arg_type = cb_port, operand_index = 1>]>, ttkernel.thread = #ttkernel.thread<noc>} {
        %0 = emitc.expression  : () -> i32 {
          %2 = "emitc.constant"() <{value = 2 : i32}> : () -> i32
          yield %2 : i32
        }
        %1 = emitc.literal "get_compile_time_arg_val(1)" : !emitc.opaque<"::tt::CB">
        emitc.call_opaque "cb_wait_front"(%1, %0) : (!emitc.opaque<"::tt::CB">, i32) -> ()
        emitc.call_opaque "cb_pop_front"(%1, %0) : (!emitc.opaque<"::tt::CB">, i32) -> ()
        return
      }
      func.func private @compute_kernel2() attributes {ttkernel.arg_spec = #ttkernel.arg_spec< ct_args = [<arg_type = cb_port, operand_index = 0>, <arg_type = cb_port, operand_index = 1>]>, ttkernel.thread = #ttkernel.thread<compute>} {
        %0 = emitc.expression  : () -> i32 {
          %4 = "emitc.constant"() <{value = 2 : i32}> : () -> i32
          yield %4 : i32
        }
        %1 = emitc.expression  : () -> i32 {
          %4 = "emitc.constant"() <{value = 1 : i32}> : () -> i32
          yield %4 : i32
        }
        %2 = emitc.literal "get_compile_time_arg_val(0)" : !emitc.opaque<"::tt::CB">
        %3 = emitc.literal "get_compile_time_arg_val(1)" : !emitc.opaque<"::tt::CB">
        emitc.call_opaque "cb_wait_front"(%2, %0) : (!emitc.opaque<"::tt::CB">, i32) -> ()
        emitc.call_opaque "cb_reserve_back"(%3, %0) : (!emitc.opaque<"::tt::CB">, i32) -> ()
        emitc.call_opaque "compute_kernel_hw_startup"(%2, %3) : (!emitc.opaque<"::tt::CB">, !emitc.opaque<"::tt::CB">) -> ()
        emitc.call_opaque "tilize_init"(%2, %1, %3) : (!emitc.opaque<"::tt::CB">, i32, !emitc.opaque<"::tt::CB">) -> ()
        emitc.call_opaque "experimental::tilize_block"(%2, %3, %0, %1) : (!emitc.opaque<"::tt::CB">, !emitc.opaque<"::tt::CB">, i32, i32) -> ()
        emitc.call_opaque "cb_pop_front"(%2, %0) : (!emitc.opaque<"::tt::CB">, i32) -> ()
        emitc.call_opaque "cb_push_back"(%3, %0) : (!emitc.opaque<"::tt::CB">, i32) -> ()
        return
      }
      func.func private @datamovement_kernel3() attributes {ttkernel.arg_spec = #ttkernel.arg_spec< ct_args = [<arg_type = cb_port, operand_index = 0>, <arg_type = cb_port, operand_index = 1>]>, ttkernel.thread = #ttkernel.thread<noc>} {
        %0 = emitc.expression  : () -> i32 {
          %2 = "emitc.constant"() <{value = 1 : i32}> : () -> i32
          yield %2 : i32
        }
        %1 = emitc.literal "get_compile_time_arg_val(0)" : !emitc.opaque<"::tt::CB">
        emitc.call_opaque "cb_reserve_back"(%1, %0) : (!emitc.opaque<"::tt::CB">, i32) -> ()
        emitc.call_opaque "cb_push_back"(%1, %0) : (!emitc.opaque<"::tt::CB">, i32) -> ()
        return
      }
      func.func private @datamovement_kernel4() attributes {ttkernel.arg_spec = #ttkernel.arg_spec< ct_args = [<arg_type = cb_port, operand_index = 0>, <arg_type = cb_port, operand_index = 1>]>, ttkernel.thread = #ttkernel.thread<noc>} {
        %0 = emitc.expression  : () -> i32 {
          %2 = "emitc.constant"() <{value = 1 : i32}> : () -> i32
          yield %2 : i32
        }
        %1 = emitc.literal "get_compile_time_arg_val(1)" : !emitc.opaque<"::tt::CB">
        emitc.call_opaque "cb_wait_front"(%1, %0) : (!emitc.opaque<"::tt::CB">, i32) -> ()
        emitc.call_opaque "cb_pop_front"(%1, %0) : (!emitc.opaque<"::tt::CB">, i32) -> ()
        return
      }
      func.func private @compute_kernel5() attributes {ttkernel.arg_spec = #ttkernel.arg_spec< ct_args = [<arg_type = cb_port, operand_index = 0>, <arg_type = cb_port, operand_index = 1>]>, ttkernel.thread = #ttkernel.thread<compute>} {
        %0 = emitc.expression  : () -> i32 {
          %3 = "emitc.constant"() <{value = 1 : i32}> : () -> i32
          yield %3 : i32
        }
        %1 = emitc.literal "get_compile_time_arg_val(0)" : !emitc.opaque<"::tt::CB">
        %2 = emitc.literal "get_compile_time_arg_val(1)" : !emitc.opaque<"::tt::CB">
        emitc.call_opaque "cb_wait_front"(%1, %0) : (!emitc.opaque<"::tt::CB">, i32) -> ()
        emitc.call_opaque "cb_reserve_back"(%2, %0) : (!emitc.opaque<"::tt::CB">, i32) -> ()
        emitc.call_opaque "compute_kernel_hw_startup"(%1, %2) : (!emitc.opaque<"::tt::CB">, !emitc.opaque<"::tt::CB">) -> ()
        emitc.call_opaque "tilize_init"(%1, %0, %2) : (!emitc.opaque<"::tt::CB">, i32, !emitc.opaque<"::tt::CB">) -> ()
        emitc.call_opaque "experimental::tilize_block"(%1, %2, %0, %0) : (!emitc.opaque<"::tt::CB">, !emitc.opaque<"::tt::CB">, i32, i32) -> ()
        emitc.call_opaque "cb_pop_front"(%1, %0) : (!emitc.opaque<"::tt::CB">, i32) -> ()
        emitc.call_opaque "cb_push_back"(%2, %0) : (!emitc.opaque<"::tt::CB">, i32) -> ()
        return
      }
      func.func private @datamovement_kernel6() attributes {ttkernel.arg_spec = #ttkernel.arg_spec< ct_args = [<arg_type = cb_port, operand_index = 0>, <arg_type = cb_port, operand_index = 1>, <arg_type = cb_port, operand_index = 2>, <arg_type = buffer_address, operand_index = 0>]>, ttkernel.thread = #ttkernel.thread<noc>} {
        %0 = emitc.expression  : () -> !emitc.size_t {
          %7 = "emitc.constant"() <{value = 2 : index}> : () -> !emitc.size_t
          yield %7 : !emitc.size_t
        }
        %1 = emitc.expression  : () -> !emitc.size_t {
          %7 = "emitc.constant"() <{value = 1 : index}> : () -> !emitc.size_t
          yield %7 : !emitc.size_t
        }
        %2 = emitc.expression  : () -> !emitc.size_t {
          %7 = "emitc.constant"() <{value = 0 : index}> : () -> !emitc.size_t
          yield %7 : !emitc.size_t
        }
        %3 = emitc.expression  : () -> i32 {
          %7 = "emitc.constant"() <{value = 2 : i32}> : () -> i32
          yield %7 : i32
        }
        %4 = emitc.expression  : () -> i32 {
          %7 = "emitc.constant"() <{value = 8192 : i32}> : () -> i32
          yield %7 : i32
        }
        %5 = emitc.literal "get_compile_time_arg_val(0)" : !emitc.opaque<"::tt::CB">
        %6 = emitc.literal "get_compile_time_arg_val(3)" : i32
        emitc.for %arg0 = %2 to %0 step %1  : !emitc.size_t {
          call_opaque "cb_reserve_back"(%5, %3) : (!emitc.opaque<"::tt::CB">, i32) -> ()
          %7 = expression %6, %arg0 : (i32, !emitc.size_t) -> i64 {
            %9 = "emitc.constant"() <{value = 18 : index}> : () -> !emitc.size_t
            %10 = "emitc.constant"() <{value = #emitc.opaque<"my_y[noc_index]">}> : () -> !emitc.size_t
            %11 = add %arg0, %9 : (!emitc.size_t, !emitc.size_t) -> !emitc.size_t
            %12 = call_opaque "get_noc_addr"(%11, %10, %6) : (!emitc.size_t, !emitc.size_t, i32) -> i64
            yield %12 : i64
          }
          %8 = expression %5 : (!emitc.opaque<"::tt::CB">) -> i32 {
            %9 = call_opaque "get_write_ptr"(%5) : (!emitc.opaque<"::tt::CB">) -> i32
            yield %9 : i32
          }
          call_opaque "noc_async_read"(%7, %8, %4) : (i64, i32, i32) -> ()
          call_opaque "noc_async_read_barrier"() : () -> ()
          call_opaque "cb_push_back"(%5, %3) : (!emitc.opaque<"::tt::CB">, i32) -> ()
        }
        return
      }
      func.func private @datamovement_kernel7() attributes {ttkernel.arg_spec = #ttkernel.arg_spec< ct_args = [<arg_type = cb_port, operand_index = 0>, <arg_type = cb_port, operand_index = 1>, <arg_type = cb_port, operand_index = 2>, <arg_type = buffer_address, operand_index = 1>]>, ttkernel.thread = #ttkernel.thread<noc>} {
        %0 = emitc.expression  : () -> !emitc.size_t {
          %8 = "emitc.constant"() <{value = 2 : index}> : () -> !emitc.size_t
          yield %8 : !emitc.size_t
        }
        %1 = emitc.expression  : () -> !emitc.size_t {
          %8 = "emitc.constant"() <{value = 1 : index}> : () -> !emitc.size_t
          yield %8 : !emitc.size_t
        }
        %2 = emitc.expression  : () -> !emitc.size_t {
          %8 = "emitc.constant"() <{value = 0 : index}> : () -> !emitc.size_t
          yield %8 : !emitc.size_t
        }
        %3 = emitc.expression  : () -> i32 {
          %8 = "emitc.constant"() <{value = 1 : i32}> : () -> i32
          yield %8 : i32
        }
        %4 = emitc.expression  : () -> i32 {
          %8 = "emitc.constant"() <{value = 4096 : i32}> : () -> i32
          yield %8 : i32
        }
        %5 = emitc.expression  : () -> !emitc.size_t {
          %8 = "emitc.constant"() <{value = 18 : index}> : () -> !emitc.size_t
          yield %8 : !emitc.size_t
        }
        %6 = emitc.literal "get_compile_time_arg_val(1)" : !emitc.opaque<"::tt::CB">
        %7 = emitc.literal "get_compile_time_arg_val(3)" : i32
        emitc.for %arg0 = %2 to %0 step %1  : !emitc.size_t {
          call_opaque "cb_reserve_back"(%6, %3) : (!emitc.opaque<"::tt::CB">, i32) -> ()
          %8 = expression %5, %5, %7 : (!emitc.size_t, !emitc.size_t, i32) -> i64 {
            %10 = call_opaque "get_noc_addr"(%5, %5, %7) : (!emitc.size_t, !emitc.size_t, i32) -> i64
            yield %10 : i64
          }
          %9 = expression %6 : (!emitc.opaque<"::tt::CB">) -> i32 {
            %10 = call_opaque "get_write_ptr"(%6) : (!emitc.opaque<"::tt::CB">) -> i32
            yield %10 : i32
          }
          call_opaque "noc_async_read"(%8, %9, %4) : (i64, i32, i32) -> ()
          call_opaque "noc_async_read_barrier"() : () -> ()
          call_opaque "cb_push_back"(%6, %3) : (!emitc.opaque<"::tt::CB">, i32) -> ()
        }
        return
      }
      func.func private @compute_kernel8() attributes {ttkernel.arg_spec = #ttkernel.arg_spec< ct_args = [<arg_type = cb_port, operand_index = 0>, <arg_type = cb_port, operand_index = 1>, <arg_type = cb_port, operand_index = 2>]>, ttkernel.thread = #ttkernel.thread<compute>} {
        %0 = emitc.expression  : () -> !emitc.size_t {
          %8 = "emitc.constant"() <{value = 2 : index}> : () -> !emitc.size_t
          yield %8 : !emitc.size_t
        }
        %1 = emitc.expression  : () -> !emitc.size_t {
          %8 = "emitc.constant"() <{value = 1 : index}> : () -> !emitc.size_t
          yield %8 : !emitc.size_t
        }
        %2 = emitc.expression  : () -> !emitc.size_t {
          %8 = "emitc.constant"() <{value = 0 : index}> : () -> !emitc.size_t
          yield %8 : !emitc.size_t
        }
        %3 = emitc.expression  : () -> i32 {
          %8 = "emitc.constant"() <{value = 2 : i32}> : () -> i32
          yield %8 : i32
        }
        %4 = emitc.expression  : () -> i32 {
          %8 = "emitc.constant"() <{value = 1 : i32}> : () -> i32
          yield %8 : i32
        }
        %5 = emitc.literal "get_compile_time_arg_val(0)" : !emitc.opaque<"::tt::CB">
        %6 = emitc.literal "get_compile_time_arg_val(1)" : !emitc.opaque<"::tt::CB">
        %7 = emitc.literal "get_compile_time_arg_val(2)" : !emitc.opaque<"::tt::CB">
        emitc.call_opaque "compute_kernel_hw_startup"(%5, %6, %7) : (!emitc.opaque<"::tt::CB">, !emitc.opaque<"::tt::CB">, !emitc.opaque<"::tt::CB">) -> ()
        emitc.for %arg0 = %2 to %0 step %1  : !emitc.size_t {
          call_opaque "cb_wait_front"(%5, %3) : (!emitc.opaque<"::tt::CB">, i32) -> ()
          call_opaque "cb_wait_front"(%6, %4) : (!emitc.opaque<"::tt::CB">, i32) -> ()
          call_opaque "cb_reserve_back"(%7, %3) : (!emitc.opaque<"::tt::CB">, i32) -> ()
          call_opaque "tile_regs_acquire"() : () -> ()
          %8 = expression %arg0, %2 : (!emitc.size_t, !emitc.size_t) -> i1 {
            %10 = cast %2 : !emitc.size_t to !emitc.ptrdiff_t
            %11 = cast %arg0 : !emitc.size_t to !emitc.ptrdiff_t
            %12 = cmp ne, %11, %10 : (!emitc.ptrdiff_t, !emitc.ptrdiff_t) -> i1
            yield %12 : i1
          }
          if %8 {
            call_opaque "copy_tile_init"(%7) : (!emitc.opaque<"::tt::CB">) -> ()
            for %arg1 = %2 to %0 step %1  : !emitc.size_t {
              call_opaque "copy_tile"(%7, %arg1, %arg1) : (!emitc.opaque<"::tt::CB">, !emitc.size_t, !emitc.size_t) -> ()
            }
          }
          %9 = expression %arg0, %1 : (!emitc.size_t, !emitc.size_t) -> i1 {
            %10 = cast %1 : !emitc.size_t to !emitc.ptrdiff_t
            %11 = cast %arg0 : !emitc.size_t to !emitc.ptrdiff_t
            %12 = cmp ne, %11, %10 : (!emitc.ptrdiff_t, !emitc.ptrdiff_t) -> i1
            yield %12 : i1
          }
          for %arg1 = %2 to %0 step %1  : !emitc.size_t {
            call_opaque "reduce_init"(%5, %6, %7) {template_args = [#emitc.opaque<"PoolType::SUM">, #emitc.opaque<"ReduceDim::REDUCE_ROW">, #emitc.opaque<"false">]} : (!emitc.opaque<"::tt::CB">, !emitc.opaque<"::tt::CB">, !emitc.opaque<"::tt::CB">) -> ()
            call_opaque "reduce_tile"(%5, %6, %arg1, %2, %arg1) {template_args = [#emitc.opaque<"PoolType::SUM">, #emitc.opaque<"ReduceDim::REDUCE_ROW">, #emitc.opaque<"false">]} : (!emitc.opaque<"::tt::CB">, !emitc.opaque<"::tt::CB">, !emitc.size_t, !emitc.size_t, !emitc.size_t) -> ()
            if %9 {
              call_opaque "reduce_uninit"() : () -> ()
            }
          }
          call_opaque "tile_regs_commit"() : () -> ()
          call_opaque "tile_regs_wait"() : () -> ()
          for %arg1 = %2 to %0 step %1  : !emitc.size_t {
            call_opaque "pack_tile"(%arg1, %7, %arg1) {template_args = [true]} : (!emitc.size_t, !emitc.opaque<"::tt::CB">, !emitc.size_t) -> ()
          }
          call_opaque "tile_regs_release"() : () -> ()
          call_opaque "cb_wait_front"(%7, %3) : (!emitc.opaque<"::tt::CB">, i32) -> ()
          call_opaque "cb_pop_front"(%5, %3) : (!emitc.opaque<"::tt::CB">, i32) -> ()
          call_opaque "cb_pop_front"(%6, %4) : (!emitc.opaque<"::tt::CB">, i32) -> ()
          call_opaque "cb_push_back"(%7, %3) : (!emitc.opaque<"::tt::CB">, i32) -> ()
          call_opaque "cb_pop_front"(%7, %3) : (!emitc.opaque<"::tt::CB">, i32) -> ()
        }
        return
      }
      func.func private @datamovement_kernel9() attributes {ttkernel.arg_spec = #ttkernel.arg_spec< ct_args = [<arg_type = cb_port, operand_index = 0>, <arg_type = cb_port, operand_index = 1>]>, ttkernel.thread = #ttkernel.thread<noc>} {
        %0 = emitc.expression  : () -> i32 {
          %2 = "emitc.constant"() <{value = 2 : i32}> : () -> i32
          yield %2 : i32
        }
        %1 = emitc.literal "get_compile_time_arg_val(0)" : !emitc.opaque<"::tt::CB">
        emitc.call_opaque "cb_reserve_back"(%1, %0) : (!emitc.opaque<"::tt::CB">, i32) -> ()
        emitc.call_opaque "cb_push_back"(%1, %0) : (!emitc.opaque<"::tt::CB">, i32) -> ()
        return
      }
      func.func private @datamovement_kernel10() attributes {ttkernel.arg_spec = #ttkernel.arg_spec< ct_args = [<arg_type = cb_port, operand_index = 0>, <arg_type = cb_port, operand_index = 1>]>, ttkernel.thread = #ttkernel.thread<noc>} {
        %0 = emitc.expression  : () -> i32 {
          %2 = "emitc.constant"() <{value = 2 : i32}> : () -> i32
          yield %2 : i32
        }
        %1 = emitc.literal "get_compile_time_arg_val(1)" : !emitc.opaque<"::tt::CB">
        emitc.call_opaque "cb_wait_front"(%1, %0) : (!emitc.opaque<"::tt::CB">, i32) -> ()
        emitc.call_opaque "cb_pop_front"(%1, %0) : (!emitc.opaque<"::tt::CB">, i32) -> ()
        return
      }
      func.func private @compute_kernel11() attributes {ttkernel.arg_spec = #ttkernel.arg_spec< ct_args = [<arg_type = cb_port, operand_index = 0>, <arg_type = cb_port, operand_index = 1>]>, ttkernel.thread = #ttkernel.thread<compute>} {
        %0 = emitc.expression  : () -> i32 {
          %4 = "emitc.constant"() <{value = 2 : i32}> : () -> i32
          yield %4 : i32
        }
        %1 = emitc.expression  : () -> i32 {
          %4 = "emitc.constant"() <{value = 1 : i32}> : () -> i32
          yield %4 : i32
        }
        %2 = emitc.literal "get_compile_time_arg_val(0)" : !emitc.opaque<"::tt::CB">
        %3 = emitc.literal "get_compile_time_arg_val(1)" : !emitc.opaque<"::tt::CB">
        emitc.call_opaque "cb_wait_front"(%2, %0) : (!emitc.opaque<"::tt::CB">, i32) -> ()
        emitc.call_opaque "cb_reserve_back"(%3, %0) : (!emitc.opaque<"::tt::CB">, i32) -> ()
        emitc.call_opaque "compute_kernel_hw_startup"(%2, %3) : (!emitc.opaque<"::tt::CB">, !emitc.opaque<"::tt::CB">) -> ()
        emitc.call_opaque "untilize_init"(%2) : (!emitc.opaque<"::tt::CB">) -> ()
        emitc.call_opaque "experimental::untilize_block"(%2, %3, %0, %1) : (!emitc.opaque<"::tt::CB">, !emitc.opaque<"::tt::CB">, i32, i32) -> ()
        emitc.call_opaque "cb_pop_front"(%2, %0) : (!emitc.opaque<"::tt::CB">, i32) -> ()
        emitc.call_opaque "cb_push_back"(%3, %0) : (!emitc.opaque<"::tt::CB">, i32) -> ()
        return
      }
    }
  }
}

