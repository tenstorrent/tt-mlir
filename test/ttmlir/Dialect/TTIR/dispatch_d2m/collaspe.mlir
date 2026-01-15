#l1 = #ttnn.buffer_type<l1>
#system_desc = #ttcore.system_desc<[{role = host, target_triple = "x86_64-pc-linux"}], [{arch = <wormhole_b0>, grid = 8x8, coord_translation_offsets = 18x18, l1_size = 1499136, num_dram_channels = 12, dram_channel_size = 1073741824, noc_l1_address_align_bytes = 16, pcie_address_align_bytes = 32, noc_dram_address_align_bytes = 32, l1_unreserved_base = 102656, erisc_l1_unreserved_base = 98304, dram_unreserved_base = 32, dram_unreserved_end = 1073125888, supported_data_types = [<f32>, <f16>, <bf16>, <bfp_f8>, <bfp_bf8>, <bfp_f4>, <bfp_bf4>, <bfp_f2>, <bfp_bf2>, <u32>, <u16>, <u8>, <si32>], supported_tile_sizes = [ 4x16,  16x16,  32x16,  4x32,  16x32,  32x32], dst_physical_size_tiles = 16, num_cbs = 32, num_compute_threads = 1, num_datamovement_threads = 2}], [0], [1 : i32], [ 0x0x0x0]>
#ttnn_layout = #ttnn.ttnn_layout<(d0, d1) -> (d0, d1), <1x1>, memref<1x1x!ttcore.tile<32x32, f32>, #l1>, <block_sharded>, exactGrid = true>

module {
  func.func @ttnn_graph(%arg0: tensor<64x64xbf16>, %arg1: tensor<64x64xbf16>) -> tensor<32x32xf32, #ttnn_layout> {
    %0 = "ttir.dispatch_d2m"(%arg0, %arg1) {subgraph = @d2m_subgraph} : (tensor<64x64xbf16>, tensor<64x64xbf16>) -> tensor<32x32xf32, #ttnn_layout>
    return %0 : tensor<32x32xf32, #ttnn_layout>
  }

  module @d2m_subgraph {
    ttcore.device_module {
      builtin.module attributes {ttcore.system_desc = #system_desc} {
        ttcore.device @default_device = <workerGrid = #ttcore.grid<8x8, (d0, d1) -> (0, d0, d1)>, l1Map = (d0, d1, d2)[s0] -> (0, d0, d1, d2 + s0), dramMap = (d0, d1, d2)[s0, s1, s2, s3, s4, s5, s6] -> (0, 0, (((d0 * s1) * (s2 * (s3 * s6)) + d1 * (s2 * (s3 * s6)) + d2) floordiv s4) mod 12, ((((d0 * s1) * (s2 * (s3 * s6)) + d1 * (s2 * (s3 * s6)) + d2) floordiv s4) floordiv 12) * s4 + ((d0 * s1) * (s2 * (s3 * s6)) + d1 * (s2 * (s3 * s6)) + d2) mod s4 + s5), meshShape = , chipIds = [0]>
        func.func @d2m_subgraph(%arg0: tensor<32x32xf32, #ttnn_layout>) -> tensor<32x32xf32, #ttnn_layout> {
          %0 = "ttnn.get_device"() <{mesh_offset = #ttnn<mesh_offset 0x0>, mesh_shape = #ttnn<mesh_shape 1x1>}> : () -> !ttnn.device
          %1 = "ttnn.empty"(%0) <{dtype = #ttcore.supportedDataTypes<f32>, layout = #ttnn.layout<tile>, memory_config = #ttnn.memory_config<#l1, <block_sharded>, #ttnn.shard_spec<<[#ttnn.core_range<(0,0), (0,0)>]>, <32x32>, <row_major>>>, shape = #ttnn.shape<32x32>}> : (!ttnn.device) -> tensor<32x32xf32, #ttnn_layout>
          "ttnn.generic"(%arg0, %1) <{program = #ttnn.program<kernels = [#ttnn.read_kernel<symbol_ref = @datamovement_kernel0, core_ranges = <[#ttnn.core_range<(0,0), (0,0)>]>, ct_args = [#ttnn.kernel_arg_cb_buffer_index<0>, #ttnn.kernel_arg_cb_buffer_index<1>], common_rt_args = [], rt_args = []>, #ttnn.write_kernel<symbol_ref = @datamovement_kernel1, core_ranges = <[#ttnn.core_range<(0,0), (0,0)>]>, ct_args = [#ttnn.kernel_arg_cb_buffer_index<0>, #ttnn.kernel_arg_cb_buffer_index<1>], common_rt_args = [], rt_args = []>, #ttnn.compute_kernel<symbol_ref = @compute_kernel2, core_ranges = <[#ttnn.core_range<(0,0), (0,0)>]>, math_fidelity = hifi4, fp32_dest_acc_en = false, dst_full_sync_en = false, unpack_to_dest_modes = [default], bfp8_pack_precise = false, math_approx_mode = false, ct_args = [#ttnn.kernel_arg_cb_buffer_index<0>, #ttnn.kernel_arg_cb_buffer_index<1>], common_rt_args = [], rt_args = []>], cbs = [<total_size = 4096, core_ranges = <[#ttnn.core_range<(0,0), (0,0)>]>, formats = [<buffer_index = 0, dtype = f32, page_size = 4096>], buffer = #ttnn.kernel_cb_global_buffer_address_of_tensor<0>>, <total_size = 4096, core_ranges = <[#ttnn.core_range<(0,0), (0,0)>]>, formats = [<buffer_index = 1, dtype = f32, page_size = 4096>], buffer = #ttnn.kernel_cb_global_buffer_address_of_tensor<1>>], semaphores = []>}> : (tensor<32x32xf32, #ttnn_layout>, tensor<32x32xf32, #ttnn_layout>) -> ()
          "ttnn.deallocate"(%arg0) <{force = false}> : (tensor<32x32xf32, #ttnn_layout>) -> ()
          return %1 : tensor<32x32xf32, #ttnn_layout>
        }
        func.func private @datamovement_kernel0() attributes {ttkernel.arg_spec = #ttkernel.arg_spec< ct_args = [<arg_type = cb_port, operand_index = 0>, <arg_type = cb_port, operand_index = 1>]>, ttkernel.thread = #ttkernel.thread<noc>} {
          %0 = emitc.expression  : () -> i32 {
            %2 = "emitc.constant"() <{value = 1 : i32}> : () -> i32
            yield %2 : i32
          }
          %1 = emitc.literal "get_compile_time_arg_val(0)" : !emitc.opaque<"::tt::CB">
          emitc.call_opaque "cb_reserve_back"(%1, %0) : (!emitc.opaque<"::tt::CB">, i32) -> ()
          emitc.call_opaque "cb_push_back"(%1, %0) : (!emitc.opaque<"::tt::CB">, i32) -> ()
          return
        }
        func.func private @datamovement_kernel1() attributes {ttkernel.arg_spec = #ttkernel.arg_spec< ct_args = [<arg_type = cb_port, operand_index = 0>, <arg_type = cb_port, operand_index = 1>]>, ttkernel.thread = #ttkernel.thread<noc>} {
          %0 = emitc.expression  : () -> i32 {
            %2 = "emitc.constant"() <{value = 1 : i32}> : () -> i32
            yield %2 : i32
          }
          %1 = emitc.literal "get_compile_time_arg_val(1)" : !emitc.opaque<"::tt::CB">
          emitc.call_opaque "cb_wait_front"(%1, %0) : (!emitc.opaque<"::tt::CB">, i32) -> ()
          emitc.call_opaque "cb_pop_front"(%1, %0) : (!emitc.opaque<"::tt::CB">, i32) -> ()
          return
        }
        func.func private @compute_kernel2() attributes {ttkernel.arg_spec = #ttkernel.arg_spec< ct_args = [<arg_type = cb_port, operand_index = 0>, <arg_type = cb_port, operand_index = 1>]>, ttkernel.thread = #ttkernel.thread<compute>} {
          %0 = emitc.expression  : () -> !emitc.size_t {
            %4 = "emitc.constant"() <{value = 0 : index}> : () -> !emitc.size_t
            yield %4 : !emitc.size_t
          }
          %1 = emitc.expression  : () -> i32 {
            %4 = "emitc.constant"() <{value = 1 : i32}> : () -> i32
            yield %4 : i32
          }
          %2 = emitc.literal "get_compile_time_arg_val(0)" : !emitc.opaque<"::tt::CB">
          %3 = emitc.literal "get_compile_time_arg_val(1)" : !emitc.opaque<"::tt::CB">
          emitc.call_opaque "init_sfpu"(%2, %3) : (!emitc.opaque<"::tt::CB">, !emitc.opaque<"::tt::CB">) -> ()
          emitc.call_opaque "cb_wait_front"(%2, %1) : (!emitc.opaque<"::tt::CB">, i32) -> ()
          emitc.call_opaque "cb_reserve_back"(%3, %1) : (!emitc.opaque<"::tt::CB">, i32) -> ()
          emitc.call_opaque "tile_regs_acquire"() : () -> ()
          emitc.call_opaque "copy_tile_init"(%2) : (!emitc.opaque<"::tt::CB">) -> ()
          emitc.call_opaque "copy_tile"(%2, %0, %0) : (!emitc.opaque<"::tt::CB">, !emitc.size_t, !emitc.size_t) -> ()
          emitc.call_opaque "abs_tile_init"() : () -> ()
          emitc.call_opaque "abs_tile"(%0) : (!emitc.size_t) -> ()
          emitc.call_opaque "tile_regs_commit"() : () -> ()
          emitc.call_opaque "tile_regs_wait"() : () -> ()
          emitc.call_opaque "pack_tile"(%0, %3, %0) {template_args = [true]} : (!emitc.size_t, !emitc.opaque<"::tt::CB">, !emitc.size_t) -> ()
          emitc.call_opaque "tile_regs_release"() : () -> ()
          emitc.call_opaque "cb_pop_front"(%2, %1) : (!emitc.opaque<"::tt::CB">, i32) -> ()
          emitc.call_opaque "cb_push_back"(%3, %1) : (!emitc.opaque<"::tt::CB">, i32) -> ()
          return
        }
      }
    }
  }
}
