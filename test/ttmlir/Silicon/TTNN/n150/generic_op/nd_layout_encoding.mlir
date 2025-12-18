// REQUIRES: opmodel, perf
// RUN: ttmlir-opt --ttcore-register-device="system-desc-path=%system_desc_path%" -o %t.mlir %s
// RUN: ttmlir-translate --ttnn-to-flatbuffer -o %t.ttnn %t.mlir
// RUN: ttrt run %t.ttnn
// Temporary test for ttnn_nd_layout support in runtime. When TTNN JIT support is brought up (#5832), this test will be removed

#l1 = #ttnn.buffer_type<l1>
#layout = #ttnn.ttnn_nd_layout<<1x1>, memref<1x1x!ttcore.tile<32x32, bf16>, #l1>, <block_sharded>, <row_major>, <grid_2d>>
#nd_shard_spec = #ttnn.nd_shard_spec<<[#ttnn.core_range<(0,0), (0,0)>]>, <32x32>, <row_major>, <grid_2d>>

module {
    func.func @test_lowered(%arg0: tensor<32x32xbf16, #layout>) -> tensor<32x32xbf16, #layout> {
        %0 = "ttnn.get_device"() <{mesh_offset = #ttnn<mesh_offset 0x0>, mesh_shape = #ttnn<mesh_shape 1x1>}> : () -> !ttnn.device
        %1 = "ttnn.empty"(%0) <{dtype = #ttcore.supportedDataTypes<bf16>, layout = #ttnn.layout<tile>, memory_config = #ttnn.memory_config<#l1, <block_sharded>, #nd_shard_spec>, shape = #ttnn.shape<32x32>}> : (!ttnn.device) -> tensor<32x32xbf16, #layout>
        "ttnn.generic"(%arg0, %1) <{program = #ttnn.program<kernels = [#ttnn.read_kernel<symbol_ref = @datamovement_kernel0, core_ranges = <[#ttnn.core_range<(0,0), (0,0)>]>, ct_args = [#ttnn.kernel_arg_cb_buffer_index<0>, #ttnn.kernel_arg_cb_buffer_index<1>], common_rt_args = []>, #ttnn.write_kernel<symbol_ref = @datamovement_kernel1, core_ranges = <[#ttnn.core_range<(0,0), (0,0)>]>, ct_args = [#ttnn.kernel_arg_cb_buffer_index<0>, #ttnn.kernel_arg_cb_buffer_index<1>], common_rt_args = []>, #ttnn.compute_kernel<symbol_ref = @compute_kernel2, core_ranges = <[#ttnn.core_range<(0,0), (0,0)>]>, math_fidelity = hifi4, fp32_dest_acc_en = false, dst_full_sync_en = false, unpack_to_dest_modes = [default], bfp8_pack_precise = false, math_approx_mode = false, ct_args = [#ttnn.kernel_arg_cb_buffer_index<0>, #ttnn.kernel_arg_cb_buffer_index<1>], common_rt_args = []>], cbs = [<total_size = 2048, core_ranges = <[#ttnn.core_range<(0,0), (0,0)>]>, formats = [<buffer_index = 0, dtype = bf16, page_size = 2048>], buffer = #ttnn.kernel_cb_global_buffer_address_of_tensor<0>>, <total_size = 2048, core_ranges = <[#ttnn.core_range<(0,0), (0,0)>]>, formats = [<buffer_index = 1, dtype = bf16, page_size = 2048>], buffer = #ttnn.kernel_cb_global_buffer_address_of_tensor<1>>], semaphores = []>}> : (tensor<32x32xbf16, #layout>, tensor<32x32xbf16, #layout>) -> ()
        return %1 : tensor<32x32xbf16, #layout>
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
