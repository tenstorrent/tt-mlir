// RUN: ttmlir-translate --ttmetal-to-flatbuffer -o output.ttm emitc_reproducer.mlir && ttrt run --init=ones output.ttm
// Expected output: 4096x4096 matrix with all elements set to 2.0
// Actual output: 4096x4096 matrix with some of the elements set to 2.0, some set to 1.0.
#dram = #ttcore.memory_space<dram>
#l1 = #ttcore.memory_space<l1>
#system_desc = #ttcore.system_desc<[{role = host, target_triple = "x86_64-pc-linux"}], [{arch = <wormhole_b0>, grid = 8x8, coord_translation_offsets = 18x18, l1_size = 1499136, num_dram_channels = 12, dram_channel_size = 1073741824, noc_l1_address_align_bytes = 16, pcie_address_align_bytes = 32, noc_dram_address_align_bytes = 32, l1_unreserved_base = 99904, erisc_l1_unreserved_base = 69632, dram_unreserved_base = 32, dram_unreserved_end = 1073184000, physical_helper_cores = {dram = [ 0x0,  0x1,  0x2,  0x3,  0x4,  0x5,  0x6,  0x7,  0x8,  0x9,  0x10,  0x11] eth = [ 17x21,  17x25] eth_inactive = [ 16x18,  16x19,  16x20,  16x21,  16x22,  16x23,  16x24,  16x25,  17x19,  17x20,  17x22,  17x23,  17x24]}, supported_data_types = [<f32>, <f16>, <bf16>, <bfp_f8>, <bfp_bf8>, <bfp_f4>, <bfp_bf4>, <bfp_f2>, <bfp_bf2>, <u32>, <u16>, <u8>, <si32>], supported_tile_sizes = [ 4x16,  16x16,  32x16,  4x32,  16x32,  32x32], dst_register_size_tiles = 8, num_cbs = 32, num_compute_threads = 1, num_datamovement_threads = 2}, {arch = <wormhole_b0>, grid = 8x8, coord_translation_offsets = 18x18, l1_size = 1499136, num_dram_channels = 12, dram_channel_size = 1073741824, noc_l1_address_align_bytes = 16, pcie_address_align_bytes = 32, noc_dram_address_align_bytes = 32, l1_unreserved_base = 99904, erisc_l1_unreserved_base = 69632, dram_unreserved_base = 32, dram_unreserved_end = 1073192352, physical_helper_cores = {dram = [ 0x0,  0x1,  0x2,  0x3,  0x4,  0x5,  0x6,  0x7,  0x8,  0x9,  0x10,  0x11] eth = [ 16x25] eth_inactive = [ 16x19,  16x20,  16x21,  16x22,  16x23,  16x24,  17x18,  17x19,  17x20,  17x21,  17x22,  17x23,  17x24,  17x25]}, supported_data_types = [<f32>, <f16>, <bf16>, <bfp_f8>, <bfp_bf8>, <bfp_f4>, <bfp_bf4>, <bfp_f2>, <bfp_bf2>, <u32>, <u16>, <u8>, <si32>], supported_tile_sizes = [ 4x16,  16x16,  32x16,  4x32,  16x32,  32x32], dst_register_size_tiles = 8, num_cbs = 32, num_compute_threads = 1, num_datamovement_threads = 2}], [0, 1], [3 : i32, 0 : i32], [ 0x0x0x0]>
module attributes {polyblocks.cpu_target_info = {l1_cache_associativity = 8 : i8, l1_cache_size = 32768 : i32, l2_cache_associativity = 8 : i8, l2_cache_line_size = 64 : i32, l2_cache_size = 524288 : i32, num_cores = 32 : i32, omp_num_threads = 32 : i32, simd_width = 256 : i32}, polyblocks.target = "tenstorrent", polyblocks.tenstorrent_target_info = {dram_alignment_bytes = 128 : i32, l1_scratchpad_size_bytes = 1536000 : i32, l1_unreserved_base = 99904 : i32, num_cores = 128 : i32}, torch.target = "tenstorrent", ttcore.system_desc = #system_desc} {
  func.func private @Compute_kernel_0() attributes {arg_spec = #ttkernel.arg_spec< ct_args = [<arg_type = cb_port, operand_index = 0>, <arg_type = cb_port, operand_index = 1>, <arg_type = cb_port, operand_index = 2>]>, ttkernel.thread = #ttkernel.thread<compute>} {
    %0 = "emitc.constant"() <{value = 64 : i32}> : () -> i32
    %1 = "emitc.constant"() <{value = 0 : index}> : () -> !emitc.size_t
    %2 = "emitc.constant"() <{value = 8 : index}> : () -> !emitc.size_t
    %3 = "emitc.constant"() <{value = 1 : index}> : () -> !emitc.size_t
    %4 = "emitc.constant"() <{value = 2 : index}> : () -> !emitc.size_t
    // %5 = 0 write
    %5 = emitc.literal "get_compile_time_arg_val(0)" : !emitc.opaque<"::tt::CB">
    emitc.call_opaque "cb_reserve_back"(%5, %0) : (!emitc.opaque<"::tt::CB">, i32) -> ()
    // %6 = 1 read
    %6 = emitc.literal "get_compile_time_arg_val(1)" : !emitc.opaque<"::tt::CB">
    %dummy = emitc.literal "get_compile_time_arg_val(2)" : !emitc.opaque<"::tt::CB">
    emitc.call_opaque "binary_op_init_common"(%6, %6, %5) : (!emitc.opaque<"::tt::CB">, !emitc.opaque<"::tt::CB">, !emitc.opaque<"::tt::CB">) -> ()
    emitc.call_opaque "add_tiles_init"(%6, %6) : (!emitc.opaque<"::tt::CB">, !emitc.opaque<"::tt::CB">) -> ()
    emitc.for %arg0 = %1 to %4 step %3  : !emitc.size_t {
      for %arg1 = %1 to %4 step %3  : !emitc.size_t {
        call_opaque "cb_wait_front"(%6, %0) : (!emitc.opaque<"::tt::CB">, i32) -> ()
        for %arg2 = %1 to %2 step %3  : !emitc.size_t {
          %7 = mul %arg2, %2 : (!emitc.size_t, !emitc.size_t) -> !emitc.size_t
          for %arg3 = %1 to %2 step %3  : !emitc.size_t {
            %8 = add %7, %arg3 : (!emitc.size_t, !emitc.size_t) -> !emitc.size_t

            call_opaque "tile_regs_acquire"() : () -> ()
            call_opaque "add_tiles"(%6, %6, %8, %8, %1) : (!emitc.opaque<"::tt::CB">, !emitc.opaque<"::tt::CB">, !emitc.size_t, !emitc.size_t, !emitc.size_t) -> ()
            call_opaque "tile_regs_commit"() : () -> ()

            call_opaque "tile_regs_wait"() : () -> ()
            call_opaque "pack_tile"(%1, %5, %8) {template_args = [true]} : (!emitc.size_t, !emitc.opaque<"::tt::CB">, !emitc.size_t) -> ()
            call_opaque "tile_regs_release"() : () -> ()
          }
        }

        // Dummy CB Workaround: Stall the unpack thread until the pack thread is done
        call_opaque "cb_wait_front"(%dummy, %0) : (!emitc.opaque<"::tt::CB">, i32) -> ()
        call_opaque "cb_pop_front"(%dummy, %0) : (!emitc.opaque<"::tt::CB">, i32) -> ()

        call_opaque "cb_push_back"(%5, %0) : (!emitc.opaque<"::tt::CB">, i32) -> ()
        call_opaque "cb_reserve_back"(%5, %0) : (!emitc.opaque<"::tt::CB">, i32) -> ()

        // Dummy CB Workaround: Signal unpack thread that pack thread is done
        call_opaque "cb_reserve_back"(%dummy, %0) : (!emitc.opaque<"::tt::CB">, i32) -> ()
        call_opaque "cb_push_back"(%dummy, %0) : (!emitc.opaque<"::tt::CB">, i32) -> ()

        call_opaque "cb_pop_front"(%6, %0) : (!emitc.opaque<"::tt::CB">, i32) -> ()
      }
    }
    return
  }
  func.func private @DataFlow1_kernel_0() attributes {arg_spec = #ttkernel.arg_spec< ct_args = [<arg_type = cb_port, operand_index = 0>, <arg_type = buffer_address, operand_index = 1>]>, ttkernel.thread = #ttkernel.thread<noc>} {
    %0 = "emitc.constant"() <{value = 128 : index}> : () -> !emitc.size_t
    %1 = "emitc.constant"() <{value = 4096 : index}> : () -> !emitc.size_t
    %2 = "emitc.constant"() <{value = 4096 : i32}> : () -> i32
    %3 = "emitc.constant"() <{value = true}> : () -> i1
    %4 = "emitc.constant"() <{value = 64 : i32}> : () -> i32
    %5 = "emitc.constant"() <{value = 2 : index}> : () -> !emitc.size_t
    %6 = "emitc.constant"() <{value = 1 : index}> : () -> !emitc.size_t
    %7 = "emitc.constant"() <{value = 0 : index}> : () -> !emitc.size_t
    %8 = "emitc.constant"() <{value = 8 : index}> : () -> !emitc.size_t
    %9 = "emitc.constant"() <{value = 16 : index}> : () -> !emitc.size_t
    %10 = "emitc.constant"() <{value = 18 : index}> : () -> !emitc.size_t
    %11 = "emitc.constant"() <{value = #emitc.opaque<"my_x[noc_index]">}> : () -> !emitc.size_t
    %12 = emitc.sub %11, %10 : (!emitc.size_t, !emitc.size_t) -> !emitc.size_t
    %13 = "emitc.constant"() <{value = #emitc.opaque<"my_y[noc_index]">}> : () -> !emitc.size_t
    %14 = emitc.sub %13, %10 : (!emitc.size_t, !emitc.size_t) -> !emitc.size_t
    %15 = emitc.literal "get_compile_time_arg_val(0)" : !emitc.opaque<"::tt::CB">
    %16 = emitc.mul %12, %9 : (!emitc.size_t, !emitc.size_t) -> !emitc.size_t
    %17 = emitc.mul %14, %9 : (!emitc.size_t, !emitc.size_t) -> !emitc.size_t
    %18 = emitc.literal "get_compile_time_arg_val(1)" : i32
    emitc.for %arg0 = %7 to %5 step %6  : !emitc.size_t {
      %19 = mul %arg0, %8 : (!emitc.size_t, !emitc.size_t) -> !emitc.size_t
      for %arg1 = %7 to %5 step %6  : !emitc.size_t {
        call_opaque "cb_wait_front"(%15, %4) : (!emitc.opaque<"::tt::CB">, i32) -> ()
        %20 = mul %arg1, %8 : (!emitc.size_t, !emitc.size_t) -> !emitc.size_t
        for %arg2 = %7 to %8 step %6  : !emitc.size_t {
          %21 = add %arg2, %16 : (!emitc.size_t, !emitc.size_t) -> !emitc.size_t
          %22 = add %21, %19 : (!emitc.size_t, !emitc.size_t) -> !emitc.size_t
          %23 = mul %arg2, %8 : (!emitc.size_t, !emitc.size_t) -> !emitc.size_t
          %24 = mul %22, %0 : (!emitc.size_t, !emitc.size_t) -> !emitc.size_t
          for %arg3 = %7 to %8 step %6  : !emitc.size_t {
            %25 = add %arg3, %17 : (!emitc.size_t, !emitc.size_t) -> !emitc.size_t
            %26 = add %25, %20 : (!emitc.size_t, !emitc.size_t) -> !emitc.size_t
            %27 = call_opaque "get_dataformat"(%15) : (!emitc.opaque<"::tt::CB">) -> !emitc.opaque<"DataFormat">
            %28 = "emitc.variable"() <{value = #emitc.opaque<"">}> : () -> !emitc.lvalue<!emitc.opaque<"InterleavedAddrGenFast<true>">>
            %29 = "emitc.member"(%28) <{member = "bank_base_address"}> : (!emitc.lvalue<!emitc.opaque<"InterleavedAddrGenFast<true>">>) -> !emitc.lvalue<i32>
            %30 = "emitc.member"(%28) <{member = "page_size"}> : (!emitc.lvalue<!emitc.opaque<"InterleavedAddrGenFast<true>">>) -> !emitc.lvalue<i32>
            %31 = "emitc.member"(%28) <{member = "data_format"}> : (!emitc.lvalue<!emitc.opaque<"InterleavedAddrGenFast<true>">>) -> !emitc.lvalue<!emitc.opaque<"DataFormat">>
            assign %18 : i32 to %29 : <i32>
            assign %2 : i32 to %30 : <i32>
            assign %27 : !emitc.opaque<"DataFormat"> to %31 : <!emitc.opaque<"DataFormat">>
            %32 = load %28 : <!emitc.opaque<"InterleavedAddrGenFast<true>">>
            %33 = call_opaque "get_read_ptr"(%15) : (!emitc.opaque<"::tt::CB">) -> i32
            %34 = add %23, %arg3 : (!emitc.size_t, !emitc.size_t) -> !emitc.size_t
            %35 = mul %34, %1 : (!emitc.size_t, !emitc.size_t) -> !emitc.size_t
            %36 = cast %35 : !emitc.size_t to !emitc.ptrdiff_t
            %37 = cast %36 : !emitc.ptrdiff_t to i32
            %38 = cast %37 : i32 to ui32
            %39 = cast %33 : i32 to ui32
            %40 = add %38, %39 : (ui32, ui32) -> ui32
            %41 = cast %40 : ui32 to i32
            %42 = add %24, %26 : (!emitc.size_t, !emitc.size_t) -> !emitc.size_t
            %43 = cast %42 : !emitc.size_t to !emitc.ptrdiff_t
            %44 = cast %43 : !emitc.ptrdiff_t to i32
            call_opaque "noc_async_write_tile"(%44, %32, %41) : (i32, !emitc.opaque<"InterleavedAddrGenFast<true>">, i32) -> ()
          }
        }
        call_opaque "noc_async_write_barrier"() : () -> ()
        call_opaque "noc_async_full_barrier"() : () -> ()
        call_opaque "cb_pop_front"(%15, %4) : (!emitc.opaque<"::tt::CB">, i32) -> ()
      }
    }
    return
  }
  func.func private @DataFlow0_kernel_0() attributes {arg_spec = #ttkernel.arg_spec< ct_args = [<arg_type = cb_port, operand_index = 1>, <arg_type = buffer_address, operand_index = 0>]>, ttkernel.thread = #ttkernel.thread<noc>} {
    %0 = "emitc.constant"() <{value = 128 : index}> : () -> !emitc.size_t
    %1 = "emitc.constant"() <{value = 4096 : index}> : () -> !emitc.size_t
    %2 = "emitc.constant"() <{value = 4096 : i32}> : () -> i32
    %3 = "emitc.constant"() <{value = true}> : () -> i1
    %4 = "emitc.constant"() <{value = 64 : i32}> : () -> i32
    %5 = "emitc.constant"() <{value = 2 : index}> : () -> !emitc.size_t
    %6 = "emitc.constant"() <{value = 1 : index}> : () -> !emitc.size_t
    %7 = "emitc.constant"() <{value = 0 : index}> : () -> !emitc.size_t
    %8 = "emitc.constant"() <{value = 8 : index}> : () -> !emitc.size_t
    %9 = "emitc.constant"() <{value = 16 : index}> : () -> !emitc.size_t
    %10 = "emitc.constant"() <{value = 18 : index}> : () -> !emitc.size_t
    %11 = "emitc.constant"() <{value = #emitc.opaque<"my_x[noc_index]">}> : () -> !emitc.size_t
    %12 = emitc.sub %11, %10 : (!emitc.size_t, !emitc.size_t) -> !emitc.size_t
    %13 = "emitc.constant"() <{value = #emitc.opaque<"my_y[noc_index]">}> : () -> !emitc.size_t
    %14 = emitc.sub %13, %10 : (!emitc.size_t, !emitc.size_t) -> !emitc.size_t
    %15 = emitc.literal "get_compile_time_arg_val(0)" : !emitc.opaque<"::tt::CB">
    emitc.call_opaque "cb_reserve_back"(%15, %4) : (!emitc.opaque<"::tt::CB">, i32) -> ()
    %16 = emitc.mul %12, %9 : (!emitc.size_t, !emitc.size_t) -> !emitc.size_t
    %17 = emitc.mul %14, %9 : (!emitc.size_t, !emitc.size_t) -> !emitc.size_t
    %18 = emitc.literal "get_compile_time_arg_val(1)" : i32
    emitc.for %arg0 = %7 to %5 step %6  : !emitc.size_t {
      %19 = mul %arg0, %8 : (!emitc.size_t, !emitc.size_t) -> !emitc.size_t
      for %arg1 = %7 to %5 step %6  : !emitc.size_t {
        %20 = mul %arg1, %8 : (!emitc.size_t, !emitc.size_t) -> !emitc.size_t
        for %arg2 = %7 to %8 step %6  : !emitc.size_t {
          %21 = add %arg2, %16 : (!emitc.size_t, !emitc.size_t) -> !emitc.size_t
          %22 = add %21, %19 : (!emitc.size_t, !emitc.size_t) -> !emitc.size_t
          %23 = mul %arg2, %8 : (!emitc.size_t, !emitc.size_t) -> !emitc.size_t
          %24 = mul %22, %0 : (!emitc.size_t, !emitc.size_t) -> !emitc.size_t
          for %arg3 = %7 to %8 step %6  : !emitc.size_t {
            %25 = add %arg3, %17 : (!emitc.size_t, !emitc.size_t) -> !emitc.size_t
            %26 = add %25, %20 : (!emitc.size_t, !emitc.size_t) -> !emitc.size_t
            %27 = call_opaque "get_dataformat"(%15) : (!emitc.opaque<"::tt::CB">) -> !emitc.opaque<"DataFormat">
            %28 = "emitc.variable"() <{value = #emitc.opaque<"">}> : () -> !emitc.lvalue<!emitc.opaque<"InterleavedAddrGenFast<true>">>
            %29 = "emitc.member"(%28) <{member = "bank_base_address"}> : (!emitc.lvalue<!emitc.opaque<"InterleavedAddrGenFast<true>">>) -> !emitc.lvalue<i32>
            %30 = "emitc.member"(%28) <{member = "page_size"}> : (!emitc.lvalue<!emitc.opaque<"InterleavedAddrGenFast<true>">>) -> !emitc.lvalue<i32>
            %31 = "emitc.member"(%28) <{member = "data_format"}> : (!emitc.lvalue<!emitc.opaque<"InterleavedAddrGenFast<true>">>) -> !emitc.lvalue<!emitc.opaque<"DataFormat">>
            assign %18 : i32 to %29 : <i32>
            assign %2 : i32 to %30 : <i32>
            assign %27 : !emitc.opaque<"DataFormat"> to %31 : <!emitc.opaque<"DataFormat">>
            %32 = load %28 : <!emitc.opaque<"InterleavedAddrGenFast<true>">>
            %33 = call_opaque "get_write_ptr"(%15) : (!emitc.opaque<"::tt::CB">) -> i32
            %34 = add %23, %arg3 : (!emitc.size_t, !emitc.size_t) -> !emitc.size_t
            %35 = mul %34, %1 : (!emitc.size_t, !emitc.size_t) -> !emitc.size_t
            %36 = cast %35 : !emitc.size_t to !emitc.ptrdiff_t
            %37 = cast %36 : !emitc.ptrdiff_t to i32
            %38 = cast %37 : i32 to ui32
            %39 = cast %33 : i32 to ui32
            %40 = add %38, %39 : (ui32, ui32) -> ui32
            %41 = cast %40 : ui32 to i32
            %42 = add %24, %26 : (!emitc.size_t, !emitc.size_t) -> !emitc.size_t
            %43 = cast %42 : !emitc.size_t to !emitc.ptrdiff_t
            %44 = cast %43 : !emitc.ptrdiff_t to i32
            call_opaque "noc_async_read_tile"(%44, %32, %41) : (i32, !emitc.opaque<"InterleavedAddrGenFast<true>">, i32) -> ()
          }
        }
        call_opaque "noc_async_read_barrier"() : () -> ()
        call_opaque "noc_async_full_barrier"() : () -> ()
        call_opaque "cb_push_back"(%15, %4) : (!emitc.opaque<"::tt::CB">, i32) -> ()
        call_opaque "cb_reserve_back"(%15, %4) : (!emitc.opaque<"::tt::CB">, i32) -> ()
      }
    }
    return
  }
  ttcore.device @default_device = <workerGrid = #ttcore.grid<8x8, (d0, d1) -> (0, d0, d1)>, l1Map = (d0, d1, d2)[s0] -> (0, d0, d1, d2 + s0), dramMap = (d0, d1, d2)[s0, s1, s2, s3, s4, s5] -> (0, 0, (((d0 * s1) * (s2 * s3) + d1 * (s2 * s3) + d2) floordiv s4) mod 12, ((d0 * s1) * (s2 * s3) + d1 * (s2 * s3) + d2) floordiv (s4 * 12) + ((d0 * s1) * (s2 * s3) + d1 * (s2 * s3) + d2) mod s4 + s5), meshShape = , chipIds = [0]>
  func.func @forward(%arg0: memref<4096x4096xf32>) -> memref<4096x4096xf32> attributes {polyblocks.entry_function, torch.inputs = "add"} {
    %0 = "ttmetal.create_buffer"() <{address = 136777760 : i64}> : () -> memref<128x128x!ttcore.tile<32x32, f32>, #dram>
    %o = "ttmetal.create_buffer"() <{address = 547111040 : i64}> : () -> memref<128x128x!ttcore.tile<32x32, f32>, #dram>
    "ttmetal.enqueue_write_buffer"(%arg0, %0) : (memref<4096x4096xf32>, memref<128x128x!ttcore.tile<32x32, f32>, #dram>) -> ()
    %1 = "ttmetal.create_buffer"() <{address = 97248 : i64}> : () -> memref<8x8x8x8x!ttcore.tile<32x32, f32>, #ttcore.shard<32768x4096>, #l1>
    %2 = "ttmetal.create_buffer"() <{address = 97248 : i64}> : () -> memref<8x8x8x8x!ttcore.tile<32x32, f32>, #ttcore.shard<32768x4096>, #l1>
    %3 = "ttmetal.create_buffer"() <{address = 97248 : i64}> : () -> memref<8x8x8x8x!ttcore.tile<32x32, f32>, #ttcore.shard<32768x4096>, #l1>
    "ttmetal.enqueue_program"(%0, %o, %1, %2, %3) <{cb_ports = array<i64: 0, 1, 2>, kernelConfigs = [#ttmetal.noc_config<@DataFlow0_kernel_0, #ttmetal.core_range<0x0, 8x8>, #ttmetal.kernel_args< ct_args = [<cb_port[1]>, <buffer_address[0]>]>, noc0>, #ttmetal.noc_config<@DataFlow1_kernel_0, #ttmetal.core_range<0x0, 8x8>, #ttmetal.kernel_args< ct_args = [<cb_port[0]>, <buffer_address[1]>]>, noc1>, #ttmetal.compute_config<@Compute_kernel_0, #ttmetal.core_range<0x0, 8x8>, #ttmetal.kernel_args< ct_args = [<cb_port[0]>, <cb_port[1]>, <cb_port[2]>]>, hifi4, false, false, false, [default]>], operandSegmentSizes = array<i32: 2, 3>}> : (memref<128x128x!ttcore.tile<32x32, f32>, #dram>, memref<128x128x!ttcore.tile<32x32, f32>, #dram>, memref<8x8x8x8x!ttcore.tile<32x32, f32>, #ttcore.shard<32768x4096>, #l1>, memref<8x8x8x8x!ttcore.tile<32x32, f32>, #ttcore.shard<32768x4096>, #l1>, memref<8x8x8x8x!ttcore.tile<32x32, f32>, #ttcore.shard<32768x4096>, #l1>) -> ()
    "ttmetal.enqueue_read_buffer"(%o, %arg0) : (memref<128x128x!ttcore.tile<32x32, f32>, #dram>, memref<4096x4096xf32>) -> ()
    "ttmetal.finish"() : () -> ()
    "ttmetal.deallocate_buffer"(%0) : (memref<128x128x!ttcore.tile<32x32, f32>, #dram>) -> ()
    return %arg0 : memref<4096x4096xf32>
  }
}
