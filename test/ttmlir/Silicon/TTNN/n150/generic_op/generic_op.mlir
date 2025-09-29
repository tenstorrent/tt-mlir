// RUN: ttmlir-opt --ttcore-register-device="system-desc-path=%system_desc_path%" -o %t.mlir %s
// RUN: FileCheck %s --input-file=%t.mlir
// RUN: ttmlir-translate --ttnn-to-flatbuffer -o %t.ttnn %t.mlir

#dram = #ttnn.buffer_type<dram>
#l1 = #ttnn.buffer_type<l1>

// Use single core
#core_range = #ttnn.core_range<(0,0), (0,0)>
#core_ranges = #ttnn.core_range_set<[#core_range]>

#dram_memory_config = #ttnn.memory_config<#dram, <interleaved>>
#l1_memory_config = #ttnn.memory_config<#l1, <block_sharded>, #ttnn.shard_spec<<[#core_range]>, <32x32>, <row_major>>>

// In- and out- CB descriptors
#in_cb_format = #ttnn.kernel_cb_format<
  buffer_index = 0,
  dtype = f32,
  page_size = 4096>

#in_cb = #ttnn.kernel_cb<
  total_size = 4096,
  core_ranges = #core_ranges,
  formats = [#in_cb_format]>

#out_cb_format = #ttnn.kernel_cb_format<
  buffer_index = 1,
  dtype = f32,
  page_size = 4096>

#out_cb = #ttnn.kernel_cb<
  total_size = 4096,
  core_ranges = #core_ranges,
  formats = [#out_cb_format]>

// CB descriptor reference by position in #cbs
#in_cb_arg = #ttnn.kernel_arg_cb_buffer_index<0>
// Tensor address by position in IO tensors
#in_addr_arg = #ttnn.kernel_arg_address_of_tensor<0>

#out_cb_arg = #ttnn.kernel_arg_cb_buffer_index<1>
#out_addr_arg = #ttnn.kernel_arg_address_of_tensor<1>

#read_kernel = #ttnn.read_kernel<
  symbol_ref = @read_kernel,
  core_ranges = #core_ranges,
  ct_args = [#in_cb_arg],
  // Pass tensor address and CB as runtime arguments
  common_rt_args = [#in_addr_arg]>

#compute_kernel = #ttnn.compute_kernel<
  symbol_ref = @compute_kernel,
  core_ranges = #core_ranges,
  math_fidelity = hifi4,
  fp32_dest_acc_en = false,
  dst_full_sync_en = false,
  unpack_to_dest_modes = [default],
  bfp8_pack_precise = false,
  math_approx_mode = false,
  ct_args = [
      #in_cb_arg,
      #out_cb_arg
  ],
  common_rt_args = []>

#write_kernel = #ttnn.write_kernel<
  symbol_ref = @write_kernel,
  core_ranges = #core_ranges,
  ct_args = [#out_cb_arg],
  common_rt_args = [#out_addr_arg]>

#program = #ttnn.program<
  kernels = [#read_kernel, #compute_kernel, #write_kernel],
  cbs = [#in_cb, #out_cb],
  semaphores = []>

// Layout for single tile in DRAM
#dram_layout = #ttnn.ttnn_layout<
  (d0, d1) -> (d0, d1),
  <1x1>,
  memref<1x1x!ttcore.tile<32x32, f32>, #dram>, <interleaved>
  >

// Layout for single tile in L1
#l1_layout = #ttnn.ttnn_layout<
  (d0, d1) -> (d0, d1),
  <1x1, (d0, d1) -> (0, d0, d1)>,
  memref<1x1x!ttcore.tile<32x32, f32>, #l1>, <block_sharded>
  >

// CHECK: module attributes {ttcore.system_desc = #system_desc}
// CHECK: ttcore.device @default_device
module {
  func.func @test(%arg0: tensor<32x32xf32, #dram_layout>) -> tensor<32x32xf32, #dram_layout> {
    %0 = "ttnn.get_device"() <{mesh_offset = #ttnn<mesh_offset 0x0>, mesh_shape = #ttnn<mesh_shape 1x1>}> : () -> !ttnn.device

    // Move single tile from DRAM to L1
    %1 = "ttnn.to_memory_config"(%arg0) <{memory_config = #l1_memory_config}> : (tensor<32x32xf32, #dram_layout>) -> tensor<32x32xf32, #l1_layout>

    // Allocate single tile in L1 for result
    %2 = "ttnn.empty"(%0) <{dtype = #ttcore.supportedDataTypes<f32>, layout = #ttnn.layout<tile>, memory_config = #l1_memory_config, shape = #ttnn.shape<32x32>}> : (!ttnn.device) -> tensor<32x32xf32, #l1_layout>

    "ttnn.generic"(%1, %2) <{program = #program}> : (tensor<32x32xf32, #l1_layout>, tensor<32x32xf32, #l1_layout>) -> ()

    // Move single tile from L1 to DRAM
    %3 = "ttnn.to_memory_config"(%2) <{memory_config = #dram_memory_config}> : (tensor<32x32xf32, #l1_layout>) -> tensor<32x32xf32, #dram_layout>

    return %3 : tensor<32x32xf32, #dram_layout>
  }
  func.func private @read_kernel() attributes {ttkernel.arg_spec = #ttkernel.arg_spec< rt_args = [<arg_type = buffer_address, operand_index = 0>] ct_args = [<arg_type = cb_port, operand_index = 0>]>, ttkernel.thread = #ttkernel.thread<noc>} {
    %0 = "emitc.constant"() <{value = 1 : i32}> : () -> i32
    %1 = "emitc.constant"() <{value = 4096 : i32}> : () -> i32

    %zero = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %2 = emitc.call_opaque "get_common_arg_val"(%zero) {template_args = [#emitc.opaque<"uint32_t">]} : (i32) -> i32
    %3 = emitc.literal "get_compile_time_arg_val(0)" : !emitc.opaque<"::tt::CB">
    %4 = emitc.literal "my_x[noc_index]" : !emitc.size_t
    %5 = emitc.literal "my_y[noc_index]" : !emitc.size_t

    // Move single tile from address in L1 to CB
    emitc.call_opaque "cb_reserve_back"(%3, %0) : (!emitc.opaque<"::tt::CB">, i32) -> ()
    %6 = emitc.call_opaque "get_write_ptr"(%3) : (!emitc.opaque<"::tt::CB">) -> i32
    %7 = emitc.call_opaque "get_noc_addr"(%4, %5, %2) : (!emitc.size_t, !emitc.size_t, i32) -> i64
    emitc.call_opaque "noc_async_read"(%7, %6, %1) : (i64, i32, i32) -> ()
    emitc.call_opaque "noc_async_read_barrier"() : () -> ()
    emitc.call_opaque "cb_push_back"(%3, %0) : (!emitc.opaque<"::tt::CB">, i32) -> ()
    return
  }
  func.func private @compute_kernel() attributes {ttkernel.arg_spec = #ttkernel.arg_spec< ct_args = [<arg_type = cb_port, operand_index = 0>, <arg_type = cb_port, operand_index = 1>]>, ttkernel.thread = #ttkernel.thread<compute>} {
    %0 = "emitc.constant"() <{value = 1 : i32}> : () -> i32
    %1 = "emitc.constant"() <{value = 0 : index}> : () -> !emitc.size_t

    %2 = emitc.literal "get_compile_time_arg_val(0)" : !emitc.opaque<"::tt::CB">
    %3 = emitc.literal "get_compile_time_arg_val(1)" : !emitc.opaque<"::tt::CB">

    // Move single tile from CB to register
    emitc.call_opaque "cb_reserve_back"(%3, %0) : (!emitc.opaque<"::tt::CB">, i32) -> ()
    emitc.call_opaque "cb_wait_front"(%2, %0) : (!emitc.opaque<"::tt::CB">, i32) -> ()
    emitc.call_opaque "init_sfpu"(%2, %3) : (!emitc.opaque<"::tt::CB">, !emitc.opaque<"::tt::CB">) -> ()
    emitc.call_opaque "tile_regs_acquire"() : () -> ()
    emitc.call_opaque "copy_tile_init"(%2) : (!emitc.opaque<"::tt::CB">) -> ()
    emitc.call_opaque "copy_tile"(%2, %1, %1) : (!emitc.opaque<"::tt::CB">, !emitc.size_t, !emitc.size_t) -> ()

    // Compute abs
    emitc.call_opaque "abs_tile_init"() : () -> ()
    emitc.call_opaque "abs_tile"(%1) : (!emitc.size_t) -> ()

    // Move single tile from register to CB
    emitc.call_opaque "tile_regs_commit"() : () -> ()
    emitc.call_opaque "tile_regs_wait"() : () -> ()
    emitc.call_opaque "pack_tile"(%1, %3, %1) {template_args = [true]} : (!emitc.size_t, !emitc.opaque<"::tt::CB">, !emitc.size_t) -> ()
    emitc.call_opaque "tile_regs_release"() : () -> ()
    emitc.call_opaque "cb_push_back"(%3, %0) : (!emitc.opaque<"::tt::CB">, i32) -> ()
    emitc.call_opaque "cb_pop_front"(%2, %0) : (!emitc.opaque<"::tt::CB">, i32) -> ()
    return
  }
  func.func private @write_kernel() attributes {ttkernel.arg_spec = #ttkernel.arg_spec< rt_args = [<arg_type = buffer_address, operand_index = 0>] ct_args = [<arg_type = cb_port, operand_index = 0>]>, ttkernel.thread = #ttkernel.thread<noc>} {
    %0 = "emitc.constant"() <{value = 1 : i32}> : () -> i32
    %1 = "emitc.constant"() <{value = 4096 : i32}> : () -> i32

    %zero = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %2 = emitc.call_opaque "get_common_arg_val"(%zero) {template_args = [#emitc.opaque<"uint32_t">]} : (i32) -> i32
    %3 = emitc.literal "get_compile_time_arg_val(0)" : !emitc.opaque<"::tt::CB">
    %4 = emitc.literal "my_x[noc_index]" : !emitc.size_t
    %5 = emitc.literal "my_y[noc_index]" : !emitc.size_t

    // Move single tile from CB to address in L1
    emitc.call_opaque "cb_wait_front"(%3, %0) : (!emitc.opaque<"::tt::CB">, i32) -> ()
    %6 = emitc.call_opaque "get_read_ptr"(%3) : (!emitc.opaque<"::tt::CB">) -> i32
    %7 = emitc.call_opaque "get_noc_addr"(%4, %5, %2) : (!emitc.size_t, !emitc.size_t, i32) -> i64
    emitc.call_opaque "noc_async_write"(%6, %7, %1) : (i32, i64, i32) -> ()
    emitc.call_opaque "noc_async_write_barrier"() : () -> ()
    emitc.call_opaque "cb_pop_front"(%3, %0) : (!emitc.opaque<"::tt::CB">, i32) -> ()
    return
  }
}
