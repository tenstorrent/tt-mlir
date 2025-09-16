// RUN: ttmlir-opt -ttir-lower-ttnn-layouts -o %t.mlir %s
// RUN: FileCheck %s --input-file=%t.mlir

// Single-case test: block-sharded L1 tiled TTNN layout is lowered to an
// explicit ttir.ttnn_to_metal_layout_cast with a ttcore.metal_layout encoding.

#l1 = #ttnn.buffer_type<l1>

// CHECK: #layout = #ttcore.metal_layout<logical_shape = 32x32, dim_alignments = 32x32, collapsed_intervals =
#ttnn_layout = #ttnn.ttnn_layout<
  (d0, d1) -> (d0, d1),
  <1x1, (d0, d1) -> (0, d0, d1)>,
  memref<1x1x!ttcore.tile<32x32, f32>, #l1>, <block_sharded>
  >
// Use single core
#core_range = #ttnn.core_range<(0,0), (0,0)>
#core_ranges = #ttnn.core_range_set<[#core_range]>

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


module {
// CHECK-LABEL: func.func @test_lower_block_sharded_l1
func.func @test_lower_block_sharded_l1(
  %arg0: tensor<32x32xf32, #ttnn_layout>
) {
  // Expect the pass to insert a single cast op converting the TTNN layout to a TTCore metal layout.
  // (Alias asserted above; ensure the cast uses it.)
  // CHECK: %[[CAST:.*]] = ttir.ttnn_to_metal_layout_cast %arg0
  // CHECK-SAME: : tensor<32x32xf32, #ttnn_layout> -> tensor<32x32xf32, #layout>

  // And the ttnn.generic should consume that cast value for both operands with metal_layout-typed tensors.
  // CHECK: "ttnn.generic"(%[[CAST]], %[[CAST]]) <{program = {{.*}}}>
  // CHECK-SAME: : (tensor<32x32xf32, #layout>, tensor<32x32xf32, #layout>) -> ()

  "ttnn.generic"(%arg0, %arg0) <{program = #program}> : (tensor<32x32xf32, #ttnn_layout>, tensor<32x32xf32, #ttnn_layout>) -> ()

    return
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
