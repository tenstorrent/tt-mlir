// RUN: ttmlir-opt --ttcore-register-device="system-desc-path=%system_desc_path%" -o %t.mlir %s
// RUN: FileCheck %s --input-file=%t.mlir
// RUN: ttmlir-translate --ttnn-to-flatbuffer -o %t.ttnn %t.mlir

#dram = #ttnn.buffer_type<dram>
#l1 = #ttnn.buffer_type<l1>

#core = #ttnn.core_coord<0, 0>
#core_range = #ttnn.core_range<(0,0), (0,0)>
#core_ranges = #ttnn.core_range_set<[#core_range]>

#in_cb_format = #ttnn.kernel_cb_format<buffer_index = 0, dtype = f32, page_size = 4096>
#in_cb = #ttnn.kernel_cb<total_size = 4096, core_ranges = #core_ranges, formats = [#in_cb_format]>
#out_cb_format = #ttnn.kernel_cb_format<buffer_index = 1, dtype = f32, page_size = 4096>
#out_cb = #ttnn.kernel_cb<total_size = 4096, core_ranges = #core_ranges, formats = [#out_cb_format]>

#in_cb_arg = #ttnn.kernel_arg_cb_buffer_index<0>
#in_addr_arg = #ttnn.kernel_arg_address_of_tensor<0>
#out_cb_arg = #ttnn.kernel_arg_cb_buffer_index<1>
#out_addr_arg = #ttnn.kernel_arg_address_of_tensor<1>

// The three scalars are `additional_args` 0, 1 and 2 (CBs/semaphores do not
// appear in the ttnn.generic's additional args, only the scalars do).
#scalar_i32_arg = #ttnn.kernel_arg_scalar<0>
#scalar_f32_arg = #ttnn.kernel_arg_scalar<1>
#scalar_i16_arg = #ttnn.kernel_arg_scalar<2>

#read_kernel = #ttnn.read_kernel<
  symbol_ref = @read_kernel,
  core_ranges = #core_ranges,
  ct_args = [#in_cb_arg],
  common_rt_args = [#in_addr_arg],
  rt_args = []>

#compute_kernel = #ttnn.compute_kernel<
  symbol_ref = @compute_kernel,
  core_ranges = #core_ranges,
  math_fidelity = hifi4,
  fp32_dest_acc_en = false,
  dst_full_sync_en = false,
  unpack_to_dest_modes = [default],
  bfp8_pack_precise = false,
  math_approx_mode = false,
  ct_args = [#in_cb_arg, #out_cb_arg],
  common_rt_args = [],
  rt_args = []>

// The scalars are consumed as runtime args by the write kernel.
#write_kernel = #ttnn.write_kernel<
  symbol_ref = @write_kernel,
  core_ranges = #core_ranges,
  ct_args = [#out_cb_arg],
  common_rt_args = [],
  rt_args = [#ttnn.core_runtime_args<core_coord = #core, args = [#out_addr_arg, #scalar_i32_arg, #scalar_f32_arg, #scalar_i16_arg]>]>

#program = #ttnn.program<
  kernels = [#read_kernel, #compute_kernel, #write_kernel],
  cbs = [#in_cb, #out_cb],
  semaphores = []>

#dram_layout = #ttnn.ttnn_layout<
  (d0, d1) -> (d0, d1),
  <1x1>,
  memref<1x1x!ttcore.tile<32x32, f32>, #dram>, <interleaved>>
#l1_layout = #ttnn.ttnn_layout<
  (d0, d1) -> (d0, d1),
  <1x1>,
  memref<1x1x!ttcore.tile<32x32, f32>, #l1>, <block_sharded>,
  core_ranges = #ttnn.core_range_set<[#ttnn.core_range<(0,0), (0,0)>]>>

// CHECK: module attributes {ttcore.system_desc = #system_desc}
// CHECK: ttcore.device @default_device
module {
  // The scalar function arguments survive device registration untouched.
  // CHECK-LABEL: func.func @test_scalar_args
  // CHECK-SAME: %arg1: i32
  // CHECK-SAME: %arg2: f32
  // CHECK-SAME: %arg3: i16
  func.func @test_scalar_args(%arg0: tensor<32x32xf32, #dram_layout>,
                              %scalar_i32: i32,
                              %scalar_f32: f32,
                              %scalar_i16: i16) -> tensor<32x32xf32, #dram_layout> attributes {tt.function_type = "forward_device"} {
    %0 = "ttnn.get_device"() <{mesh_offset = #ttnn<mesh_offset 0x0>, mesh_shape = #ttnn<mesh_shape 1x1>}> : () -> !ttnn.device
    %1 = "ttnn.to_memory_config"(%arg0) : (tensor<32x32xf32, #dram_layout>) -> tensor<32x32xf32, #l1_layout>
    %2 = "ttnn.empty"(%0) <{layout = #ttnn.layout<tile>, shape = #ttnn.shape<32x32>}> : (!ttnn.device) -> tensor<32x32xf32, #l1_layout>

    // The three scalars are passed as the generic op's additional args.
    // CHECK: "ttnn.generic"(%{{.*}}, %{{.*}}, %arg1, %arg2, %arg3)
    // CHECK-SAME: operandSegmentSizes = array<i32: 2, 3>
    // CHECK-SAME: (tensor<32x32xf32, {{.*}}>, tensor<32x32xf32, {{.*}}>, i32, f32, i16) -> ()
    "ttnn.generic"(%1, %2, %scalar_i32, %scalar_f32, %scalar_i16) <{program = #program, operandSegmentSizes = array<i32: 2, 3>}> : (tensor<32x32xf32, #l1_layout>, tensor<32x32xf32, #l1_layout>, i32, f32, i16) -> ()

    %3 = "ttnn.to_memory_config"(%2) : (tensor<32x32xf32, #l1_layout>) -> tensor<32x32xf32, #dram_layout>
    return %3 : tensor<32x32xf32, #dram_layout>
  }
  func.func private @read_kernel() attributes {tt.function_type = "kernel", ttkernel.arg_spec = #ttkernel.arg_spec< rt_args = [<arg_type = buffer_address, operand_index = 0>] ct_args = [<arg_type = cb_port, operand_index = 0>]>, ttkernel.thread = #ttkernel.thread<noc>} {
    return
  }
  func.func private @compute_kernel() attributes {tt.function_type = "kernel", ttkernel.arg_spec = #ttkernel.arg_spec< ct_args = [<arg_type = cb_port, operand_index = 0>, <arg_type = cb_port, operand_index = 1>]>, ttkernel.thread = #ttkernel.thread<compute>} {
    return
  }
  // The write kernel reads the scalar runtime args (operand indices 2, 3, 4:
  // after the 2 io tensors).
  func.func private @write_kernel() attributes {tt.function_type = "kernel", ttkernel.arg_spec = #ttkernel.arg_spec< rt_args = [<arg_type = buffer_address, operand_index = 1>, <arg_type = scalar, operand_index = 2>, <arg_type = scalar, operand_index = 3>, <arg_type = scalar, operand_index = 4>] ct_args = [<arg_type = cb_port, operand_index = 0>]>, ttkernel.thread = #ttkernel.thread<noc>} {
    return
  }
}
