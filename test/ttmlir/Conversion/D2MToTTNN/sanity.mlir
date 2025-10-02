// RUN: ttmlir-opt --ttcore-register-device="system-desc-path=%system_desc_path%" --ttir-to-ttmetal-be-pipeline="ttnn-mode=true" -o %t.mlir %s
// RUN: FileCheck %s --input-file=%t.mlir

#dram = #ttnn.buffer_type<dram>
#l1 = #ttnn.buffer_type<l1>

#core_range = #ttnn.core_range<(0,0), (0,0)>
#core_ranges = #ttnn.core_range_set<[#core_range]>

#dram_memory_config = #ttnn.memory_config<#dram, <interleaved>>
#l1_memory_config = #ttnn.memory_config<#l1, <block_sharded>, #ttnn.shard_spec<<[#core_range]>, <32x32>, <row_major>>>

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

module {
  func.func @abs(%arg0: tensor<32x32xf32, #dram_layout>) -> tensor<32x32xf32, #dram_layout> {
    %device = "ttnn.get_device"() <{mesh_offset = #ttnn<mesh_offset 0x0>, mesh_shape = #ttnn<mesh_shape 1x1>}> : () -> !ttnn.device

    // CHECK: %[[T1:.*]] = "ttnn.to_memory_config"
    // CHECK: %[[T2:.*]] = "ttnn.empty"
    %ttnn_input_l1 = "ttnn.to_memory_config"(%arg0) <{memory_config = #l1_memory_config}> : (tensor<32x32xf32, #dram_layout>) -> tensor<32x32xf32, #l1_layout>
    %ttnn_output_l1 = d2m.empty() : tensor<32x32xf32, #l1_layout>

    // CHECK-NOT: ttir.ttnn_metal_layout_cast
    // CHECK-NOT: ttir.ttnn_metal_layout_cast
    %metal_input_l1 = ttir.ttnn_metal_layout_cast %ttnn_input_l1 : tensor<32x32xf32, #l1_layout> -> memref<1x1x1x1x!ttcore.tile<32x32, f32>, #ttcore.shard<4096x4096>, #ttcore.memory_space<l1>>
    %metal_output_l1 = ttir.ttnn_metal_layout_cast %ttnn_output_l1 : tensor<32x32xf32, #l1_layout> -> memref<1x1x1x1x!ttcore.tile<32x32, f32>, #ttcore.shard<4096x4096>, #ttcore.memory_space<l1>>

    // CHECK-NOT: memref.alloc()
    // CHECK-NOT: d2m.stream_layout
    %storage_in = memref.alloc() : memref<1x1x1x1x!ttcore.tile<32x32, f32>, #ttcore.shard<4096x4096>, #ttcore.memory_space<l1>>
    %stream_input  = "d2m.stream_layout" (%metal_input_l1, %storage_in)
          : (memref<1x1x1x1x!ttcore.tile<32x32, f32>, #ttcore.shard<4096x4096>, #ttcore.memory_space<l1>>,
             memref<1x1x1x1x!ttcore.tile<32x32, f32>, #ttcore.shard<4096x4096>, #ttcore.memory_space<l1>>)
          -> memref<1x1x1x1x!ttcore.tile<32x32, f32>, #ttcore.view<map(4)>, #ttcore.memory_space<l1>>

    // CHECK-NOT: memref.alloc()
    // CHECK-NOT: d2m.stream_layout
    %storage_out = memref.alloc() : memref<1x1x1x1x!ttcore.tile<32x32, f32>, #ttcore.shard<4096x4096>, #ttcore.memory_space<l1>>
    %stream_output  = "d2m.stream_layout" (%metal_output_l1, %storage_out)
          : (memref<1x1x1x1x!ttcore.tile<32x32, f32>, #ttcore.shard<4096x4096>, #ttcore.memory_space<l1>>,
             memref<1x1x1x1x!ttcore.tile<32x32, f32>, #ttcore.shard<4096x4096>, #ttcore.memory_space<l1>>)
          -> memref<1x1x1x1x!ttcore.tile<32x32, f32>, #ttcore.view<map(4)>, #ttcore.memory_space<l1>>

    // CHECK: "ttnn.generic"(%[[T1]], %[[T2]])
    // CHECK-SAME: #ttnn.read_kernel<symbol_ref = @read_kernel
    // CHECK-SAME: ct_args = [#ttnn.kernel_arg_cb_buffer_index<0>]
    // CHECK-SAME: common_rt_args = [#ttnn.kernel_arg_address_of_tensor<0>]
    // CHECK-SAME: #ttnn.write_kernel<symbol_ref = @write_kernel
    // CHECK-SAME: ct_args = [#ttnn.kernel_arg_cb_buffer_index<1>]
    // CHECK-SAME: common_rt_args = [#ttnn.kernel_arg_address_of_tensor<1>]
    // CHECK-SAME: #ttnn.compute_kernel<symbol_ref = @compute_kernel0
    // CHECK-SAME: ct_args = [#ttnn.kernel_arg_cb_buffer_index<0>, #ttnn.kernel_arg_cb_buffer_index<1>]
    // CHECK-SAME: common_rt_args = []
    // CHECK-SAME: page_size = 4096
    d2m.generic {block_factors = [1, 1], grid = #ttcore.grid<1x1>, indexing_maps = [], iterator_types = [], threads = [#d2m.thread<datamovement, @read_kernel>, #d2m.thread<datamovement, @write_kernel>, #d2m.thread<compute, @compute_kernel0>]}
        ins(%stream_input : memref<1x1x1x1x!ttcore.tile<32x32, f32>, #ttcore.view<map(4)>, #ttcore.memory_space<l1>>)
        outs(%stream_output : memref<1x1x1x1x!ttcore.tile<32x32, f32>, #ttcore.view<map(4)>, #ttcore.memory_space<l1>>)

    // CHECK: "ttnn.generic"(%[[T1]], %[[T2]])
    // CHECK-SAME: buffer = #ttnn.kernel_cb_global_buffer_address_of_tensor<0>>
    // CHECK-SAME: buffer = #ttnn.kernel_cb_global_buffer_address_of_tensor<1>>
    d2m.generic {block_factors = [1, 1], grid = #ttcore.grid<1x1>, indexing_maps = [], iterator_types = [], threads = [#d2m.thread<datamovement, @read_kernel>, #d2m.thread<datamovement, @write_kernel>, #d2m.thread<compute, @compute_kernel0>]}
        ins(%metal_input_l1 : memref<1x1x1x1x!ttcore.tile<32x32, f32>, #ttcore.shard<4096x4096>, #ttcore.memory_space<l1>>)
        outs(%metal_output_l1 : memref<1x1x1x1x!ttcore.tile<32x32, f32>, #ttcore.shard<4096x4096>, #ttcore.memory_space<l1>>)

    // CHECK-NOT: ttir.ttnn_metal_layout_cast
    %output_l1 = ttir.ttnn_metal_layout_cast %metal_output_l1 : memref<1x1x1x1x!ttcore.tile<32x32, f32>, #ttcore.shard<4096x4096>, #ttcore.memory_space<l1>> -> tensor<32x32xf32, #l1_layout>

    %output_dram = "ttnn.to_memory_config"(%output_l1) <{memory_config = #dram_memory_config}> : (tensor<32x32xf32, #l1_layout>) -> tensor<32x32xf32, #dram_layout>
    return %output_dram : tensor<32x32xf32, #dram_layout>
  }
  func.func private @read_kernel() attributes {ttkernel.arg_spec = #ttkernel.arg_spec< rt_args = [<arg_type = buffer_address, operand_index = 0>] ct_args = [<arg_type = cb_port, operand_index = 0>]>, ttkernel.thread = #ttkernel.thread<noc>} {
    return
  }
  func.func private @compute_kernel0() attributes {ttkernel.arg_spec = #ttkernel.arg_spec< ct_args = [<arg_type = cb_port, operand_index = 0>, <arg_type = cb_port, operand_index = 1>]>, ttkernel.thread = #ttkernel.thread<compute>} {
    return
  }
  func.func private @write_kernel() attributes {ttkernel.arg_spec = #ttkernel.arg_spec< rt_args = [<arg_type = buffer_address, operand_index = 1>] ct_args = [<arg_type = cb_port, operand_index = 1>]>, ttkernel.thread = #ttkernel.thread<noc>} {
    return
  }
}
