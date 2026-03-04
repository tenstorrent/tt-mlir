// RUN: ttmlir-opt --ttcore-register-device="system-desc-path=%system_desc_path%" --convert-d2m-to-ttnn -o %t.mlir %s
// RUN: FileCheck %s --input-file=%t.mlir
//
// Test that a generic output backed by a tensor-backed memref retains
// kernel_cb_global_buffer_address_of_tensor even when that memref later feeds a
// d2m.stream_layout user.

#dram = #ttnn.buffer_type<dram>
#l1 = #ttnn.buffer_type<l1>

#core_range = #ttnn.core_range<(0,0), (0,0)>
#l1_memory_config = #ttnn.memory_config<#l1, <block_sharded>, #ttnn.shard_spec<<[#core_range]>, <32x32>, <row_major>>>
#dram_memory_config = #ttnn.memory_config<#dram, <interleaved>>

#dram_layout = #ttnn.ttnn_layout<
  (d0, d1) -> (d0, d1),
  <1x1>,
  memref<1x1x!ttcore.tile<32x32, f32>, #dram>, <interleaved>
  >

#l1_layout = #ttnn.ttnn_layout<
  (d0, d1) -> (d0, d1),
  <1x1, (d0, d1) -> (0, d0, d1)>,
  memref<1x1x!ttcore.tile<32x32, f32>, #l1>, <block_sharded>
  >

module {
  ttcore.device @default_device = <workerGrid = #ttcore.grid<8x8, (d0, d1) -> (0, d0, d1)>, l1Map = (d0, d1, d2)[s0] -> (0, d0, d1, d2 + s0), dramMap = (d0, d1, d2)[s0, s1, s2, s3, s4, s5, s6] -> (0, 0, (((d0 * s1) * (s2 * (s3 * s6)) + d1 * (s2 * (s3 * s6)) + d2) floordiv s4) mod 12, ((((d0 * s1) * (s2 * (s3 * s6)) + d1 * (s2 * (s3 * s6)) + d2) floordiv s4) floordiv 12) * s4 + ((d0 * s1) * (s2 * (s3 * s6)) + d1 * (s2 * (s3 * s6)) + d2) mod s4 + s5), meshShape = , chipIds = [0]>

  // CHECK-LABEL: func.func @test_streamed_output_alias
  func.func @test_streamed_output_alias(%arg0: tensor<32x32xf32, #dram_layout>) -> tensor<32x32xf32, #dram_layout> {
    %device = "ttnn.get_device"() <{mesh_offset = #ttnn<mesh_offset 0x0>, mesh_shape = #ttnn<mesh_shape 1x1>}> : () -> !ttnn.device
    // CHECK: %[[TIN:.*]] = "ttnn.to_memory_config"
    // CHECK: %[[TOUT:.*]] = "ttnn.empty"
    // CHECK: %[[TMID:.*]] = "ttnn.empty"
    %ttnn_input_l1 = "ttnn.to_memory_config"(%arg0) <{memory_config = #l1_memory_config}> : (tensor<32x32xf32, #dram_layout>) -> tensor<32x32xf32, #l1_layout>
    %ttnn_output_l1 = d2m.empty() : tensor<32x32xf32, #l1_layout>

    %metal_input_l1 = ttir.ttnn_metal_layout_cast %ttnn_input_l1 : tensor<32x32xf32, #l1_layout> -> memref<1x1x1x1x!ttcore.tile<32x32, f32>, #ttcore.shard<4096x4096, 1>, #ttcore.memory_space<l1>>
    %metal_output_l1 = ttir.ttnn_metal_layout_cast %ttnn_output_l1 : tensor<32x32xf32, #l1_layout> -> memref<1x1x1x1x!ttcore.tile<32x32, f32>, #ttcore.shard<4096x4096, 1>, #ttcore.memory_space<l1>>
    %metal_mid_l1 = memref.alloc() : memref<1x1x1x1x!ttcore.tile<32x32, f32>, #ttcore.shard<4096x4096, 1>, #ttcore.memory_space<l1>>
    %stream_storage = memref.alloc() : memref<1x1x1x1x!ttcore.tile<32x32, f32>, #ttcore.shard<4096x4096, 1>, #ttcore.memory_space<l1>>
    %stream_mid = "d2m.stream_layout"(%metal_mid_l1, %stream_storage) <{remapping = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>}>
          : (memref<1x1x1x1x!ttcore.tile<32x32, f32>, #ttcore.shard<4096x4096, 1>, #ttcore.memory_space<l1>>,
             memref<1x1x1x1x!ttcore.tile<32x32, f32>, #ttcore.shard<4096x4096, 1>, #ttcore.memory_space<l1>>)
          -> memref<1x1x1x1x!ttcore.tile<32x32, f32>, #ttcore.view<4>, #ttcore.memory_space<l1>>

    // CHECK: "ttnn.generic"(%[[TIN]], %[[TMID]])
    // CHECK-SAME: buffer = #ttnn.kernel_cb_global_buffer_address_of_tensor<0>>
    // CHECK-SAME: buffer = #ttnn.kernel_cb_global_buffer_address_of_tensor<1>>
    d2m.generic {block_factors = [1, 1], grid = #ttcore.grid<1x1>, indexing_maps = [], iterator_types = [], threads = [#d2m.thread<datamovement, noc = 0, @read_kernel>, #d2m.thread<datamovement, noc = 1, @write_kernel>, #d2m.thread<compute, noc = 0, @compute_kernel0>]}
        ins(%metal_input_l1 : memref<1x1x1x1x!ttcore.tile<32x32, f32>, #ttcore.shard<4096x4096, 1>, #ttcore.memory_space<l1>>)
        outs(%metal_mid_l1 : memref<1x1x1x1x!ttcore.tile<32x32, f32>, #ttcore.shard<4096x4096, 1>, #ttcore.memory_space<l1>>)

    // CHECK: "ttnn.generic"(%[[TMID]], %[[TOUT]])
    // CHECK-SAME: buffer = #ttnn.kernel_cb_global_buffer_address_of_tensor<1>>
    d2m.generic {block_factors = [1, 1], grid = #ttcore.grid<1x1>, indexing_maps = [], iterator_types = [], threads = [#d2m.thread<datamovement, noc = 0, @read_kernel>, #d2m.thread<datamovement, noc = 1, @write_kernel>, #d2m.thread<compute, noc = 0, @compute_kernel0>]}
        ins(%stream_mid : memref<1x1x1x1x!ttcore.tile<32x32, f32>, #ttcore.view<4>, #ttcore.memory_space<l1>>)
        outs(%metal_output_l1 : memref<1x1x1x1x!ttcore.tile<32x32, f32>, #ttcore.shard<4096x4096, 1>, #ttcore.memory_space<l1>>)

    %output_l1 = ttir.ttnn_metal_layout_cast %metal_output_l1 : memref<1x1x1x1x!ttcore.tile<32x32, f32>, #ttcore.shard<4096x4096, 1>, #ttcore.memory_space<l1>> -> tensor<32x32xf32, #l1_layout>
    %output_dram = "ttnn.to_memory_config"(%output_l1) <{memory_config = #dram_memory_config}> : (tensor<32x32xf32, #l1_layout>) -> tensor<32x32xf32, #dram_layout>
    return %output_dram : tensor<32x32xf32, #dram_layout>
  }

  func.func private @read_kernel() attributes {
    ttkernel.arg_spec = #ttkernel.arg_spec<
      rt_args = [<arg_type = buffer_address, operand_index = 0>]
      ct_args = [
        <arg_type = cb_port, operand_index = 0>,
        <arg_type = semaphore, operand_index = 0>,
        <arg_type = semaphore, operand_index = 1>
      ]
    >,
    ttkernel.thread = #ttkernel.thread<noc>
  } {
    return
  }

  func.func private @compute_kernel0() attributes {
    ttkernel.arg_spec = #ttkernel.arg_spec<
      ct_args = [
        <arg_type = cb_port, operand_index = 0>,
        <arg_type = cb_port, operand_index = 1>
      ]
    >,
    ttkernel.thread = #ttkernel.thread<compute>
  } {
    return
  }

  func.func private @write_kernel() attributes {
    ttkernel.arg_spec = #ttkernel.arg_spec<
      rt_args = [<arg_type = buffer_address, operand_index = 1>]
      ct_args = [
        <arg_type = cb_port, operand_index = 1>,
        <arg_type = semaphore, operand_index = 2>,
        <arg_type = semaphore, operand_index = 3>
      ]
    >,
    ttkernel.thread = #ttkernel.thread<noc>
  } {
    return
  }
}
