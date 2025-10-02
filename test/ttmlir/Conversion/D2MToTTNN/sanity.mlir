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
    %c1_i32 = arith.constant 1 : i32
    %c4096_i32 = arith.constant 4096 : i32

    %c0_i32 = arith.constant 0 : i32
    %0 = ttkernel.get_common_arg_val(%c0_i32) : (i32) -> i32
    %1 = ttkernel.get_compile_time_arg_val(0) : () -> !ttkernel.cb<memref<32x32xf32, #ttcore.memory_space<l1>>>
    %2 = ttkernel.my_x() : () -> index
    %3 = ttkernel.my_y() : () -> index

    // Move single tile from address in L1 to CB
    ttkernel.cb_reserve_back(%1, %c1_i32) : (!ttkernel.cb<memref<32x32xf32, #ttcore.memory_space<l1>>>, i32) -> ()
    %4 = ttkernel.get_write_ptr(%1) : (!ttkernel.cb<memref<32x32xf32, #ttcore.memory_space<l1>>>) -> i32
    %5 = ttkernel.get_noc_addr(%2, %3, %0) : (index, index, i32) -> !ttkernel.noc_addr
    ttkernel.noc_async_read(%5, %4, %c4096_i32) : (!ttkernel.noc_addr, i32, i32) -> ()
    ttkernel.noc_async_read_barrier() : () -> ()
    ttkernel.cb_push_back(%1, %c1_i32) : (!ttkernel.cb<memref<32x32xf32, #ttcore.memory_space<l1>>>, i32) -> ()
    return
  }
  func.func private @compute_kernel0() attributes {ttkernel.arg_spec = #ttkernel.arg_spec< ct_args = [<arg_type = cb_port, operand_index = 0>, <arg_type = cb_port, operand_index = 1>]>, ttkernel.thread = #ttkernel.thread<compute>} {
    %c1_i32 = arith.constant 1 : i32
    %c0 = arith.constant 0 : index
    %0 = ttkernel.get_compile_time_arg_val(0) : () -> !ttkernel.cb<memref<1x1x!ttcore.tile<32x32, f32>, #ttcore.memory_space<l1>>>
    %1 = ttkernel.get_compile_time_arg_val(1) : () -> !ttkernel.cb<memref<1x1x!ttcore.tile<32x32, f32>, #ttcore.memory_space<l1>>>
    ttkernel.cb_reserve_back(%1, %c1_i32) : (!ttkernel.cb<memref<1x1x!ttcore.tile<32x32, f32>, #ttcore.memory_space<l1>>>, i32) -> ()
    ttkernel.cb_wait_front(%0, %c1_i32) : (!ttkernel.cb<memref<1x1x!ttcore.tile<32x32, f32>, #ttcore.memory_space<l1>>>, i32) -> ()
    %2 = ttkernel.cb_reinterpret_shape(%0) : (!ttkernel.cb<memref<1x1x!ttcore.tile<32x32, f32>, #ttcore.memory_space<l1>>>) -> !ttkernel.cb<memref<1x!ttcore.tile<32x32, f32>, #ttcore.memory_space<l1>>>
    %3 = ttkernel.cb_reinterpret_shape(%1) : (!ttkernel.cb<memref<1x1x!ttcore.tile<32x32, f32>, #ttcore.memory_space<l1>>>) -> !ttkernel.cb<memref<1x!ttcore.tile<32x32, f32>, #ttcore.memory_space<l1>>>
    ttkernel.init_sfpu(%2, %3) : (!ttkernel.cb<memref<1x!ttcore.tile<32x32, f32>, #ttcore.memory_space<l1>>>, !ttkernel.cb<memref<1x!ttcore.tile<32x32, f32>, #ttcore.memory_space<l1>>>) -> ()
    ttkernel.tile_regs_acquire() : () -> ()
    ttkernel.copy_tile_init(%2) : (!ttkernel.cb<memref<1x!ttcore.tile<32x32, f32>, #ttcore.memory_space<l1>>>) -> ()
    ttkernel.copy_tile(%2, %c0, %c0) : (!ttkernel.cb<memref<1x!ttcore.tile<32x32, f32>, #ttcore.memory_space<l1>>>, index, index) -> ()
    ttkernel.abs_tile_init() : () -> ()
    ttkernel.abs_tile(%c0) : (index) -> ()
    ttkernel.tile_regs_commit() : () -> ()
    ttkernel.tile_regs_wait() : () -> ()
    ttkernel.pack_tile(%c0, %3, %c0, true) : (index, !ttkernel.cb<memref<1x!ttcore.tile<32x32, f32>, #ttcore.memory_space<l1>>>, index) -> ()
    ttkernel.tile_regs_release() : () -> ()
    ttkernel.cb_push_back(%1, %c1_i32) : (!ttkernel.cb<memref<1x1x!ttcore.tile<32x32, f32>, #ttcore.memory_space<l1>>>, i32) -> ()
    ttkernel.cb_pop_front(%0, %c1_i32) : (!ttkernel.cb<memref<1x1x!ttcore.tile<32x32, f32>, #ttcore.memory_space<l1>>>, i32) -> ()
    return
  }
  func.func private @write_kernel() attributes {ttkernel.arg_spec = #ttkernel.arg_spec< rt_args = [<arg_type = buffer_address, operand_index = 1>] ct_args = [<arg_type = cb_port, operand_index = 1>]>, ttkernel.thread = #ttkernel.thread<noc>} {
    %c1_i32 = arith.constant 1 : i32
    %c4096_i32 = arith.constant 4096 : i32

    %c0_i32 = arith.constant 0 : i32
    %0 = ttkernel.get_common_arg_val(%c0_i32) : (i32) -> i32
    %1 = ttkernel.get_compile_time_arg_val(0) : () -> !ttkernel.cb<memref<32x32xf32, #ttcore.memory_space<l1>>>
    %2 = ttkernel.my_x() : () -> index
    %3 = ttkernel.my_y() : () -> index

    // Move single tile from CB to address in L1
    ttkernel.cb_wait_front(%1, %c1_i32) : (!ttkernel.cb<memref<32x32xf32, #ttcore.memory_space<l1>>>, i32) -> ()
    %4 = ttkernel.get_read_ptr(%1) : (!ttkernel.cb<memref<32x32xf32, #ttcore.memory_space<l1>>>) -> i32
    %5 = ttkernel.get_noc_addr(%2, %3, %0) : (index, index, i32) -> !ttkernel.noc_addr
    ttkernel.noc_async_write(%4, %5, %c4096_i32) : (i32, !ttkernel.noc_addr, i32) -> ()
    ttkernel.noc_async_write_barrier() : () -> ()
    ttkernel.cb_pop_front(%1, %c1_i32) : (!ttkernel.cb<memref<32x32xf32, #ttcore.memory_space<l1>>>, i32) -> ()
    return
  }
}
