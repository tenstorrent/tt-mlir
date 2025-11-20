#dram = #ttnn.buffer_type<dram>
#l1 = #ttnn.buffer_type<l1>
#l1_1 = #ttcore.memory_space<l1>
#ttnn_layout = #ttnn.ttnn_layout<(d0, d1) -> (d0, d1), <1x1>, memref<1x1x!ttcore.tile<32x32, f32>, #dram>, <interleaved>>
#ttnn_layout1 = #ttnn.ttnn_layout<(d0, d1) -> (d0, d1), <1x1, (d0, d1) -> (0, d0, d1)>, memref<1x1x!ttcore.tile<32x32, f32>, #l1>, <block_sharded>>
module {
  func.func @abs(%arg0: tensor<32x32xf32, #ttnn_layout>) -> tensor<32x32xf32, #ttnn_layout> {
    %0 = "ttnn.to_memory_config"(%arg0) <{memory_config = #ttnn.memory_config<#l1, <block_sharded>, #ttnn.shard_spec<<[#ttnn.core_range<(0,0), (0,0)>]>, <32x32>, <row_major>>>}> : (tensor<32x32xf32, #ttnn_layout>) -> tensor<32x32xf32, #ttnn_layout1>
    %1 = d2m.empty() : tensor<32x32xf32, #ttnn_layout1>
    %cast = ttir.ttnn_metal_layout_cast %0 : tensor<32x32xf32, #ttnn_layout1> -> memref<1x1x1x1x!ttcore.tile<32x32, f32>, #ttcore.shard<4096x4096, 1>, #l1_1>
    %cast_0 = ttir.ttnn_metal_layout_cast %1 : tensor<32x32xf32, #ttnn_layout1> -> memref<1x1x1x1x!ttcore.tile<32x32, f32>, #ttcore.shard<4096x4096, 1>, #l1_1>
    %alloc = memref.alloc() : memref<1x1x1x1x!ttcore.tile<32x32, f32>, #ttcore.shard<4096x4096, 1>, #l1_1>
    %stream = "d2m.stream_layout"(%cast, %alloc) : (memref<1x1x1x1x!ttcore.tile<32x32, f32>, #ttcore.shard<4096x4096, 1>, #l1_1>, memref<1x1x1x1x!ttcore.tile<32x32, f32>, #ttcore.shard<4096x4096, 1>, #l1_1>) -> memref<1x1x1x1x!ttcore.tile<32x32, f32>, #ttcore.view<map(4)>, #l1_1>
    %alloc_1 = memref.alloc() : memref<1x1x1x1x!ttcore.tile<32x32, f32>, #ttcore.shard<4096x4096, 1>, #l1_1>
    %stream_2 = "d2m.stream_layout"(%cast_0, %alloc_1) : (memref<1x1x1x1x!ttcore.tile<32x32, f32>, #ttcore.shard<4096x4096, 1>, #l1_1>, memref<1x1x1x1x!ttcore.tile<32x32, f32>, #ttcore.shard<4096x4096, 1>, #l1_1>) -> memref<1x1x1x1x!ttcore.tile<32x32, f32>, #ttcore.view<map(4)>, #l1_1>
    d2m.generic {block_factors = [1, 1], grid = #ttcore.grid<1x1>, indexing_maps = [], iterator_types = [], threads = [#d2m.thread<datamovement, @read_kernel>, #d2m.thread<datamovement, @write_kernel>, #d2m.thread<compute, @compute_kernel0>]}
        ins(%stream : memref<1x1x1x1x!ttcore.tile<32x32, f32>, #ttcore.view<map(4)>, #l1_1>)
        outs(%stream_2 : memref<1x1x1x1x!ttcore.tile<32x32, f32>, #ttcore.view<map(4)>, #l1_1>)
    d2m.generic {block_factors = [1, 1], grid = #ttcore.grid<1x1>, indexing_maps = [], iterator_types = [], threads = [#d2m.thread<datamovement, @read_kernel>, #d2m.thread<datamovement, @write_kernel>, #d2m.thread<compute, @compute_kernel0>]}
        ins(%cast : memref<1x1x1x1x!ttcore.tile<32x32, f32>, #ttcore.shard<4096x4096, 1>, #l1_1>)
        outs(%cast_0 : memref<1x1x1x1x!ttcore.tile<32x32, f32>, #ttcore.shard<4096x4096, 1>, #l1_1>)
    %cast_3 = ttir.ttnn_metal_layout_cast %cast_0 : memref<1x1x1x1x!ttcore.tile<32x32, f32>, #ttcore.shard<4096x4096, 1>, #l1_1> -> tensor<32x32xf32, #ttnn_layout1>
    %2 = "ttnn.to_memory_config"(%cast_3) <{memory_config = #ttnn.memory_config<#dram, <interleaved>>}> : (tensor<32x32xf32, #ttnn_layout1>) -> tensor<32x32xf32, #ttnn_layout>
    return %2 : tensor<32x32xf32, #ttnn_layout>
  }
  func.func private @read_kernel() attributes {ttkernel.arg_spec = #ttkernel.arg_spec<rt_args = [<arg_type = buffer_address, operand_index = 0>] ct_args = [<arg_type = cb_port, operand_index = 0>]>, ttkernel.thread = #ttkernel.thread<noc>} {
    %c1_i32 = arith.constant 1 : i32
    %c4096_i32 = arith.constant 4096 : i32
    %c0_i32 = arith.constant 0 : i32
    %0 = ttkernel.get_common_arg_val(%c0_i32) : (i32) -> i32
    %1 = ttkernel.get_compile_time_arg_val(0) : () -> !ttkernel.cb<1024, f32>
    %2 = ttkernel.my_x() : () -> index
    %3 = ttkernel.my_y() : () -> index
    ttkernel.cb_reserve_back(%1, %c1_i32) : (!ttkernel.cb<1024, f32>, i32) -> ()
    %4 = ttkernel.get_write_ptr(%1) : (!ttkernel.cb<1024, f32>) -> i32
    %5 = ttkernel.get_noc_addr(%2, %3, %0) : (index, index, i32) -> !ttkernel.noc_addr
    ttkernel.noc_async_read(%5, %4, %c4096_i32) : (!ttkernel.noc_addr, i32, i32) -> ()
    ttkernel.noc_async_read_barrier() : () -> ()
    ttkernel.cb_push_back(%1, %c1_i32) : (!ttkernel.cb<1024, f32>, i32) -> ()
    return
  }
  func.func private @compute_kernel0() attributes {ttkernel.arg_spec = #ttkernel.arg_spec< ct_args = [<arg_type = cb_port, operand_index = 0>, <arg_type = cb_port, operand_index = 1>]>, ttkernel.thread = #ttkernel.thread<compute>} {
    %c1_i32 = arith.constant 1 : i32
    %c0 = arith.constant 0 : index
    %0 = ttkernel.get_compile_time_arg_val(0) : () -> !ttkernel.cb<1, !ttcore.tile<32x32, f32>>
    %1 = ttkernel.get_compile_time_arg_val(1) : () -> !ttkernel.cb<1, !ttcore.tile<32x32, f32>>
    ttkernel.cb_reserve_back(%1, %c1_i32) : (!ttkernel.cb<1, !ttcore.tile<32x32, f32>>, i32) -> ()
    ttkernel.cb_wait_front(%0, %c1_i32) : (!ttkernel.cb<1, !ttcore.tile<32x32, f32>>, i32) -> ()
    ttkernel.init_sfpu(%0, %1) : (!ttkernel.cb<1, !ttcore.tile<32x32, f32>>, !ttkernel.cb<1, !ttcore.tile<32x32, f32>>) -> ()
    ttkernel.tile_regs_acquire() : () -> ()
    ttkernel.copy_tile_init(%0) : (!ttkernel.cb<1, !ttcore.tile<32x32, f32>>) -> ()
    ttkernel.copy_tile(%0, %c0, %c0) : (!ttkernel.cb<1, !ttcore.tile<32x32, f32>>, index, index) -> ()
    ttkernel.abs_tile_init() : () -> ()
    ttkernel.abs_tile(%c0) : (index) -> ()
    ttkernel.tile_regs_commit() : () -> ()
    ttkernel.tile_regs_wait() : () -> ()
    ttkernel.pack_tile(%c0, %1, %c0, true) : (index, !ttkernel.cb<1, !ttcore.tile<32x32, f32>>, index) -> ()
    ttkernel.tile_regs_release() : () -> ()
    ttkernel.cb_push_back(%1, %c1_i32) : (!ttkernel.cb<1, !ttcore.tile<32x32, f32>>, i32) -> ()
    ttkernel.cb_pop_front(%0, %c1_i32) : (!ttkernel.cb<1, !ttcore.tile<32x32, f32>>, i32) -> ()
    return
  }
  func.func private @write_kernel() attributes {ttkernel.arg_spec = #ttkernel.arg_spec<rt_args = [<arg_type = buffer_address, operand_index = 1>] ct_args = [<arg_type = cb_port, operand_index = 1>]>, ttkernel.thread = #ttkernel.thread<noc>} {
    %c1_i32 = arith.constant 1 : i32
    %c4096_i32 = arith.constant 4096 : i32
    %c0_i32 = arith.constant 0 : i32
    %0 = ttkernel.get_common_arg_val(%c0_i32) : (i32) -> i32
    %1 = ttkernel.get_compile_time_arg_val(0) : () -> !ttkernel.cb<1024, f32>
    %2 = ttkernel.my_x() : () -> index
    %3 = ttkernel.my_y() : () -> index
    ttkernel.cb_wait_front(%1, %c1_i32) : (!ttkernel.cb<1024, f32>, i32) -> ()
    %4 = ttkernel.get_read_ptr(%1) : (!ttkernel.cb<1024, f32>) -> i32
    %5 = ttkernel.get_noc_addr(%2, %3, %0) : (index, index, i32) -> !ttkernel.noc_addr
    ttkernel.noc_async_write(%4, %5, %c4096_i32) : (i32, !ttkernel.noc_addr, i32) -> ()
    ttkernel.noc_async_write_barrier() : () -> ()
    ttkernel.cb_pop_front(%1, %c1_i32) : (!ttkernel.cb<1024, f32>, i32) -> ()
    return
  }
}
