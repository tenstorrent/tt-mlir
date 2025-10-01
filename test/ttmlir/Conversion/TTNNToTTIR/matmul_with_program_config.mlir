// RUN: ttmlir-opt --convert-ttnn-to-ttir -o %t.mlir %s
// RUN: FileCheck %s --input-file=%t.mlir

#dram = #ttnn.buffer_type<dram>
#l1 = #ttnn.buffer_type<l1>

#core_range = #ttnn.core_range<(0,0), (0,0)>
#core_ranges = #ttnn.core_range_set<[#core_range]>

#dram_memory_config = #ttnn.memory_config<#dram, <interleaved>>
#l1_memory_config = #ttnn.memory_config<#l1, <block_sharded>, #ttnn.shard_spec<<[#core_range]>, <32x32>, <row_major>>>

#dram_layout = #ttnn.ttnn_layout<(d0, d1) -> (d0, d1), <1x1>, memref<1x1x!ttcore.tile<32x32, f32>, #dram>, <interleaved>>
#l1_layout = #ttnn.ttnn_layout<(d0, d1) -> (d0, d1), <1x1, (d0, d1) -> (0, d0, d1)>, memref<1x1x!ttcore.tile<32x32, f32>, #l1>, <block_sharded>>

#matmul_program_config = #ttnn.matmul_multi_core_reuse_program_config<
  compute_with_storage_grid_size = #ttnn.core_coord<7, 9>,
  in0_block_w = 8,
  out_subblock_h = 1,
  out_subblock_w = 8,
  per_core_m = 8,
  per_core_n = 8
>

module {
    func.func @test_matmul_with_config(%arg0: tensor<32x32xf32, #dram_layout>, %arg1: tensor<32x32xf32, #dram_layout>) -> tensor<32x32xf32, #dram_layout> {
        %1 = "ttnn.to_memory_config"(%arg0) <{memory_config = #ttnn.memory_config<#l1, <block_sharded>, #ttnn.shard_spec<<[#ttnn.core_range<(0,0), (0,0)>]>, <32x32>, <row_major>>>}> : (tensor<32x32xf32, #dram_layout>) -> tensor<32x32xf32, #l1_layout>
        %2 = "ttnn.to_memory_config"(%arg1) <{memory_config = #ttnn.memory_config<#l1, <block_sharded>, #ttnn.shard_spec<<[#ttnn.core_range<(0,0), (0,0)>]>, <32x32>, <row_major>>>}> : (tensor<32x32xf32, #dram_layout>) -> tensor<32x32xf32, #l1_layout>

        // CHECK: %{{[0-9]+}} = ttir.empty() : tensor<32x32xf32, #ttnn_layout1>
        // CHECK: %{{[0-9]+}} = "ttir.matmul"(%{{[0-9]+}}, %{{[0-9]+}}, %{{[0-9]+}}) <{transpose_a = false, transpose_b = true}> {matmul_program_config = #ttnn.matmul_multi_core_reuse_program_config<{{.*}}>} : (tensor<32x32xf32, #ttnn_layout1>, tensor<32x32xf32, #ttnn_layout1>, tensor<32x32xf32, #ttnn_layout1>) -> tensor<32x32xf32, #ttnn_layout1>
        // CHECK-NOT: "ttnn.matmul"
        %3 = "ttnn.matmul"(%1, %2) {ttnn.hoist_generic_via_d2m, transpose_a = false, transpose_b = true, matmul_program_config = #matmul_program_config} : (tensor<32x32xf32, #l1_layout>, tensor<32x32xf32, #l1_layout>) -> tensor<32x32xf32, #l1_layout>

        %4 = "ttnn.to_memory_config"(%3) <{memory_config = #ttnn.memory_config<#dram, <interleaved>>}> : (tensor<32x32xf32, #l1_layout>) -> tensor<32x32xf32, #dram_layout>

        return %4 : tensor<32x32xf32, #dram_layout>
    }
}
