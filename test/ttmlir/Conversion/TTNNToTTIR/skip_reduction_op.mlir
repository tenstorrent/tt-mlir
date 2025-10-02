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

module {
    func.func @test_sum_no_tag(%arg0: tensor<32x32xf32, #dram_layout>) -> tensor<32xf32, #dram_layout> {
        %1 = "ttnn.to_memory_config"(%arg0) <{memory_config = #ttnn.memory_config<#l1, <block_sharded>, #ttnn.shard_spec<<[#ttnn.core_range<(0,0), (0,0)>]>, <32x32>, <row_major>>>}> : (tensor<32x32xf32, #dram_layout>) -> tensor<32x32xf32, #l1_layout>

        // CHECK-NOT: %{{[0-9]+}} = ttir.empty() : tensor<32xf32, #ttnn_layout1>
        // CHECK-NOT: %{{[0-9]+}} = "ttir.sum"(%{{[0-9]+}}, %{{[0-9]+}}) <{dim_arg = array<i32: 1>, keep_dim = false}> : (tensor<32x32xf32, #ttnn_layout1>, tensor<32xf32, #ttnn_layout1>) -> tensor<32xf32, #ttnn_layout1>
        // CHECK: %{{[0-9]+}} = "ttnn.sum"(%{{[0-9]+}}) <{dim_arg = [1 : i32], keep_dim = false}> : (tensor<32x32xf32, #ttnn_layout1>) -> tensor<32xf32, #ttnn_layout1>
        %2 = "ttnn.sum"(%1) {dim_arg = [1 : i32], keep_dim = false} : (tensor<32x32xf32, #l1_layout>) -> tensor<32xf32, #l1_layout>

        %3 = "ttnn.to_memory_config"(%2) <{memory_config = #ttnn.memory_config<#dram, <interleaved>>}> : (tensor<32xf32, #l1_layout>) -> tensor<32xf32, #dram_layout>

        return %3 : tensor<32xf32, #dram_layout>
    }

    func.func @test_argmax_no_tag(%arg0: tensor<32x32xf32, #dram_layout>) -> tensor<32xi32, #dram_layout> {
        %1 = "ttnn.to_memory_config"(%arg0) <{memory_config = #ttnn.memory_config<#l1, <block_sharded>, #ttnn.shard_spec<<[#ttnn.core_range<(0,0), (0,0)>]>, <32x32>, <row_major>>>}> : (tensor<32x32xf32, #dram_layout>) -> tensor<32x32xf32, #l1_layout>

        // CHECK-NOT: %{{[0-9]+}} = ttir.empty() : tensor<32xi32, #ttnn_layout1>
        // CHECK-NOT: %{{[0-9]+}} = "ttir.argmax"(%{{[0-9]+}}, %{{[0-9]+}}) <{dim_arg = [1 : i32], keep_dim = false}> : (tensor<32x32xf32, #ttnn_layout1>, tensor<32xi32, #ttnn_layout1>) -> tensor<32xi32, #ttnn_layout1>
        // CHECK: %{{[0-9]+}} = "ttnn.argmax"(%{{[0-9]+}}) <{dim = 1 : i32, keep_dim = false, use_multicore = true}> : (tensor<32x32xf32, #ttnn_layout1>) -> tensor<32xi32, #ttnn_layout1>
        %2 = "ttnn.argmax"(%1) {dim = 1 : i32, keep_dim = false, use_multicore = true} : (tensor<32x32xf32, #l1_layout>) -> tensor<32xi32, #l1_layout>

        %3 = "ttnn.to_memory_config"(%2) <{memory_config = #ttnn.memory_config<#dram, <interleaved>>}> : (tensor<32xi32, #l1_layout>) -> tensor<32xi32, #dram_layout>

        return %3 : tensor<32xi32, #dram_layout>
    }

    func.func @test_moreh_cumsum_no_tag(%arg0: tensor<32x32xf32, #dram_layout>) -> tensor<32x32xf32, #dram_layout> {
        %1 = "ttnn.to_memory_config"(%arg0) <{memory_config = #ttnn.memory_config<#l1, <block_sharded>, #ttnn.shard_spec<<[#ttnn.core_range<(0,0), (0,0)>]>, <32x32>, <row_major>>>}> : (tensor<32x32xf32, #dram_layout>) -> tensor<32x32xf32, #l1_layout>

        // CHECK-NOT: %{{[0-9]+}} = ttir.empty() : tensor<32x32xf32, #ttnn_layout1>
        // CHECK-NOT: %{{[0-9]+}} = "ttir.cumsum"(%{{[0-9]+}}, %{{[0-9]+}}) <{dim = 1 : i64}> : (tensor<32x32xf32, #ttnn_layout1>, tensor<32x32xf32, #ttnn_layout1>) -> tensor<32x32xf32, #ttnn_layout1>
        // CHECK: %{{[0-9]+}} = "ttnn.moreh_cumsum"(%{{[0-9]+}}) <{dim = 1 : i64}> : (tensor<32x32xf32, #ttnn_layout1>) -> tensor<32x32xf32, #ttnn_layout1>
        %2 = "ttnn.moreh_cumsum"(%1) {dim = 1 : i64} : (tensor<32x32xf32, #l1_layout>) -> tensor<32x32xf32, #l1_layout>

        %3 = "ttnn.to_memory_config"(%2) <{memory_config = #ttnn.memory_config<#dram, <interleaved>>}> : (tensor<32x32xf32, #l1_layout>) -> tensor<32x32xf32, #dram_layout>

        return %3 : tensor<32x32xf32, #dram_layout>
    }
}
