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
    func.func @test_sum(%arg0: tensor<32x32xf32, #dram_layout>) -> tensor<32xf32, #dram_layout> {
        %1 = "ttnn.to_memory_config"(%arg0) <{memory_config = #ttnn.memory_config<#l1, <block_sharded>, #ttnn.shard_spec<<[#ttnn.core_range<(0,0), (0,0)>]>, <32x32>, <row_major>>>}> : (tensor<32x32xf32, #dram_layout>) -> tensor<32x32xf32, #l1_layout>

        // CHECK: %{{[0-9]+}} = ttir.empty() : tensor<32xf32, #ttnn_layout1>
        // CHECK: %{{[0-9]+}} = "ttir.sum"(%{{[0-9]+}}, %{{[0-9]+}}) <{dim_arg = [1 : i32], keep_dim = false}> : (tensor<32x32xf32, #ttnn_layout1>, tensor<32xf32, #ttnn_layout1>) -> tensor<32xf32, #ttnn_layout1>
        // CHECK-NOT: "ttnn.sum"
        %2 = "ttnn.sum"(%1) {ttnn.hoist_generic_via_d2m, dim_arg = [1 : i32], keep_dim = false} : (tensor<32x32xf32, #l1_layout>) -> tensor<32xf32, #l1_layout>

        %3 = "ttnn.to_memory_config"(%2) <{memory_config = #ttnn.memory_config<#dram, <interleaved>>}> : (tensor<32xf32, #l1_layout>) -> tensor<32xf32, #dram_layout>

        return %3 : tensor<32xf32, #dram_layout>
    }

    func.func @test_mean(%arg0: tensor<32x32xf32, #dram_layout>) -> tensor<32xf32, #dram_layout> {
        %1 = "ttnn.to_memory_config"(%arg0) <{memory_config = #ttnn.memory_config<#l1, <block_sharded>, #ttnn.shard_spec<<[#ttnn.core_range<(0,0), (0,0)>]>, <32x32>, <row_major>>>}> : (tensor<32x32xf32, #dram_layout>) -> tensor<32x32xf32, #l1_layout>

        // CHECK: %{{[0-9]+}} = ttir.empty() : tensor<32xf32, #ttnn_layout1>
        // CHECK: %{{[0-9]+}} = "ttir.mean"(%{{[0-9]+}}, %{{[0-9]+}}) <{dim_arg = [1 : i32], keep_dim = false}> : (tensor<32x32xf32, #ttnn_layout1>, tensor<32xf32, #ttnn_layout1>) -> tensor<32xf32, #ttnn_layout1>
        // CHECK-NOT: "ttnn.mean"
        %2 = "ttnn.mean"(%1) {ttnn.hoist_generic_via_d2m, dim_arg = [1 : i32], keep_dim = false} : (tensor<32x32xf32, #l1_layout>) -> tensor<32xf32, #l1_layout>

        %3 = "ttnn.to_memory_config"(%2) <{memory_config = #ttnn.memory_config<#dram, <interleaved>>}> : (tensor<32xf32, #l1_layout>) -> tensor<32xf32, #dram_layout>

        return %3 : tensor<32xf32, #dram_layout>
    }

    func.func @test_max(%arg0: tensor<32x32xf32, #dram_layout>) -> tensor<32xf32, #dram_layout> {
        %1 = "ttnn.to_memory_config"(%arg0) <{memory_config = #ttnn.memory_config<#l1, <block_sharded>, #ttnn.shard_spec<<[#ttnn.core_range<(0,0), (0,0)>]>, <32x32>, <row_major>>>}> : (tensor<32x32xf32, #dram_layout>) -> tensor<32x32xf32, #l1_layout>

        // CHECK: %{{[0-9]+}} = ttir.empty() : tensor<32xf32, #ttnn_layout1>
        // CHECK: %{{[0-9]+}} = "ttir.max"(%{{[0-9]+}}, %{{[0-9]+}}) <{dim_arg = [1 : i32], keep_dim = false}> : (tensor<32x32xf32, #ttnn_layout1>, tensor<32xf32, #ttnn_layout1>) -> tensor<32xf32, #ttnn_layout1>
        // CHECK-NOT: "ttnn.max"
        %2 = "ttnn.max"(%1) {ttnn.hoist_generic_via_d2m, dim_arg = [1 : i32], keep_dim = false} : (tensor<32x32xf32, #l1_layout>) -> tensor<32xf32, #l1_layout>

        %3 = "ttnn.to_memory_config"(%2) <{memory_config = #ttnn.memory_config<#dram, <interleaved>>}> : (tensor<32xf32, #l1_layout>) -> tensor<32xf32, #dram_layout>

        return %3 : tensor<32xf32, #dram_layout>
    }

    func.func @test_min(%arg0: tensor<32x32xf32, #dram_layout>) -> tensor<32xf32, #dram_layout> {
        %1 = "ttnn.to_memory_config"(%arg0) <{memory_config = #ttnn.memory_config<#l1, <block_sharded>, #ttnn.shard_spec<<[#ttnn.core_range<(0,0), (0,0)>]>, <32x32>, <row_major>>>}> : (tensor<32x32xf32, #dram_layout>) -> tensor<32x32xf32, #l1_layout>

        // CHECK: %{{[0-9]+}} = ttir.empty() : tensor<32xf32, #ttnn_layout1>
        // CHECK: %{{[0-9]+}} = "ttir.min"(%{{[0-9]+}}, %{{[0-9]+}}) <{dim_arg = [1 : i32], keep_dim = false}> : (tensor<32x32xf32, #ttnn_layout1>, tensor<32xf32, #ttnn_layout1>) -> tensor<32xf32, #ttnn_layout1>
        // CHECK-NOT: "ttnn.min"
        %2 = "ttnn.min"(%1) {ttnn.hoist_generic_via_d2m, dim_arg = [1 : i32], keep_dim = false} : (tensor<32x32xf32, #l1_layout>) -> tensor<32xf32, #l1_layout>

        %3 = "ttnn.to_memory_config"(%2) <{memory_config = #ttnn.memory_config<#dram, <interleaved>>}> : (tensor<32xf32, #l1_layout>) -> tensor<32xf32, #dram_layout>

        return %3 : tensor<32xf32, #dram_layout>
    }

    func.func @test_prod(%arg0: tensor<32x32xf32, #dram_layout>) -> tensor<32xf32, #dram_layout> {
        %1 = "ttnn.to_memory_config"(%arg0) <{memory_config = #ttnn.memory_config<#l1, <block_sharded>, #ttnn.shard_spec<<[#ttnn.core_range<(0,0), (0,0)>]>, <32x32>, <row_major>>>}> : (tensor<32x32xf32, #dram_layout>) -> tensor<32x32xf32, #l1_layout>

        // CHECK: %{{[0-9]+}} = ttir.empty() : tensor<32xf32, #ttnn_layout1>
        // CHECK: %{{[0-9]+}} = "ttir.prod"(%{{[0-9]+}}, %{{[0-9]+}}) <{dim_arg = [1 : i32], keep_dim = false}> : (tensor<32x32xf32, #ttnn_layout1>, tensor<32xf32, #ttnn_layout1>) -> tensor<32xf32, #ttnn_layout1>
        // CHECK-NOT: "ttnn.prod"
        %2 = "ttnn.prod"(%1) {ttnn.hoist_generic_via_d2m, dim_arg = 1, keep_dim = false} : (tensor<32x32xf32, #l1_layout>) -> tensor<32xf32, #l1_layout>

        %3 = "ttnn.to_memory_config"(%2) <{memory_config = #ttnn.memory_config<#dram, <interleaved>>}> : (tensor<32xf32, #l1_layout>) -> tensor<32xf32, #dram_layout>

        return %3 : tensor<32xf32, #dram_layout>
    }

}