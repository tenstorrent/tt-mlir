// RUN: ttmlir-opt --convert-ttnn-to-ttir --ttcore-register-device --ttir-to-d2m="ttnn-mode=true" -o %t.mlir %s
// RUN: FileCheck %s --input-file=%t.mlir

#dram = #ttnn.buffer_type<dram>
#l1 = #ttnn.buffer_type<l1>

#core_range = #ttnn.core_range<(0,0), (0,0)>
#core_ranges = #ttnn.core_range_set<[#core_range]>

#dram_memory_config = #ttnn.memory_config<#dram, <interleaved>>
#l1_memory_config = #ttnn.memory_config<#l1, <block_sharded>, #ttnn.shard_spec<<[#core_range]>, <32x32>, <row_major>>>

#ttnn_layout = #ttnn.ttnn_layout<(d0, d1) -> (d0, d1), <1x1>, memref<1x1x!ttcore.tile<32x32, f32>, #dram>, <interleaved>>
#ttnn_layout1 = #ttnn.ttnn_layout<(d0, d1) -> (d0, d1), <1x1, (d0, d1) -> (d0, d1)>, memref<1x1x!ttcore.tile<32x32, f32>, #l1>, <block_sharded>>

module {
    func.func @test(%arg0: tensor<32x32xf32, #ttnn_layout>, %arg1: tensor<32x32xf32, #ttnn_layout>) -> tensor<32x32xf32, #ttnn_layout> {
        %1 = "ttnn.to_memory_config"(%arg0) <{memory_config = #ttnn.memory_config<#l1, <block_sharded>, #ttnn.shard_spec<<[#ttnn.core_range<(0,0), (0,0)>]>, <32x32>, <row_major>>>}> : (tensor<32x32xf32, #ttnn_layout>) -> tensor<32x32xf32, #ttnn_layout1>
        %2 = "ttnn.to_memory_config"(%arg1) <{memory_config = #ttnn.memory_config<#l1, <block_sharded>, #ttnn.shard_spec<<[#ttnn.core_range<(0,0), (0,0)>]>, <32x32>, <row_major>>>}> : (tensor<32x32xf32, #ttnn_layout>) -> tensor<32x32xf32, #ttnn_layout1>
        // CHECK: %[[memConfig1:.*]] = "ttnn.to_memory_config"(%arg0) <{memory_config = #ttnn.memory_config<#l1, <block_sharded>, #ttnn.shard_spec<<[#ttnn.core_range<(0,0), (0,0)>]>, <32x32>, <row_major>>>}> : (tensor<32x32xf32, #ttnn_layout>) -> tensor<32x32xf32, #ttnn_layout1>
        // CHECK: %[[memConfig2:.*]] = "ttnn.to_memory_config"(%arg1) <{memory_config = #ttnn.memory_config<#l1, <block_sharded>, #ttnn.shard_spec<<[#ttnn.core_range<(0,0), (0,0)>]>, <32x32>, <row_major>>>}> : (tensor<32x32xf32, #ttnn_layout>) -> tensor<32x32xf32, #ttnn_layout1>
        // CHECK: %[[TTNNResult:.*]] = d2m.empty() : tensor<32x32xf32, #ttnn_layout1>
        // CHECK: %[[cast1:.*]] = ttir.ttnn_metal_layout_cast %[[memConfig1]] : tensor<32x32xf32, #ttnn_layout1> -> tensor<1x1x1x1x!ttcore.tile<32x32, f32>, #layout>
        // CHECK: %[[cast2:.*]] = ttir.ttnn_metal_layout_cast %[[memConfig2]] : tensor<32x32xf32, #ttnn_layout1> -> tensor<1x1x1x1x!ttcore.tile<32x32, f32>, #layout>
        // CHECK: %[[cast3:.*]] = ttir.ttnn_metal_layout_cast %[[TTNNResult]] : tensor<32x32xf32, #ttnn_layout1> -> tensor<1x1x1x1x!ttcore.tile<32x32, f32>, #layout>
        // CHECK: %[[MetalResult:.*]] = d2m.generic{{.*}}
        // CHECK: ins(%[[cast1]], %[[cast2]] : tensor<1x1x1x1x!ttcore.tile<32x32, f32>, #layout>, tensor<1x1x1x1x!ttcore.tile<32x32, f32>, #layout>)
        // CHECK: outs(%[[cast3]] : tensor<1x1x1x1x!ttcore.tile<32x32, f32>, #layout>)
        // CHECK-DAG: d2m.tile_add
        %3 = "ttnn.add"(%1, %2) {ttnn.hoist_generic_via_d2m, dtype = #ttcore.supportedDataTypes<f32>} : (tensor<32x32xf32, #ttnn_layout1>, tensor<32x32xf32, #ttnn_layout1>) -> tensor<32x32xf32, #ttnn_layout1>
        // CHECK: %[[cast4:.*]] = ttir.ttnn_metal_layout_cast %[[MetalResult]] : tensor<1x1x1x1x!ttcore.tile<32x32, f32>, #layout> -> tensor<32x32xf32, #ttnn_layout1>
        // CHECK: %[[memConfig3:.*]] = "ttnn.to_memory_config"(%[[cast4]]) <{memory_config = #ttnn.memory_config<#dram, <interleaved>>}> : (tensor<32x32xf32, #ttnn_layout1>) -> tensor<32x32xf32, #ttnn_layout>
        %4 = "ttnn.to_memory_config"(%3) <{memory_config = #ttnn.memory_config<#dram, <interleaved>>}> : (tensor<32x32xf32, #ttnn_layout1>) -> tensor<32x32xf32, #ttnn_layout>
        // CHECK: return %[[memConfig3]] : tensor<32x32xf32, #ttnn_layout>
        return %4 : tensor<32x32xf32, #ttnn_layout>
    }
}
