// RUN: ttmlir-opt --convert-ttnn-to-ttir --ttir-to-ttmetal-pipeline="system-desc-path=%system_desc_path% ttnn-mode=true enable-elementwise-fusion=true" -o %t.mlir %s
// RUN: FileCheck %s --input-file=%t.mlir
// RUN: ttmlir-translate --ttnn-to-flatbuffer -o %t.ttnn %t.mlir

#dram = #ttnn.buffer_type<dram>
#l1 = #ttnn.buffer_type<l1>

#core_range = #ttnn.core_range<(0,0), (0,0)>
#core_ranges = #ttnn.core_range_set<[#core_range]>

#dram_memory_config = #ttnn.memory_config<#dram, <interleaved>>
#l1_memory_config = #ttnn.memory_config<#l1, <block_sharded>, #ttnn.shard_spec<<[#core_range]>, <32x32>, <row_major>>>

#dram_layout = #ttnn.ttnn_layout<(d0, d1) -> (d0, d1), <1x1>, memref<1x1x!ttcore.tile<32x32, f32>, #dram>, <interleaved>>
#l1_layout = #ttnn.ttnn_layout<(d0, d1) -> (d0, d1), <1x1, virt_to_physical_map = (d0, d1) -> (0, d0, d1), physical_to_virt_map = (d0, d1, d2) -> (d1, d2)>, memref<1x1x!ttcore.tile<32x32, f32>, #l1>, <block_sharded>>


#dram_layout_0 = #ttnn.ttnn_layout<(d0, d1) -> (d0, d1), <1x1>, memref<4x4x!ttcore.tile<32x32, f32>, #dram>, <interleaved>>
#l1_layout_0 = #ttnn.ttnn_layout<(d0, d1) -> (d0, d1), <1x1, virt_to_physical_map = (d0, d1) -> (0, d0, d1), physical_to_virt_map = (d0, d1, d2) -> (d1, d2)>, memref<4x4x!ttcore.tile<32x32, f32>, #l1>, <block_sharded>>

module {
  // CHECK-LABEL: func.func @test_composite_fuse
  func.func @test_composite_fuse(%arg0: tensor<32x32xf32, #dram_layout>) -> tensor<32x32xf32, #dram_layout> {
    %0 = "ttnn.to_memory_config"(%arg0) <{memory_config = #ttnn.memory_config<#l1, <block_sharded>, #ttnn.shard_spec<<[#ttnn.core_range<(0,0), (0,0)>]>, <32x32>, <row_major>>>}> : (tensor<32x32xf32, #dram_layout>) -> tensor<32x32xf32, #l1_layout>
    // CHECK: %[[OUT0:.*]] = "ttnn.to_memory_config"
    %2 = "ttnn.abs"(%0) {ttnn.hoist_generic_via_d2m} : (tensor<32x32xf32, #l1_layout>) -> tensor<32x32xf32, #l1_layout>
    // CHECK: %[[OUT1:.*]] = "ttnn.empty"
    // CHECK: "ttnn.generic"(%[[OUT0]], %[[OUT1]])
    %3 = "ttnn.neg"(%2) {ttnn.hoist_generic_via_d2m} : (tensor<32x32xf32, #l1_layout>) -> tensor<32x32xf32, #l1_layout>
    %4 = "ttnn.to_memory_config"(%3) <{memory_config = #ttnn.memory_config<#dram, <interleaved>>}> : (tensor<32x32xf32, #l1_layout>) -> tensor<32x32xf32, #dram_layout>
    return %4 : tensor<32x32xf32, #dram_layout>
  }

  // TODO: need to address fusion concerns to fix removed tests
}
