// RUN: ttmlir-opt --convert-ttnn-to-ttir --mlir-print-local-scope -o %t.mlir %s
// RUN: FileCheck %s --input-file=%t.mlir

#dram = #ttnn.buffer_type<dram>
#l1 = #ttnn.buffer_type<l1>

#dram_layout = #ttnn.ttnn_layout<(d0, d1) -> (d0, d1), <1x1>, memref<1x1x!ttcore.tile<32x32, bf16>, #dram>, <interleaved>>
#l1_layout = #ttnn.ttnn_layout<(d0, d1) -> (d0, d1), <1x1>, memref<1x1x!ttcore.tile<32x32, bf16>, #l1>, <block_sharded>>

module {
  func.func @test(%arg0: tensor<32x32xbf16, #dram_layout>) -> tensor<32x32xbf16, #l1_layout> {
    // CHECK: %[[EMPTY:.*]] = ttir.empty() : tensor<32x32xbf16,
    // CHECK-SAME: #ttnn.buffer_type<l1>>, <block_sharded>>>
    // CHECK: %[[RESULT:.*]] = ttir.to_layout %arg0, %[[EMPTY]] :
    // CHECK-SAME: #ttnn.buffer_type<dram>>, <interleaved>>>
    // CHECK-SAME: into
    // CHECK-SAME: #ttnn.buffer_type<l1>>, <block_sharded>>>
    // CHECK-SAME: ->
    // CHECK-SAME: #ttnn.buffer_type<l1>>, <block_sharded>>>
    // CHECK-NOT: "ttnn.to_layout"
    %0 = "ttnn.to_layout"(%arg0) <{dtype = #ttcore.supportedDataTypes<bf16>, layout = #ttnn.layout<tile>, memory_config = #ttnn.memory_config<#l1, <block_sharded>, #ttnn.shard_spec<<[#ttnn.core_range<(0,0), (0,0)>]>, <32x32>, <row_major>>>}> {ttnn.hoist_generic_via_d2m} : (tensor<32x32xbf16, #dram_layout>) -> tensor<32x32xbf16, #l1_layout>
    return %0 : tensor<32x32xbf16, #l1_layout>
  }
}
