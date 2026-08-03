// RUN: ttmlir-opt --convert-ttnn-to-ttir --mlir-print-local-scope -o %t.mlir %s
// RUN: FileCheck %s --input-file=%t.mlir

#dram = #ttnn.buffer_type<dram>
#l1 = #ttnn.buffer_type<l1>

#dram_layout = #ttnn.ttnn_layout<(d0, d1) -> (d0, d1), <1x1>, memref<1x1x!ttcore.tile<32x32, bf16>, #dram>, <interleaved>>
#dram_rm_layout = #ttnn.ttnn_layout<(d0, d1) -> (d0, d1), <1x1>, memref<32x32xbf16, #dram>, <interleaved>>
#l1_layout = #ttnn.ttnn_layout<(d0, d1) -> (d0, d1), <1x1>, memref<1x1x!ttcore.tile<32x32, bf16>, #l1>, <block_sharded>, core_ranges = <[#ttnn.core_range<(0, 0), (0, 0)>]>>

module {
  func.func @test_to_layout(%arg0: tensor<32x32xbf16, #dram_layout>) -> tensor<32x32xbf16, #dram_rm_layout> {
    // CHECK: %[[EMPTY:.*]] = ttir.empty() : tensor<32x32xbf16,
    // CHECK-SAME: memref<32x32xbf16, #ttnn.buffer_type<dram>>, <interleaved>>>
    // CHECK: %[[RESULT:.*]] = ttir.to_layout %arg0, %[[EMPTY]] :
    // CHECK-SAME: memref<1x1x!ttcore.tile<32x32, bf16>, #ttnn.buffer_type<dram>>, <interleaved>>>
    // CHECK-SAME: into
    // CHECK-SAME: memref<32x32xbf16, #ttnn.buffer_type<dram>>, <interleaved>>>
    // CHECK-SAME: ->
    // CHECK-SAME: memref<32x32xbf16, #ttnn.buffer_type<dram>>, <interleaved>>>
    // CHECK-NOT: "ttnn.to_layout"
    %0 = "ttnn.to_layout"(%arg0)  {ttnn.hoist_generic_via_d2m} : (tensor<32x32xbf16, #dram_layout>) -> tensor<32x32xbf16, #dram_rm_layout>
    return %0 : tensor<32x32xbf16, #dram_rm_layout>
  }

  func.func @test_to_tensor_spec(%arg0: tensor<32x32xbf16, #dram_layout>) -> tensor<32x32xbf16, #l1_layout> {
    // CHECK: %[[EMPTY:.*]] = ttir.empty() : tensor<32x32xbf16,
    // CHECK-SAME: #ttnn.buffer_type<l1>>, <block_sharded>, core_ranges = <[#ttnn.core_range<(0,0), (0,0)>]>>>
    // CHECK: %[[RESULT:.*]] = ttir.to_layout %arg0, %[[EMPTY]] :
    // CHECK-SAME: #ttnn.buffer_type<dram>>, <interleaved>>>
    // CHECK-SAME: into
    // CHECK-SAME: #ttnn.buffer_type<l1>>, <block_sharded>, core_ranges = <[#ttnn.core_range<(0,0), (0,0)>]>>>
    // CHECK-SAME: ->
    // CHECK-SAME: #ttnn.buffer_type<l1>>, <block_sharded>, core_ranges = <[#ttnn.core_range<(0,0), (0,0)>]>>>
    // CHECK-NOT: "ttnn.to_tensor_spec"
    %0 = "ttnn.to_tensor_spec"(%arg0)  {ttnn.hoist_generic_via_d2m} : (tensor<32x32xbf16, #dram_layout>) -> tensor<32x32xbf16, #l1_layout>
    return %0 : tensor<32x32xbf16, #l1_layout>
  }
}
