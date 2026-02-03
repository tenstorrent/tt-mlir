// REQUIRES: opmodel
// RUN: ttmlir-opt --ttcore-register-device --ttcore-mark-functions-as-forward --ttnn-d2m-fusing --ttnn-optimizer="memory-layout-analysis-enabled=true memreconfig-enabled=true memory-layout-analysis-policy=DFSharding" -o %t %s
// RUN: FileCheck %s --input-file=%t

#layout = #ttnn.ttnn_layout<(d0, d1) -> (d0, d1), <1x1>, memref<2x2x!ttcore.tile<32x32, bf16>, #ttnn.buffer_type<l1>>, <interleaved>>

// CHECK: #[[L1_INTERLEAVED:.*]] = #ttnn.ttnn_layout<(d0, d1) -> (d0, d1), <1x1>, memref<2x2x!ttcore.tile<32x32, bf16>, #l1>, <interleaved>>
// CHECK: #[[DRAM_INTERLEAVED:.*]] = #ttnn.ttnn_layout<(d0, d1) -> (d0, d1), <1x1>, memref<1x1x!ttcore.tile<32x32, bf16>, #dram>, <interleaved>>
// CHECK: #[[BLOCK_SHARDED:.*]] = #ttnn.ttnn_layout<(d0, d1) -> (d0, d1), <2x2, (d0, d1) -> (0, d0, d1)>, memref<1x1x!ttcore.tile<32x32, bf16>, #l1>, <block_sharded>>

module {
  func.func @sync_noop(
      %arg0: tensor<64x64xbf16, #layout>,
      %arg1: tensor<64x64xbf16, #layout>) -> tensor<64x64xbf16, #layout> {
    %0 = "ttnn.add"(%arg0, %arg1) <{dtype = #ttcore.supportedDataTypes<bf16>}> : (tensor<64x64xbf16, #layout>, tensor<64x64xbf16, #layout>) -> tensor<64x64xbf16, #layout>
    %1 = "ttnn.multiply"(%0, %arg0) <{dtype = #ttcore.supportedDataTypes<bf16>}> : (tensor<64x64xbf16, #layout>, tensor<64x64xbf16, #layout>) -> tensor<64x64xbf16, #layout>
    // No to_layout before d2m_subgraph: inputs stay L1 (sync no-op).
    // CHECK: %[[DEVICE:.*]] = "ttnn.get_device"()
    // CHECK: %[[OUTPUT_BUFFER:.*]] = "ttnn.empty"(%[[DEVICE]]){{.*}} -> tensor<64x64xbf16, #[[BLOCK_SHARDED]]>
    // CHECK: %[[DISPATCH_OUTPUT:.*]] = ttnn.d2m_subgraph @d2m_subgraph_0
    // CHECK: ins(%arg0, %arg1
    // CHECK: outs(%[[OUTPUT_BUFFER]]
    // CHECK: %[[SPILLED:.*]] = "ttnn.to_layout"(%[[DISPATCH_OUTPUT]]){{.*}} -> tensor<64x64xbf16, #[[DRAM_INTERLEAVED]]>
    return %1 : tensor<64x64xbf16, #layout>
    // CHECK: return %[[SPILLED]] : tensor<64x64xbf16, #[[DRAM_INTERLEAVED]]>
  }

  // CHECK: func.func private @d2m_subgraph_0(%arg0: tensor<64x64xbf16, #[[L1_INTERLEAVED]]>, %arg1: tensor<64x64xbf16, #[[L1_INTERLEAVED]]>) -> tensor<64x64xbf16, #[[BLOCK_SHARDED]]> {
  // CHECK: "ttnn.add"(
  // CHECK: "ttnn.multiply"(
}
