// REQUIRES: opmodel
// RUN: ttmlir-opt --ttcore-register-device --ttcore-mark-functions-as-forward --ttnn-d2m-fusing --ttnn-optimizer="memory-layout-analysis-enabled=true memreconfig-enabled=true memory-layout-analysis-policy=DFSharding" -o %t %s
// RUN: FileCheck %s --input-file=%t

#layout = #ttnn.ttnn_layout<(d0, d1) -> (d0, d1), <1x1>, memref<2x2x!ttcore.tile<32x32, bf16>, #ttnn.buffer_type<l1>>, <interleaved>>

// CHECK: #[[L1_INTERLEAVED:.*]] = #ttnn.ttnn_layout<(d0, d1) -> (d0, d1), <1x1>, memref<2x2x!ttcore.tile<32x32, bf16>, #l1>, <interleaved>>
// CHECK: #[[DRAM_INTERLEAVED:.*]] = #ttnn.ttnn_layout<(d0, d1) -> (d0, d1), <1x1>, memref<1x1x!ttcore.tile<32x32, bf16>, #dram>, <interleaved>>
// CHECK: #[[BLOCK_SHARDED:.*]] = #ttnn.ttnn_layout<(d0, d1) -> (d0, d1), <2x2, (d0, d1) -> (0, d0, d1)>, memref<1x1x!ttcore.tile<32x32, bf16>, #l1>, <block_sharded>>

module {
  func.func @fork(
      %arg0: tensor<64x64xbf16, #layout>,
      %arg1: tensor<64x64xbf16, #layout>,
      %arg2: tensor<64x64xbf16, #layout>,
      %arg3: tensor<64x64xbf16, #layout>) -> (tensor<64x64xbf16, #layout>, tensor<64x64xbf16, #layout>) {
    %0 = "ttnn.matmul"(%arg0, %arg1) : (tensor<64x64xbf16, #layout>, tensor<64x64xbf16, #layout>) -> tensor<64x64xbf16, #layout>
    // CHECK: %[[DEVICE:.*]] = "ttnn.get_device"()
    // CHECK: %[[MATMUL_OUT:.*]] = "ttnn.matmul"(%arg0, %arg1){{.*}} -> tensor<64x64xbf16, #[[BLOCK_SHARDED]]>
    // CHECK: %[[SPILLED:.*]] = "ttnn.to_layout"(%[[MATMUL_OUT]]){{.*}} -> tensor<64x64xbf16, #[[DRAM_INTERLEAVED]]>
    %1 = "ttnn.add"(%0, %arg2) <{dtype = #ttcore.supportedDataTypes<bf16>}> : (tensor<64x64xbf16, #layout>, tensor<64x64xbf16, #layout>) -> tensor<64x64xbf16, #layout>
    %2 = "ttnn.multiply"(%1, %arg0) <{dtype = #ttcore.supportedDataTypes<bf16>}> : (tensor<64x64xbf16, #layout>, tensor<64x64xbf16, #layout>) -> tensor<64x64xbf16, #layout>
    // CHECK: %[[OUTPUT_BUF:.*]] = "ttnn.empty"(%[[DEVICE]]){{.*}} -> tensor<64x64xbf16, #[[BLOCK_SHARDED]]>
    // CHECK: %[[DISPATCH_OUT:.*]] = ttnn.d2m_subgraph @d2m_subgraph_0
    // CHECK: ins(%[[SPILLED]], %arg2, %arg0
    // CHECK: outs(%[[OUTPUT_BUF]]
    // CHECK: %[[DISPATCH_SPILLED:.*]] = "ttnn.to_layout"(%[[DISPATCH_OUT]]){{.*}} -> tensor<64x64xbf16, #[[DRAM_INTERLEAVED]]>
    %3 = "ttnn.matmul"(%0, %arg3) : (tensor<64x64xbf16, #layout>, tensor<64x64xbf16, #layout>) -> tensor<64x64xbf16, #layout>
    // CHECK: %[[MM2:.*]] = "ttnn.matmul"(%[[SPILLED]], %arg3){{.*}} -> tensor<64x64xbf16, #[[BLOCK_SHARDED]]>
    // CHECK: %[[MM2_SPILLED:.*]] = "ttnn.to_layout"(%[[MM2]]){{.*}} -> tensor<64x64xbf16, #[[DRAM_INTERLEAVED]]>
    return %2, %3 : tensor<64x64xbf16, #layout>, tensor<64x64xbf16, #layout>
    // CHECK: return %[[DISPATCH_SPILLED]], %[[MM2_SPILLED]] : tensor<64x64xbf16, #[[DRAM_INTERLEAVED]]>, tensor<64x64xbf16, #[[DRAM_INTERLEAVED]]>
  }

  // CHECK: func.func private @d2m_subgraph_0(%arg0: tensor<64x64xbf16, #[[DRAM_INTERLEAVED]]>, %arg1: tensor<64x64xbf16, #[[L1_INTERLEAVED]]>, %arg2: tensor<64x64xbf16, #[[L1_INTERLEAVED]]>) -> tensor<64x64xbf16, #[[BLOCK_SHARDED]]> {
  // CHECK: "ttnn.add"(
  // CHECK: "ttnn.multiply"(
}
