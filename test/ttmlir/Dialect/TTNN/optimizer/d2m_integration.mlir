// REQUIRES: opmodel
// RUN: ttmlir-opt --ttcore-register-device --ttcore-mark-functions-as-forward --ttnn-d2m-fusing --ttnn-optimizer="memory-layout-analysis-enabled=true memreconfig-enabled=true memory-layout-analysis-policy=DFSharding" -o %t %s
// RUN: FileCheck %s --input-file=%t

#l1 = #ttnn.buffer_type<l1>
#layout = #ttnn.ttnn_layout<(d0, d1) -> (d0, d1), <1x1>, memref<2x8x!ttcore.tile<32x32, bf16>, #l1>, <interleaved>>

// CHECK: #[[L1_INTERLEAVED:.*]] = #ttnn.ttnn_layout<(d0, d1) -> (d0, d1), <1x1>, memref<2x8x!ttcore.tile<32x32, bf16>, #l1>, <interleaved>>
// CHECK: #[[DRAM_INTERLEAVED:.*]] = #ttnn.ttnn_layout<(d0, d1) -> (d0, d1), <1x1>, memref<1x1x!ttcore.tile<32x32, bf16>, #dram>, <interleaved>>
// CHECK: #[[BLOCK_SHARDED:.*]] = #ttnn.ttnn_layout<(d0, d1) -> (d0, d1), <2x2, (d0, d1) -> (0, d0, d1)>, memref<1x1x!ttcore.tile<32x32, bf16>, #l1>, <block_sharded>>

module {
  func.func @dispatch_with_matmul_producer(%arg0: tensor<64x64xbf16, #layout>, %arg1: tensor<64x64xbf16, #layout>) -> tensor<64x64xbf16, #layout> {
    %0 = "ttnn.matmul"(%arg0, %arg1) : (tensor<64x64xbf16, #layout>, tensor<64x64xbf16, #layout>) -> tensor<64x64xbf16, #layout>
    // Optimizer decides to use 2x2 grid and block_sharded layout for the matmul:
    // CHECK: %[[MATMUL_OUTPUT:.*]] = "ttnn.matmul"{{.*}} -> tensor<64x64xbf16, #[[BLOCK_SHARDED]]>

    // Since the d2m_subgraph is a boundary op, the matmul output is spilled to DRAM:
    // CHECK: %[[MATMUL_SPILLED_TO_DRAM:.*]] = "ttnn.to_layout"(%[[MATMUL_OUTPUT]]){{.*}} -> tensor<64x64xbf16, #[[DRAM_INTERLEAVED]]>
    %1 = "ttnn.exp"(%0) : (tensor<64x64xbf16, #layout>) -> tensor<64x64xbf16, #layout>
    %2 = "ttnn.neg"(%1) : (tensor<64x64xbf16, #layout>) -> tensor<64x64xbf16, #layout>
    %3 = "ttnn.abs"(%2) : (tensor<64x64xbf16, #layout>) -> tensor<64x64xbf16, #layout>
    // The output buffer is determined by the optimizer to  be block_sharded:
    // CHECK: %[[OUTPUT_BUFFER:.*]] = "ttnn.empty"{{.*}} -> tensor<64x64xbf16, #[[BLOCK_SHARDED]]>

    // The d2m_subgraph op is created with the spilled matmul output and the output buffer:
    // CHECK: %[[DISPATCH_D2M_OUTPUT:.*]] = ttnn.d2m_subgraph @d2m_subgraph_0
    // CHECK: ins(%[[MATMUL_SPILLED_TO_DRAM]]
    // CHECK: outs(%[[OUTPUT_BUFFER]]
    // d2m_subgraph output is spilled to DRAM (since d2m_subgraph is a boundary op):
    // CHECK: %[[DISPATCH_D2M_SPILLED_TO_DRAM:.*]] = "ttnn.to_layout"(%[[DISPATCH_D2M_OUTPUT]]){{.*}} -> tensor<64x64xbf16, #[[DRAM_INTERLEAVED]]>

    return %3 : tensor<64x64xbf16, #layout>
    // CHECK: return %[[DISPATCH_D2M_SPILLED_TO_DRAM]] : tensor<64x64xbf16, #[[DRAM_INTERLEAVED]]>
  }

  // We also check for necessary modifications to the internal subgraph function:
  // CHECK: func.func private @d2m_subgraph_{{.+}}(%arg0: tensor<64x64xbf16, #[[DRAM_INTERLEAVED]]>) -> tensor<64x64xbf16, #[[BLOCK_SHARDED]]> {
  // CHECK: %0 = "ttnn.exp"(%arg0) : (tensor<64x64xbf16, #[[DRAM_INTERLEAVED]]>) -> tensor<64x64xbf16, #[[BLOCK_SHARDED]]>
  // CHECK: %1 = "ttnn.neg"(%0) : (tensor<64x64xbf16, #[[BLOCK_SHARDED]]>) -> tensor<64x64xbf16, #[[BLOCK_SHARDED]]>
  // CHECK: %2 = "ttnn.abs"(%1) : (tensor<64x64xbf16, #[[BLOCK_SHARDED]]>) -> tensor<64x64xbf16, #[[BLOCK_SHARDED]]>
  // CHECK: return %2 : tensor<64x64xbf16, #[[BLOCK_SHARDED]]>
}
