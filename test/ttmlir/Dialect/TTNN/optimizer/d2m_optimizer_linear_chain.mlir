// REQUIRES: opmodel
// RUN: ttmlir-opt --ttcore-register-device --ttcore-mark-functions-as-forward --ttnn-d2m-fusing --ttnn-optimizer="memory-layout-analysis-enabled=true memreconfig-enabled=true memory-layout-analysis-policy=DFSharding" -o %t %s
// RUN: FileCheck %s --input-file=%t

// Linear chain with D2M: matmul -> matmul -> d2m_subgraph -> matmul.
// D2MSubgraphOp is not in validForSharding, so it is not part of an L1 chain and
// resolves to DRAM interleaved by default. The chain breaks at D2M: the first
// two matmuls form an L1 chain (spilled to DRAM before D2M), D2M in/out are
// DRAM, and the last matmul forms a separate L1 chain (spilled to DRAM for return).
// This issue tracks allowing D2M to be part of an L1 chain: https://github.com/tenstorrent/tt-mlir/issues/7025

#layout = #ttnn.ttnn_layout<(d0, d1) -> (d0, d1), <1x1>, memref<2x2x!ttcore.tile<32x32, bf16>, #ttnn.buffer_type<l1>>, <interleaved>>
// CHECK: #[[L1_INTERLEAVED:.*]] = #ttnn.ttnn_layout<(d0, d1) -> (d0, d1), <1x1>, memref<2x2x!ttcore.tile<32x32, bf16>, #l1>, <interleaved>>
// CHECK: #[[DRAM_INTERLEAVED:.*]] = #ttnn.ttnn_layout<(d0, d1) -> (d0, d1), <1x1>, memref<1x1x!ttcore.tile<32x32, bf16>, #dram>, <interleaved>>
// CHECK: #[[L1_BLOCK_SHARDED_A:.*]] = #ttnn.ttnn_layout<{{.*}}>, memref<1x2x!ttcore.tile<32x32, bf16>, #l1>, <block_sharded>>
// CHECK: #[[L1_BLOCK_SHARDED_B:.*]] = #ttnn.ttnn_layout<{{.*}}>, memref<1x1x!ttcore.tile<32x32, bf16>, #l1>, <block_sharded>>
// CHECK: #[[DRAM_INTERLEAVED_2:.*]] = #ttnn.ttnn_layout<(d0, d1) -> (d0, d1), <1x1>, memref<2x2x!ttcore.tile<32x32, bf16>, #dram>, <interleaved>>

module {
  func.func @simple_64x64(
      %arg0: tensor<64x64xbf16, #layout>,
      %arg1: tensor<64x64xbf16, #layout>,
      %arg2: tensor<64x64xbf16, #layout>,
      %arg3: tensor<64x64xbf16, #layout>,
      %arg4: tensor<64x64xbf16, #layout>) -> tensor<64x64xbf16, #layout> {
    %0 = "ttnn.matmul"(%arg0, %arg1) : (tensor<64x64xbf16, #layout>, tensor<64x64xbf16, #layout>) -> tensor<64x64xbf16, #layout>
    // CHECK: %[[MATMUL_0_OUT:.*]] = "ttnn.matmul"
    // CHECK-SAME: -> tensor<64x64xbf16, #[[L1_BLOCK_SHARDED_A]]>
    %1 = "ttnn.matmul"(%0, %arg2) : (tensor<64x64xbf16, #layout>, tensor<64x64xbf16, #layout>) -> tensor<64x64xbf16, #layout>
    // CHECK: %[[MATMUL_1_OUT:.*]] = "ttnn.matmul"
    // CHECK-SAME: -> tensor<64x64xbf16, #[[L1_BLOCK_SHARDED_B]]>
    // CHECK: %[[SPILLED:.*]] = "ttnn.to_layout"(%[[MATMUL_1_OUT]])
    // CHECK-SAME: -> tensor<64x64xbf16, #[[DRAM_INTERLEAVED]]>
    %2 = "ttnn.add"(%1, %arg3) <{dtype = #ttcore.supportedDataTypes<bf16>}> : (tensor<64x64xbf16, #layout>, tensor<64x64xbf16, #layout>) -> tensor<64x64xbf16, #layout>
    %3 = "ttnn.multiply"(%2, %arg0) <{dtype = #ttcore.supportedDataTypes<bf16>}> : (tensor<64x64xbf16, #layout>, tensor<64x64xbf16, #layout>) -> tensor<64x64xbf16, #layout>
    // CHECK: %[[D2M_OUTPUT_BUFFER:.*]] = "ttnn.empty"
    // CHECK-SAME: -> tensor<64x64xbf16, #[[DRAM_INTERLEAVED_2]]>
    // CHECK: %[[D2M_SUBGRAPH_OUT:.*]] = ttnn.d2m_subgraph @d2m_subgraph_0
    // CHECK: ins(%[[SPILLED]], %arg3, %arg0 : tensor<64x64xbf16, #[[DRAM_INTERLEAVED]]>, tensor<64x64xbf16, #[[L1_INTERLEAVED]]>, tensor<64x64xbf16, #[[L1_INTERLEAVED]]>)
    // CHECK: outs(%[[D2M_OUTPUT_BUFFER]] : tensor<64x64xbf16, #[[DRAM_INTERLEAVED_2]]>) : tensor<64x64xbf16, #[[DRAM_INTERLEAVED_2]]>
    %4 = "ttnn.matmul"(%3, %arg4) : (tensor<64x64xbf16, #layout>, tensor<64x64xbf16, #layout>) -> tensor<64x64xbf16, #layout>
    // CHECK: %[[MATMUL_2_OUT:.*]] = "ttnn.matmul"
    // CHECK-SAME: -> tensor<64x64xbf16, #[[L1_BLOCK_SHARDED_B]]>
    // CHECK: %[[MATMUL_2_SPILLED:.*]] = "ttnn.to_layout"(%[[MATMUL_2_OUT]])
    // CHECK-SAME: -> tensor<64x64xbf16, #[[DRAM_INTERLEAVED]]>
    return %4 : tensor<64x64xbf16, #layout>
    // CHECK: return %[[MATMUL_2_SPILLED]] : tensor<64x64xbf16, #[[DRAM_INTERLEAVED]]>
  }

  // D2M callee: spilled input (DRAM 1x1) and output (DRAM 2x2); block args from main stay L1.
  // Internal ops keep their natural result types; trailing to_layout converts to chosen layout.
  // CHECK: func.func private @d2m_subgraph_0
  // CHECK-SAME: (%arg0: tensor<64x64xbf16, #[[DRAM_INTERLEAVED]]>, %arg1: tensor<64x64xbf16, #[[L1_INTERLEAVED]]>, %arg2: tensor<64x64xbf16, #[[L1_INTERLEAVED]]>) -> tensor<64x64xbf16, #[[DRAM_INTERLEAVED_2]]> {
  // CHECK: %[[ADD_OUT:.*]] = "ttnn.add"
  // CHECK-SAME: (tensor<64x64xbf16, #[[DRAM_INTERLEAVED]]>, tensor<64x64xbf16, #[[L1_INTERLEAVED]]>)
  // CHECK: %[[MULTIPLY_OUT:.*]] = "ttnn.multiply"
  // CHECK-SAME: (%[[ADD_OUT]]
  // CHECK: %[[TO_LAYOUT:.*]] = "ttnn.to_layout"(%[[MULTIPLY_OUT]])
  // CHECK-SAME: -> tensor<64x64xbf16, #[[DRAM_INTERLEAVED_2]]>
  // CHECK: return %[[TO_LAYOUT]] : tensor<64x64xbf16, #[[DRAM_INTERLEAVED_2]]>
}
