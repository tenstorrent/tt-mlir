// REQUIRES: opmodel
// RUN: ttmlir-opt --ttcore-register-device --ttcore-mark-functions-as-forward --ttnn-d2m-fusing --ttnn-optimizer="memory-layout-analysis-enabled=true memreconfig-enabled=true memory-layout-analysis-policy=DFSharding" -o %t %s
// RUN: FileCheck %s --input-file=%t

// Linear chain with D2M kept in L1 when a d2m_subgraph is present.
// matmul -> matmul -> d2m_subgraph -> matmul.
// Optimizer should keep them in one L1 chain.
// Since not falling back to DRAM is important for this test, more explicit checks are performed.

#layout = #ttnn.ttnn_layout<(d0, d1) -> (d0, d1), <1x1>, memref<2x2x!ttcore.tile<32x32, bf16>, #ttnn.buffer_type<l1>>, <interleaved>>
// CHECK: #[[L1_INTERLEAVED:.*]] = #ttnn.ttnn_layout<(d0, d1) -> (d0, d1), <1x1>, memref<2x2x!ttcore.tile<32x32, bf16>, #l1>, <interleaved>>
// CHECK: #[[DRAM_INTERLEAVED:.*]] = #ttnn.ttnn_layout<(d0, d1) -> (d0, d1), <1x1>, memref<1x1x!ttcore.tile<32x32, bf16>, #dram>, <interleaved>>
// CHECK: #[[L1_BLOCK_SHARDED_2x1:.*]] = #ttnn.ttnn_layout<(d0, d1) -> (d0, d1), <2x1, (d0, d1) -> (0, d0, d1)>, memref<1x2x!ttcore.tile<32x32, bf16>, #l1>, <block_sharded>>
// CHECK: #[[L1_BLOCK_SHARDED_2x2:.*]] = #ttnn.ttnn_layout<(d0, d1) -> (d0, d1), <2x2, (d0, d1) -> (0, d0, d1)>, memref<1x1x!ttcore.tile<32x32, bf16>, #l1>, <block_sharded>>

module {
  func.func @simple_64x64(
      %arg0: tensor<64x64xbf16, #layout>,
      %arg1: tensor<64x64xbf16, #layout>,
      %arg2: tensor<64x64xbf16, #layout>,
      %arg3: tensor<64x64xbf16, #layout>,
      %arg4: tensor<64x64xbf16, #layout>) -> tensor<64x64xbf16, #layout> {
    %0 = "ttnn.matmul"(%arg0, %arg1) : (tensor<64x64xbf16, #layout>, tensor<64x64xbf16, #layout>) -> tensor<64x64xbf16, #layout>
    // CHECK: %[[MATMUL_0_OUT:.*]] = "ttnn.matmul"
    // CHECK-SAME: -> tensor<64x64xbf16, #[[L1_BLOCK_SHARDED_2x1]]>
    %1 = "ttnn.matmul"(%0, %arg2) : (tensor<64x64xbf16, #layout>, tensor<64x64xbf16, #layout>) -> tensor<64x64xbf16, #layout>
    // CHECK: %[[MATMUL_1_OUT:.*]] = "ttnn.matmul"
    // CHECK-SAME: -> tensor<64x64xbf16, #[[L1_BLOCK_SHARDED_2x2]]>
    %2 = "ttnn.add"(%1, %arg3) <{dtype = #ttcore.supportedDataTypes<bf16>}> : (tensor<64x64xbf16, #layout>, tensor<64x64xbf16, #layout>) -> tensor<64x64xbf16, #layout>
    %3 = "ttnn.multiply"(%2, %arg0) <{dtype = #ttcore.supportedDataTypes<bf16>}> : (tensor<64x64xbf16, #layout>, tensor<64x64xbf16, #layout>) -> tensor<64x64xbf16, #layout>
    // CHECK: %[[D2M_OUTPUT_BUFFER:.*]] = "ttnn.empty"
    // CHECK-SAME: -> tensor<64x64xbf16, #[[L1_BLOCK_SHARDED_2x1]]>
    // CHECK: %[[D2M_SUBGRAPH_OUT:.*]] = ttnn.d2m_subgraph @d2m_subgraph_0
    // CHECK: ins(%[[MATMUL_1_OUT]], %arg3, %arg0 : tensor<64x64xbf16, #[[L1_BLOCK_SHARDED_2x2]]>, tensor<64x64xbf16, #[[L1_INTERLEAVED]]>, tensor<64x64xbf16, #[[L1_INTERLEAVED]]>)
    // CHECK: outs(%[[D2M_OUTPUT_BUFFER]] : tensor<64x64xbf16, #[[L1_BLOCK_SHARDED_2x1]]>) : tensor<64x64xbf16, #[[L1_BLOCK_SHARDED_2x1]]>
    %4 = "ttnn.matmul"(%3, %arg4) : (tensor<64x64xbf16, #layout>, tensor<64x64xbf16, #layout>) -> tensor<64x64xbf16, #layout>
    // CHECK: %[[MATMUL_2_OUT:.*]] = "ttnn.matmul"
    // CHECK-SAME: -> tensor<64x64xbf16, #[[L1_BLOCK_SHARDED_2x2]]>

    // The optimizer by default spills the output of the last op to DRAM:
    // CHECK: %[[TO_LAYOUT_OUT:.*]] = "ttnn.to_layout"
    // CHECK-SAME: -> tensor<64x64xbf16, #[[DRAM_INTERLEAVED]]>
    return %4 : tensor<64x64xbf16, #layout>
    // CHECK: return %[[TO_LAYOUT_OUT]] : tensor<64x64xbf16, #[[DRAM_INTERLEAVED]]>
  }

  // Now checking for the d2m_subgraph_0 function:
  // CHECK: func.func private @d2m_subgraph_0
  // CHECK-SAME: (%arg0: tensor<64x64xbf16, #[[L1_BLOCK_SHARDED_2x2]]>, %arg1: tensor<64x64xbf16, #[[L1_INTERLEAVED]]>, %arg2: tensor<64x64xbf16, #[[L1_INTERLEAVED]]>) -> tensor<64x64xbf16, #[[L1_BLOCK_SHARDED_2x1]]> {
  // CHECK: %[[ADD_OUT:.*]] = "ttnn.add"
  // CHECK-SAME: (tensor<64x64xbf16, #[[L1_BLOCK_SHARDED_2x2]]>, tensor<64x64xbf16, #[[L1_INTERLEAVED]]>) -> tensor<64x64xbf16, #[[L1_BLOCK_SHARDED_2x1]]>
  // CHECK: %[[MULTIPLY_OUT:.*]] = "ttnn.multiply"
  // CHECK-SAME: (tensor<64x64xbf16, #[[L1_BLOCK_SHARDED_2x1]]>, tensor<64x64xbf16, #[[L1_INTERLEAVED]]>) -> tensor<64x64xbf16, #[[L1_BLOCK_SHARDED_2x1]]>
  // CHECK: return %[[MULTIPLY_OUT]] : tensor<64x64xbf16, #[[L1_BLOCK_SHARDED_2x1]]>
}
