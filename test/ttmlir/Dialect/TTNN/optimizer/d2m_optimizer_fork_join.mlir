// REQUIRES: opmodel
// RUN: ttmlir-opt --ttcore-register-device --ttcore-mark-functions-as-forward --ttnn-d2m-fusing --ttnn-optimizer="memory-layout-analysis-enabled=true memreconfig-enabled=true memory-layout-analysis-policy=DFSharding" -o %t %s
// RUN: FileCheck %s --input-file=%t

// This test shows that the optimizer can handle fork-join patterns when a d2m_subgraph is present.
// Note 1: rn, the behavior of the optimizer is to spill to DRAM with a fork-join pattern.
// That's why we check for the to_layout insertion (even when a d2m_subgraph is present).
// Note 2: The add and multiply ops will be fused into the d2m_subgraph when ttnn-d2m-fusing is enabled.
//             matmul
//          /        \
//         /          \
//   D2M_Subgraph      \
//         \          /
//          \       /
//            matmul

#layout = #ttnn.ttnn_layout<(d0, d1) -> (d0, d1), <1x1>, memref<2x2x!ttcore.tile<32x32, bf16>, #ttnn.buffer_type<l1>>, <interleaved>>
// CHECK: #[[L1_INTERLEAVED:.*]] = #ttnn.ttnn_layout<(d0, d1) -> (d0, d1), <1x1>, memref<2x2x!ttcore.tile<32x32, bf16>, #l1>, <interleaved>>
// CHECK: #[[DRAM_INTERLEAVED:.*]] = #ttnn.ttnn_layout<(d0, d1) -> (d0, d1), <1x1>, memref<1x1x!ttcore.tile<32x32, bf16>, #dram>, <interleaved>>
// CHECK: #[[L1_BLOCK_SHARDED:.*]] = #ttnn.ttnn_layout<(d0, d1) -> (d0, d1), <2x2, (d0, d1) -> (0, d0, d1)>, memref<1x1x!ttcore.tile<32x32, bf16>, #l1>, <block_sharded>>
// CHECK: #[[DRAM_INTERLEAVED_2:.*]] = #ttnn.ttnn_layout<(d0, d1) -> (d0, d1), <1x1>, memref<2x2x!ttcore.tile<32x32, bf16>, #dram>, <interleaved>>

module {
  func.func @fork_join(%arg0: tensor<64x64xbf16, #layout>,%arg1: tensor<64x64xbf16, #layout>,%arg2: tensor<64x64xbf16, #layout>) -> (tensor<64x64xbf16, #layout>) {
    %0 = "ttnn.matmul"(%arg0, %arg1) : (tensor<64x64xbf16, #layout>, tensor<64x64xbf16, #layout>) -> tensor<64x64xbf16, #layout>
    // CHECK: %[[MATMUL_0_OUT:.*]] = "ttnn.matmul"
    // CHECK-SAME: -> tensor<64x64xbf16, #[[L1_BLOCK_SHARDED]]>
    // CHECK: %[[MATMUL_0_SPILLED:.*]] = "ttnn.to_layout"(%[[MATMUL_0_OUT]])
    // CHECK-SAME: -> tensor<64x64xbf16, #[[DRAM_INTERLEAVED]]>
    %1 = "ttnn.add"(%0, %arg2) <{dtype = #ttcore.supportedDataTypes<bf16>}> : (tensor<64x64xbf16, #layout>, tensor<64x64xbf16, #layout>) -> tensor<64x64xbf16, #layout>
    %2 = "ttnn.multiply"(%1, %arg0) <{dtype = #ttcore.supportedDataTypes<bf16>}> : (tensor<64x64xbf16, #layout>, tensor<64x64xbf16, #layout>) -> tensor<64x64xbf16, #layout>
    // CHECK: %[[D2M_OUTPUT_BUFFER:.*]] = "ttnn.empty"
    // CHECK-SAME: -> tensor<64x64xbf16, #[[DRAM_INTERLEAVED_2]]>
    // CHECK: %[[D2M_SUBGRAPH_OUT:.*]] = ttnn.d2m_subgraph @d2m_subgraph_0
    // CHECK: ins(%[[MATMUL_0_SPILLED]], %arg2, %arg0 : tensor<64x64xbf16, #[[DRAM_INTERLEAVED]]>, tensor<64x64xbf16, #[[L1_INTERLEAVED]]>, tensor<64x64xbf16, #[[L1_INTERLEAVED]]>)
    // CHECK: outs(%[[D2M_OUTPUT_BUFFER]] : tensor<64x64xbf16, #[[DRAM_INTERLEAVED_2]]>) : tensor<64x64xbf16, #[[DRAM_INTERLEAVED_2]]>
    %3 = "ttnn.matmul"(%0, %2) : (tensor<64x64xbf16, #layout>, tensor<64x64xbf16, #layout>) -> tensor<64x64xbf16, #layout>
    // CHECK: %[[MATMUL_1_OUT:.*]] = "ttnn.matmul"
    // CHECK-SAME: -> tensor<64x64xbf16, #[[L1_BLOCK_SHARDED]]>
    // CHECK: %[[MATMUL_1_SPILLED:.*]] = "ttnn.to_layout"(%[[MATMUL_1_OUT]])
    // CHECK-SAME: -> tensor<64x64xbf16, #[[DRAM_INTERLEAVED]]>
    return %3 : tensor<64x64xbf16, #layout>
    // CHECK: return %[[MATMUL_1_SPILLED]] : tensor<64x64xbf16, #[[DRAM_INTERLEAVED]]>
  }
  // D2M callee: internal ops unchanged; trailing to_layout converts to chosen layout.
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
