// REQUIRES: opmodel
// RUN: ttmlir-opt --ttcore-register-device --ttcore-mark-functions-as-forward --ttnn-d2m-fusing --ttnn-optimizer="memory-layout-analysis-enabled=true memreconfig-enabled=true memory-layout-analysis-policy=DFSharding" --mlir-print-local-scope -o %t %s
// RUN: FileCheck %s --input-file=%t

// Fork-join with D2M: matmul -> fork -> { d2m_subgraph(add, multiply) } -> matmul.
// After the optimizer, the first matmul output is consumed by both the d2m_subgraph
// and the final matmul (fork point). D2M is spilled because the fork point must be
// in DRAM for the second matmul.
#layout = #ttnn.ttnn_layout<(d0, d1) -> (d0, d1), <1x1>, memref<2x2x!ttcore.tile<32x32, bf16>, #ttnn.buffer_type<l1>>, <interleaved>>
module {
  func.func @fork_join(%arg0: tensor<64x64xbf16, #layout>,
                       %arg1: tensor<64x64xbf16, #layout>,
                       %arg2: tensor<64x64xbf16, #layout>) -> (tensor<64x64xbf16, #layout>) {
    %0 = "ttnn.matmul"(%arg0, %arg1) : (tensor<64x64xbf16, #layout>, tensor<64x64xbf16, #layout>) -> tensor<64x64xbf16, #layout>
    // CHECK: %[[MATMUL_0_OUT:.*]] = "ttnn.matmul"
    // CHECK-SAME: memref<1x1x!ttcore.tile<32x32, bf16>, #ttnn.buffer_type<l1>>, <block_sharded>>>
    // CHECK: %[[MATMUL_0_SPILLED:.*]] = "ttnn.to_layout"(%[[MATMUL_0_OUT]])
    // CHECK-SAME: memref<1x1x!ttcore.tile<32x32, bf16>, #ttnn.buffer_type<dram>>, <interleaved>>>
    %1 = "ttnn.add"(%0, %arg2) <{dtype = #ttcore.supportedDataTypes<bf16>}> : (tensor<64x64xbf16, #layout>, tensor<64x64xbf16, #layout>) -> tensor<64x64xbf16, #layout>
    %2 = "ttnn.multiply"(%1, %arg0) <{dtype = #ttcore.supportedDataTypes<bf16>}> : (tensor<64x64xbf16, #layout>, tensor<64x64xbf16, #layout>) -> tensor<64x64xbf16, #layout>
    // CHECK: %[[D2M_SUBGRAPH_OUT:.*]] = ttnn.d2m_subgraph @d2m_subgraph_0
    // CHECK: ins(%[[MATMUL_0_SPILLED]], %arg2, %arg0 :
    // CHECK-SAME: tensor<64x64xbf16, #ttnn.ttnn_layout<{{.*}}memref<1x1x!ttcore.tile<32x32, bf16>, #ttnn.buffer_type<dram>>, <interleaved>>>,
    // CHECK-SAME: tensor<64x64xbf16, #ttnn.ttnn_layout<{{.*}}memref<2x2x!ttcore.tile<32x32, bf16>, #ttnn.buffer_type<l1>>, <interleaved>>>,
    // CHECK-SAME: tensor<64x64xbf16, #ttnn.ttnn_layout<{{.*}}memref<2x2x!ttcore.tile<32x32, bf16>, #ttnn.buffer_type<l1>>, <interleaved>>>)
    // CHECK: outs({{.*}} : tensor<64x64xbf16, #ttnn.ttnn_layout<{{.*}}memref<1x1x!ttcore.tile<32x32, bf16>, #ttnn.buffer_type<l1>>, <block_sharded>>>)
    %3 = "ttnn.matmul"(%0, %2) : (tensor<64x64xbf16, #layout>, tensor<64x64xbf16, #layout>) -> tensor<64x64xbf16, #layout>
    // CHECK: %[[D2M_SPILLED:.*]] = "ttnn.to_layout"(%[[D2M_SUBGRAPH_OUT]])
    // CHECK-SAME: memref<1x1x!ttcore.tile<32x32, bf16>, #ttnn.buffer_type<dram>>, <interleaved>>>
    // CHECK: %[[MATMUL_1_OUT:.*]] = "ttnn.matmul"(%[[MATMUL_0_SPILLED]], %[[D2M_SPILLED]])
    return %3 : tensor<64x64xbf16, #layout>
  }

  // CHECK: func.func private @d2m_subgraph_0
  // CHECK-SAME: (%arg0: tensor<64x64xbf16, #ttnn.ttnn_layout<{{.*}}#ttnn.buffer_type<dram>>, <interleaved>>>,
  // CHECK-SAME: %arg1: tensor<64x64xbf16, #ttnn.ttnn_layout<{{.*}}#ttnn.buffer_type<l1>>, <interleaved>>>,
  // CHECK-SAME: %arg2: tensor<64x64xbf16, #ttnn.ttnn_layout<{{.*}}#ttnn.buffer_type<l1>>, <interleaved>>>)
  // CHECK: %[[ADD_OUT:.*]] = "ttnn.add"
  // CHECK: %[[MULTIPLY_OUT:.*]] = "ttnn.multiply"
  // CHECK: return %[[MULTIPLY_OUT]]
}
