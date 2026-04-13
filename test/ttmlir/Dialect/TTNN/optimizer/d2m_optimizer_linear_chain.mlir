// REQUIRES: opmodel
// RUN: ttmlir-opt --ttcore-register-device --ttcore-mark-functions-as-forward --ttnn-d2m-fusing --ttnn-optimizer="memory-layout-analysis-enabled=true memreconfig-enabled=true memory-layout-analysis-policy=DFSharding" --mlir-print-local-scope -o %t %s
// RUN: FileCheck %s --input-file=%t

// Linear chain with D2M: matmul -> matmul -> d2m_subgraph -> matmul.
// D2MSubgraphOp is now in validForSharding, so it can be part of an L1 chain.
// The chain is not broken at D2M (no spill to_layout between second matmul and
// D2M). When chain resolution succeeds with sufficient L1, D2M and surrounding
// ops may get L1-sharded layouts.
#layout = #ttnn.ttnn_layout<(d0, d1) -> (d0, d1), <1x1>, memref<2x2x!ttcore.tile<32x32, bf16>, #ttnn.buffer_type<l1>>, <interleaved>>
module {
  func.func @simple_64x64(
      %arg0: tensor<64x64xbf16, #layout>,
      %arg1: tensor<64x64xbf16, #layout>,
      %arg2: tensor<64x64xbf16, #layout>,
      %arg3: tensor<64x64xbf16, #layout>,
      %arg4: tensor<64x64xbf16, #layout>) -> tensor<64x64xbf16, #layout> {
    %0 = "ttnn.matmul"(%arg0, %arg1) : (tensor<64x64xbf16, #layout>, tensor<64x64xbf16, #layout>) -> tensor<64x64xbf16, #layout>
    // CHECK: %[[MATMUL_0_OUT:.*]] = "ttnn.matmul"
    // CHECK-SAME: memref<1x2x!ttcore.tile<32x32, bf16>, #ttnn.buffer_type<l1>>, <block_sharded>>>
    %1 = "ttnn.matmul"(%0, %arg2) : (tensor<64x64xbf16, #layout>, tensor<64x64xbf16, #layout>) -> tensor<64x64xbf16, #layout>
    // CHECK: %[[MATMUL_1_OUT:.*]] = "ttnn.matmul"
    // CHECK-SAME: memref<1x1x!ttcore.tile<32x32, bf16>, #ttnn.buffer_type<l1>>, <block_sharded>>>
    %2 = "ttnn.add"(%1, %arg3) <{dtype = #ttcore.supportedDataTypes<bf16>}> : (tensor<64x64xbf16, #layout>, tensor<64x64xbf16, #layout>) -> tensor<64x64xbf16, #layout>
    %3 = "ttnn.multiply"(%2, %arg0) <{dtype = #ttcore.supportedDataTypes<bf16>}> : (tensor<64x64xbf16, #layout>, tensor<64x64xbf16, #layout>) -> tensor<64x64xbf16, #layout>
    // CHECK: %[[EMPTY:.*]] = "ttnn.empty"
    // CHECK-SAME: memref<1x2x!ttcore.tile<32x32, bf16>, #ttnn.buffer_type<l1>>, <block_sharded>>>
    // CHECK: %[[D2M_SUBGRAPH_OUT:.*]] = ttnn.d2m_subgraph @d2m_subgraph_0
    // CHECK: ins(%[[MATMUL_1_OUT]], %arg3, %arg0 :
    // CHECK-SAME: tensor<64x64xbf16, #ttnn.ttnn_layout<{{.*}}memref<1x1x!ttcore.tile<32x32, bf16>, #ttnn.buffer_type<l1>>, <block_sharded>>>,
    // CHECK-SAME: tensor<64x64xbf16, #ttnn.ttnn_layout<{{.*}}memref<2x2x!ttcore.tile<32x32, bf16>, #ttnn.buffer_type<l1>>, <interleaved>>>,
    // CHECK-SAME: tensor<64x64xbf16, #ttnn.ttnn_layout<{{.*}}memref<2x2x!ttcore.tile<32x32, bf16>, #ttnn.buffer_type<l1>>, <interleaved>>>)
    // CHECK: outs(%[[EMPTY]] : tensor<64x64xbf16, #ttnn.ttnn_layout<{{.*}}memref<1x2x!ttcore.tile<32x32, bf16>, #ttnn.buffer_type<l1>>, <block_sharded>>>)
    %4 = "ttnn.matmul"(%3, %arg4) : (tensor<64x64xbf16, #layout>, tensor<64x64xbf16, #layout>) -> tensor<64x64xbf16, #layout>
    // CHECK: %[[MATMUL_2_OUT:.*]] = "ttnn.matmul"(%[[D2M_SUBGRAPH_OUT]], %arg4)
    // CHECK-SAME: memref<1x1x!ttcore.tile<32x32, bf16>, #ttnn.buffer_type<l1>>, <block_sharded>>>
    // CHECK: %[[MATMUL_2_SPILLED:.*]] = "ttnn.to_layout"(%[[MATMUL_2_OUT]])
    // CHECK-SAME: memref<1x1x!ttcore.tile<32x32, bf16>, #ttnn.buffer_type<dram>>, <interleaved>>>
    return %4 : tensor<64x64xbf16, #layout>
    // CHECK: return %[[MATMUL_2_SPILLED]]
  }

  // CHECK: func.func private @d2m_subgraph_0
  // CHECK-SAME: (%arg0: tensor<64x64xbf16, #ttnn.ttnn_layout<{{.*}}memref<1x1x!ttcore.tile<32x32, bf16>, #ttnn.buffer_type<l1>>, <block_sharded>>>,
  // CHECK-SAME: %arg1: tensor<64x64xbf16, #ttnn.ttnn_layout<{{.*}}memref<2x2x!ttcore.tile<32x32, bf16>, #ttnn.buffer_type<l1>>, <interleaved>>>,
  // CHECK-SAME: %arg2: tensor<64x64xbf16, #ttnn.ttnn_layout<{{.*}}memref<2x2x!ttcore.tile<32x32, bf16>, #ttnn.buffer_type<l1>>, <interleaved>>>)
  // CHECK-SAME: -> tensor<64x64xbf16, #ttnn.ttnn_layout<{{.*}}memref<1x2x!ttcore.tile<32x32, bf16>, #ttnn.buffer_type<l1>>, <block_sharded>>> {
  // CHECK: %[[ADD_OUT:.*]] = "ttnn.add"
  // CHECK-SAME: (tensor<64x64xbf16, #ttnn.ttnn_layout<{{.*}}#ttnn.buffer_type<l1>>, <block_sharded>>>, tensor<64x64xbf16, #ttnn.ttnn_layout<{{.*}}#ttnn.buffer_type<l1>>, <interleaved>>>)
  // CHECK: %[[MULTIPLY_OUT:.*]] = "ttnn.multiply"
  // CHECK-SAME: (%[[ADD_OUT]]
  // CHECK-SAME: memref<1x2x!ttcore.tile<32x32, bf16>, #ttnn.buffer_type<l1>>, <block_sharded>>>
  // CHECK: return %[[MULTIPLY_OUT]]
}
