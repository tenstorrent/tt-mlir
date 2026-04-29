// REQUIRES: opmodel
// RUN: ttmlir-opt --ttcore-register-device --ttcore-mark-functions-as-forward --ttnn-d2m-fusing --ttnn-optimizer="memory-layout-analysis-enabled=true memreconfig-enabled=true memory-layout-analysis-policy=DFSharding" --mlir-print-local-scope -o %t %s
// RUN: FileCheck %s --input-file=%t

// This test shows that the optimizer can handle fork-join patterns when a d2m_subgraph is present.
// Note 1: The behavior of the optimizer is to spill to DRAM with a fork-join pattern.
// That's why we check for the to_layout insertion (even when a d2m_subgraph is present).
// Note 2: The add and multiply ops will be fused into the d2m_subgraph when ttnn-d2m-fusing is enabled.
// Uses --mlir-print-local-scope and inline layout checks (no aliasing) for
// easier debugging and to avoid fragility from layout reordering.
//             matmul
//          /        \
//         /          \
//   D2M_Subgraph      \
//         \          /
//          \       /
//            matmul

#layout = #ttnn.ttnn_layout<(d0, d1) -> (d0, d1), <1x1>, memref<2x2x!ttcore.tile<32x32, bf16>, #ttnn.buffer_type<l1>>, <interleaved>>
module {
  func.func @fork_join(%arg0: tensor<64x64xbf16, #layout>,%arg1: tensor<64x64xbf16, #layout>,%arg2: tensor<64x64xbf16, #layout>) -> (tensor<64x64xbf16, #layout>) {
    %0 = "ttnn.matmul"(%arg0, %arg1) : (tensor<64x64xbf16, #layout>, tensor<64x64xbf16, #layout>) -> tensor<64x64xbf16, #layout>
    // CHECK: %[[MATMUL_0_OUT:.*]] = "ttnn.matmul"
    // CHECK-SAME: memref<1x1x!ttcore.tile<32x32, bf16>, #ttnn.buffer_type<l1>>, <block_sharded>{{(, core_ranges = <\[[^]]*\]>)?}}>>
    // CHECK: %[[MATMUL_0_SPILLED:.*]] = "ttnn.to_layout"(%[[MATMUL_0_OUT]])
    // CHECK-SAME: memref<2x2x!ttcore.tile<32x32, bf16>, #ttnn.buffer_type<dram>>, <interleaved>>>
    %1 = "ttnn.add"(%0, %arg2) <{dtype = #ttcore.supportedDataTypes<bf16>}> : (tensor<64x64xbf16, #layout>, tensor<64x64xbf16, #layout>) -> tensor<64x64xbf16, #layout>
    %2 = "ttnn.multiply"(%1, %arg0) <{dtype = #ttcore.supportedDataTypes<bf16>}> : (tensor<64x64xbf16, #layout>, tensor<64x64xbf16, #layout>) -> tensor<64x64xbf16, #layout>
    // CHECK: %[[D2M_OUTPUT_BUFFER:.*]] = "ttnn.empty"
    // CHECK-SAME: memref<1x1x!ttcore.tile<32x32, bf16>, #ttnn.buffer_type<l1>>, <block_sharded>{{(, core_ranges = <\[[^]]*\]>)?}}>>
    // CHECK: %[[D2M_SUBGRAPH_OUT:.*]] = ttnn.d2m_subgraph @d2m_subgraph_0
    // CHECK: ins(%[[MATMUL_0_SPILLED]], %arg2, %arg0 :
    // CHECK-SAME: tensor<64x64xbf16, #ttnn.ttnn_layout<{{.*}}memref<2x2x!ttcore.tile<32x32, bf16>, #ttnn.buffer_type<dram>>, <interleaved>>>,
    // CHECK-SAME: tensor<64x64xbf16, #ttnn.ttnn_layout<{{.*}}memref<2x2x!ttcore.tile<32x32, bf16>, #ttnn.buffer_type<l1>>, <interleaved>>>,
    // CHECK-SAME: tensor<64x64xbf16, #ttnn.ttnn_layout<{{.*}}memref<2x2x!ttcore.tile<32x32, bf16>, #ttnn.buffer_type<l1>>, <interleaved>>>)
    // CHECK: outs(%[[D2M_OUTPUT_BUFFER]] : tensor<64x64xbf16, #ttnn.ttnn_layout<{{.*}}memref<1x1x!ttcore.tile<32x32, bf16>, #ttnn.buffer_type<l1>>, <block_sharded>{{(, core_ranges = <\[[^]]*\]>)?}}>>)
    // CHECK: %[[D2M_TO_DRAM:.*]] = "ttnn.to_layout"(%[[D2M_SUBGRAPH_OUT]])
    // CHECK-SAME: memref<2x2x!ttcore.tile<32x32, bf16>, #ttnn.buffer_type<dram>>, <interleaved>>>
    %3 = "ttnn.matmul"(%0, %2) : (tensor<64x64xbf16, #layout>, tensor<64x64xbf16, #layout>) -> tensor<64x64xbf16, #layout>
    // CHECK: %[[MATMUL_1_OUT:.*]] = "ttnn.matmul"(%[[MATMUL_0_SPILLED]], %[[D2M_TO_DRAM]]
    // CHECK-SAME: memref<1x1x!ttcore.tile<32x32, bf16>, #ttnn.buffer_type<l1>>, <block_sharded>{{(, core_ranges = <\[[^]]*\]>)?}}>>
    // CHECK: %[[MATMUL_1_SPILLED:.*]] = "ttnn.to_layout"(%[[MATMUL_1_OUT]])
    // CHECK-SAME: memref<2x2x!ttcore.tile<32x32, bf16>, #ttnn.buffer_type<dram>>, <interleaved>>>
    return %3 : tensor<64x64xbf16, #layout>
    // CHECK: return %[[MATMUL_1_SPILLED]]
  }

  // CHECK: func.func private @d2m_subgraph_0
  // CHECK-SAME: (%arg0: tensor<64x64xbf16, #ttnn.ttnn_layout<{{.*}}memref<2x2x!ttcore.tile<32x32, bf16>, #ttnn.buffer_type<dram>>, <interleaved>>>,
  // CHECK-SAME: %arg1: tensor<64x64xbf16, #ttnn.ttnn_layout<{{.*}}memref<2x2x!ttcore.tile<32x32, bf16>, #ttnn.buffer_type<l1>>, <interleaved>>>,
  // CHECK-SAME: %arg2: tensor<64x64xbf16, #ttnn.ttnn_layout<{{.*}}memref<2x2x!ttcore.tile<32x32, bf16>, #ttnn.buffer_type<l1>>, <interleaved>>>)
  // CHECK-SAME: -> tensor<64x64xbf16, #ttnn.ttnn_layout<{{.*}}memref<1x1x!ttcore.tile<32x32, bf16>, #ttnn.buffer_type<l1>>, <block_sharded>{{(, core_ranges = <\[[^]]*\]>)?}}>> {
  // CHECK: %[[ADD_OUT:.*]] = "ttnn.add"
  // CHECK-SAME: (tensor<64x64xbf16, #ttnn.ttnn_layout<{{.*}}#ttnn.buffer_type<dram>>, <interleaved>>>, tensor<64x64xbf16, #ttnn.ttnn_layout<{{.*}}#ttnn.buffer_type<l1>>, <interleaved>>>)
  // CHECK: %[[MULTIPLY_OUT:.*]] = "ttnn.multiply"
  // CHECK-SAME: (%[[ADD_OUT]]
  // CHECK: %[[TO_LAYOUT:.*]] = "ttnn.to_layout"(%[[MULTIPLY_OUT]])
  // CHECK-SAME: memref<1x1x!ttcore.tile<32x32, bf16>, #ttnn.buffer_type<l1>>, <block_sharded>{{(, core_ranges = <\[[^]]*\]>)?}}>>
  // CHECK: return %[[TO_LAYOUT]]
}
