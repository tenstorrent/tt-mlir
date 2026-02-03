// REQUIRES: opmodel
// RUN: ttmlir-opt --ttcore-register-device --ttcore-mark-functions-as-forward --ttnn-d2m-fusing --ttnn-optimizer="memory-layout-analysis-enabled=true memreconfig-enabled=true memory-layout-analysis-policy=DFSharding" -o %t %s
// RUN: FileCheck %s --input-file=%t

#layout = #ttnn.ttnn_layout<(d0, d1) -> (d0, d1), <1x1>, memref<2x2x!ttcore.tile<32x32, bf16>, #ttnn.buffer_type<l1>>, <interleaved>>

// CHECK: #[[L1_INTERLEAVED:.*]] = #ttnn.ttnn_layout<(d0, d1) -> (d0, d1), <1x1>, memref<2x2x!ttcore.tile<32x32, bf16>, #l1>, <interleaved>>
// CHECK: #[[DRAM_INTERLEAVED:.*]] = #ttnn.ttnn_layout<(d0, d1) -> (d0, d1), <1x1>, memref<1x1x!ttcore.tile<32x32, bf16>, #dram>, <interleaved>>
// CHECK: #[[BLOCK_SHARDED_2x1:.*]] = #ttnn.ttnn_layout<(d0, d1) -> (d0, d1), <2x1, (d0, d1) -> (0, d0, d1)>, memref<1x2x!ttcore.tile<32x32, bf16>, #l1>, <block_sharded>>
// CHECK: #[[BLOCK_SHARDED_2x2:.*]] = #ttnn.ttnn_layout<(d0, d1) -> (d0, d1), <2x2, (d0, d1) -> (0, d0, d1)>, memref<1x1x!ttcore.tile<32x32, bf16>, #l1>, <block_sharded>>

module {
  func.func @simple_64x64(
      %arg0: tensor<64x64xbf16, #layout>,
      %arg1: tensor<64x64xbf16, #layout>,
      %arg2: tensor<64x64xbf16, #layout>,
      %arg3: tensor<64x64xbf16, #layout>,
      %arg4: tensor<64x64xbf16, #layout>,
      %arg5: tensor<64x64xbf16, #layout>) -> tensor<64x64xbf16, #layout> {
    %0 = "ttnn.matmul"(%arg0, %arg1) : (tensor<64x64xbf16, #layout>, tensor<64x64xbf16, #layout>) -> tensor<64x64xbf16, #layout>
    %1 = "ttnn.matmul"(%0, %arg2) : (tensor<64x64xbf16, #layout>, tensor<64x64xbf16, #layout>) -> tensor<64x64xbf16, #layout>
    // CHECK: %[[DEVICE:.*]] = "ttnn.get_device"()
    // CHECK: %[[MM0:.*]] = "ttnn.matmul"(%arg0, %arg1){{.*}} -> tensor<64x64xbf16, #[[BLOCK_SHARDED_2x1]]>
    // CHECK: %[[MM1:.*]] = "ttnn.matmul"(%[[MM0]], %arg2){{.*}} -> tensor<64x64xbf16, #[[BLOCK_SHARDED_2x2]]>
    // CHECK: %[[SPILLED:.*]] = "ttnn.to_layout"(%[[MM1]]){{.*}} -> tensor<64x64xbf16, #[[DRAM_INTERLEAVED]]>
    %2 = "ttnn.add"(%1, %arg3) <{dtype = #ttcore.supportedDataTypes<bf16>}> : (tensor<64x64xbf16, #layout>, tensor<64x64xbf16, #layout>) -> tensor<64x64xbf16, #layout>
    %3 = "ttnn.multiply"(%2, %arg0) <{dtype = #ttcore.supportedDataTypes<bf16>}> : (tensor<64x64xbf16, #layout>, tensor<64x64xbf16, #layout>) -> tensor<64x64xbf16, #layout>
    // CHECK: %[[OUTPUT_BUF:.*]] = "ttnn.empty"(%[[DEVICE]]){{.*}} -> tensor<64x64xbf16, #[[BLOCK_SHARDED_2x2]]>
    // CHECK: %[[DISPATCH_OUT:.*]] = ttnn.d2m_subgraph @d2m_subgraph_0
    // CHECK: ins(%[[SPILLED]], %arg3, %arg0
    // CHECK: outs(%[[OUTPUT_BUF]]
    // CHECK: %[[DISPATCH_SPILLED:.*]] = "ttnn.to_layout"(%[[DISPATCH_OUT]]){{.*}} -> tensor<64x64xbf16, #[[DRAM_INTERLEAVED]]>
    %4 = "ttnn.matmul"(%3, %arg4) : (tensor<64x64xbf16, #layout>, tensor<64x64xbf16, #layout>) -> tensor<64x64xbf16, #layout>
    %5 = "ttnn.matmul"(%4, %arg5) : (tensor<64x64xbf16, #layout>, tensor<64x64xbf16, #layout>) -> tensor<64x64xbf16, #layout>
    // CHECK: %[[MM2:.*]] = "ttnn.matmul"(%[[DISPATCH_SPILLED]], %arg4)
    // CHECK: %[[MM3:.*]] = "ttnn.matmul"(%[[MM2]], %arg5)
    // CHECK: %[[FINAL_SPILLED:.*]] = "ttnn.to_layout"(%[[MM3]]){{.*}} -> tensor<64x64xbf16, #[[DRAM_INTERLEAVED]]>
    return %5 : tensor<64x64xbf16, #layout>
    // CHECK: return %[[FINAL_SPILLED]] : tensor<64x64xbf16, #[[DRAM_INTERLEAVED]]>
  }

  // CHECK: func.func private @d2m_subgraph_0(%arg0: tensor<64x64xbf16, #[[DRAM_INTERLEAVED]]>, %arg1: tensor<64x64xbf16, #[[L1_INTERLEAVED]]>, %arg2: tensor<64x64xbf16, #[[L1_INTERLEAVED]]>) -> tensor<64x64xbf16, #[[BLOCK_SHARDED_2x2]]> {
  // CHECK: "ttnn.add"(
  // CHECK: "ttnn.multiply"(
}
