// REQUIRES: opmodel
// RUN: ttmlir-opt --ttcore-register-device --ttcore-mark-functions-as-forward --ttnn-d2m-fusing --ttnn-optimizer="memory-layout-analysis-enabled=true memreconfig-enabled=true memory-layout-analysis-policy=DFSharding" -o %t %s
// RUN: FileCheck %s --input-file=%t

#l1 = #ttnn.buffer_type<l1>
#layout = #ttnn.ttnn_layout<(d0, d1) -> (d0, d1), <1x1>, memref<2x8x!ttcore.tile<32x32, bf16>, #l1>, <interleaved>>
#layout_64x64 = #ttnn.ttnn_layout<(d0, d1) -> (d0, d1), <1x1>, memref<2x2x!ttcore.tile<32x32, bf16>, #l1>, <interleaved>>
#layout_256x64 = #ttnn.ttnn_layout<(d0, d1) -> (d0, d1), <1x1>, memref<8x2x!ttcore.tile<32x32, bf16>, #l1>, <interleaved>>

// CHECK: #[[DRAM_INTERLEAVED:.*]] = #ttnn.ttnn_layout<(d0, d1) -> (d0, d1), <1x1>, memref<1x1x!ttcore.tile<32x32, bf16>, #dram>, <interleaved>>
// CHECK: #[[BLOCK_SHARDED_64x256:.*]] = #ttnn.ttnn_layout<(d0, d1) -> (d0, d1), <2x8, (d0, d1) -> (0, d0, d1)>, memref<1x1x!ttcore.tile<32x32, bf16>, #l1>, <block_sharded>>

module {
  func.func @dispatch_chain_boundary(
      %arg0: tensor<64x128xbf16, #layout>,
      %arg1: tensor<128x256xbf16, #layout>,
      %arg2: tensor<256x64xbf16, #layout_256x64>,
      %arg3: tensor<64x256xbf16, #layout>) -> tensor<64x256xbf16, #layout> {
    %0 = "ttnn.matmul"(%arg0, %arg1) : (tensor<64x128xbf16, #layout>, tensor<128x256xbf16, #layout>) -> tensor<64x256xbf16, #layout>
    // CHECK: %[[DEVICE:.*]] = "ttnn.get_device"()
    // CHECK: %[[MATMUL_OUT:.*]] = "ttnn.matmul"(%arg0, %arg1){{.*}} -> tensor<64x256xbf16, #[[BLOCK_SHARDED_64x256]]>
    // CHECK: %[[SPILLED:.*]] = "ttnn.to_layout"(%[[MATMUL_OUT]]){{.*}} -> tensor<64x256xbf16, #[[DRAM_INTERLEAVED]]>
    %1 = "ttnn.exp"(%0) : (tensor<64x256xbf16, #layout>) -> tensor<64x256xbf16, #layout>
    %2 = "ttnn.neg"(%1) : (tensor<64x256xbf16, #layout>) -> tensor<64x256xbf16, #layout>
    %3 = "ttnn.abs"(%2) : (tensor<64x256xbf16, #layout>) -> tensor<64x256xbf16, #layout>
    // CHECK: %[[OUTPUT_BUF:.*]] = "ttnn.empty"(%[[DEVICE]]){{.*}} -> tensor<64x256xbf16, #[[BLOCK_SHARDED_64x256]]>
    // CHECK: %[[DISPATCH_OUT:.*]] = ttnn.d2m_subgraph @d2m_subgraph_0
    // CHECK: ins(%[[SPILLED]]
    // CHECK: outs(%[[OUTPUT_BUF]]
    // CHECK: %[[DISPATCH_SPILLED:.*]] = "ttnn.to_layout"(%[[DISPATCH_OUT]]){{.*}} -> tensor<64x256xbf16, #[[DRAM_INTERLEAVED]]>
    %4 = "ttnn.matmul"(%3, %arg2) <{transpose_a = false, transpose_b = false}> : (tensor<64x256xbf16, #layout>, tensor<256x64xbf16, #layout_256x64>) -> tensor<64x64xbf16, #layout_64x64>
    %5 = "ttnn.matmul"(%4, %arg3) <{transpose_a = false, transpose_b = false}> : (tensor<64x64xbf16, #layout_64x64>, tensor<64x256xbf16, #layout>) -> tensor<64x256xbf16, #layout>
    // CHECK: %[[MM1:.*]] = "ttnn.matmul"(%[[DISPATCH_SPILLED]], %arg2)
    // CHECK: %[[MM2:.*]] = "ttnn.matmul"(%[[MM1]], %arg3)
    // CHECK: %[[FINAL_SPILLED:.*]] = "ttnn.to_layout"(%[[MM2]]){{.*}} -> tensor<64x256xbf16, #[[DRAM_INTERLEAVED]]>
    return %5 : tensor<64x256xbf16, #layout>
    // CHECK: return %[[FINAL_SPILLED]] : tensor<64x256xbf16, #[[DRAM_INTERLEAVED]]>
  }

  // CHECK: func.func private @d2m_subgraph_0(%arg0: tensor<64x256xbf16, #[[DRAM_INTERLEAVED]]>) -> tensor<64x256xbf16, #[[BLOCK_SHARDED_64x256]]> {
  // CHECK: %0 = "ttnn.exp"(%arg0)
  // CHECK: %1 = "ttnn.neg"(%0)
  // CHECK: %2 = "ttnn.abs"(%1)
  // CHECK: return %2
}
