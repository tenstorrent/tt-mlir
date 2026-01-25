// REQUIRES: opmodel
// RUN: ttmlir-opt --ttcore-register-device --ttcore-mark-functions-as-forward --ttnn-d2m-fusing --ttnn-optimizer="memory-layout-analysis-enabled=true memreconfig-enabled=true memory-layout-analysis-policy=DFSharding" -o %t %s
// RUN: FileCheck %s --input-file=%t

// ttmlir-opt --debug --mlir-print-ir-after-all --ttcore-register-device --ttcore-mark-functions-as-forward --ttnn-d2m-fusing --ttnn-optimizer="memory-layout-analysis-enabled=true memreconfig-enabled=true memory-layout-analysis-policy=DFSharding" test/ttmlir/Dialect/TTNN/optimizer/d2m.mlir 2>&1 | tee

#l1 = #ttnn.buffer_type<l1>
#layout = #ttnn.ttnn_layout<(d0, d1) -> (d0, d1), <1x1>, memref<2x8x!ttcore.tile<32x32, bf16>, #l1>, <interleaved>>
#layout_64x64 = #ttnn.ttnn_layout<(d0, d1) -> (d0, d1), <1x1>, memref<2x2x!ttcore.tile<32x32, bf16>, #l1>, <interleaved>>
#layout_256x64 = #ttnn.ttnn_layout<(d0, d1) -> (d0, d1), <1x1>, memref<8x2x!ttcore.tile<32x32, bf16>, #l1>, <interleaved>>

module {
  // Test: dispatch_d2m with producer and consumer chains
  // Chain 1: matmul1 (producer) -> dispatch_d2m (boundary)
  // Chain 2: matmul2 -> matmul3 (consumers after dispatch_d2m form a multi-op chain)
  // CHECK-LABEL: func.func @dispatch_with_matmul_producer
  func.func @dispatch_with_matmul_producer(
      %arg0: tensor<64x128xbf16, #layout>,
      %arg1: tensor<128x256xbf16, #layout>,
      %arg2: tensor<256x64xbf16, #layout_256x64>,
      %arg3: tensor<64x256xbf16, #layout>) -> tensor<64x256xbf16, #layout> {

    %0 = "ttnn.matmul"(%arg0, %arg1) : (tensor<64x128xbf16, #layout>, tensor<128x256xbf16, #layout>) -> tensor<64x256xbf16, #layout>
    // exp, neg, abs will be fused into dispatch_d2m
    %1 = "ttnn.exp"(%0) : (tensor<64x256xbf16, #layout>) -> tensor<64x256xbf16, #layout>
    %2 = "ttnn.neg"(%1) : (tensor<64x256xbf16, #layout>) -> tensor<64x256xbf16, #layout>
    %3 = "ttnn.abs"(%2) : (tensor<64x256xbf16, #layout>) -> tensor<64x256xbf16, #layout>
    // Consumer chain after dispatch_d2m: matmul2 -> matmul3 (both are chain anchors)
    %4 = "ttnn.matmul"(%3, %arg2) <{transpose_a = false, transpose_b = false}> : (tensor<64x256xbf16, #layout>, tensor<256x64xbf16, #layout_256x64>) -> tensor<64x64xbf16, #layout_64x64>
    %5 = "ttnn.matmul"(%4, %arg3) <{transpose_a = false, transpose_b = false}> : (tensor<64x64xbf16, #layout_64x64>, tensor<64x256xbf16, #layout>) -> tensor<64x256xbf16, #layout>

    return %5 : tensor<64x256xbf16, #layout>
  }
}
