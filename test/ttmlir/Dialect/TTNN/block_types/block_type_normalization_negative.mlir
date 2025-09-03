// RUN: not ttmlir-opt %s 2>&1 | FileCheck %s

#dram = #ttnn.buffer_type<dram>
#ttnn_layout = #ttnn.ttnn_layout<(d0, d1) -> (d0, d1), <1x1>, memref<1x1x!ttcore.tile<32x32, bf16>, #dram>, <interleaved>>
module {
  // CHECK: error: 'ttnn.add' op Output element type must match the scalar element type from encoding. Element type: '!ttcore.tile<32x32, bfp_bf8>', Scalar element type: 'bf16'.
  func.func @forward(%arg0 : tensor<32x32x!ttcore.tile<32x32, bfp_bf8>, #ttnn_layout>) -> tensor<32x32x!ttcore.tile<32x32, bfp_bf8>, #ttnn_layout> {
    %0 = "ttnn.add"(%arg0, %arg0) <{ output_dtype = #ttcore.supportedDataTypes<bfp_bf8> }> : (tensor<32x32x!ttcore.tile<32x32, bfp_bf8>, #ttnn_layout>, tensor<32x32x!ttcore.tile<32x32, bfp_bf8>, #ttnn_layout>) -> tensor<32x32x!ttcore.tile<32x32, bfp_bf8>, #ttnn_layout>
    return %0 : tensor<32x32x!ttcore.tile<32x32, bfp_bf8>, #ttnn_layout>
  }
}
