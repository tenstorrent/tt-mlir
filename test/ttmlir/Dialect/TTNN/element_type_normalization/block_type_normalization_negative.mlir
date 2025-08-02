// RUN: not ttmlir-opt --split-input-file %s 2>&1 | FileCheck %s

#system_memory = #ttnn.buffer_type<system_memory>
#ttnn_layout = #ttnn.ttnn_layout<(d0, d1) -> (d0, d1), <1x1>, memref<32x32x!ttcore.bfp8_b, #system_memory>>
module {
  // CHECK: error: 'ttnn.add' op BFloat8B type is only supported in Tile layout, but got row_major.
  func.func @forward(%arg0 : tensor<32x32x!ttcore.bfp8_b, #ttnn_layout>) -> tensor<32x32x!ttcore.bfp8_b> {
    %0 = "ttnn.add"(%arg0, %arg0) : (tensor<32x32x!ttcore.bfp8_b, #ttnn_layout>, tensor<32x32x!ttcore.bfp8_b, #ttnn_layout>) -> tensor<32x32x!ttcore.bfp8_b, #ttnn_layout>
    return %0 : tensor<32x32x!ttcore.bfp8_b, #ttnn_layout>
  }
}

// -----

#dram = #ttnn.buffer_type<dram>
#ttnn_layout = #ttnn.ttnn_layout<(d0, d1) -> (d0, d1), <1x1>, memref<1x1x!ttcore.tile<32x32, bf16>, #dram>, <interleaved>>
module {
  // CHECK: error: 'ttnn.add' op Output element type must match the scalar element type from encoding. Element type: '!ttcore.bfp8_b', Scalar element type: 'bf16'.
  func.func @forward(%arg0 : tensor<32x32x!ttcore.bfp8_b, #ttnn_layout>) -> tensor<32x32x!ttcore.bfp8_b> {
    %0 = "ttnn.add"(%arg0, %arg0) : (tensor<32x32x!ttcore.bfp8_b, #ttnn_layout>, tensor<32x32x!ttcore.bfp8_b, #ttnn_layout>) -> tensor<32x32x!ttcore.bfp8_b, #ttnn_layout>
    return %0 : tensor<32x32x!ttcore.bfp8_b, #ttnn_layout>
  }
}
