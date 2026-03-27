// RUN: ttmlir-opt --ttcore-register-device --ttnn-workaround --canonicalize -o %t %s
// RUN: FileCheck %s --input-file=%t

#dram = #ttnn.buffer_type<dram>
#layout_1d = #ttnn.ttnn_layout<(d0) -> (d0), <1x1>, memref<1x1x!ttcore.tile<32x32, f32>, #dram>, <interleaved>>
#layout_weight = #ttnn.ttnn_layout<(d0, d1) -> (d0, d1), <1x1>, memref<16x4x!ttcore.tile<32x32, f32>, #dram>, <interleaved>>
#layout_grad = #ttnn.ttnn_layout<(d0, d1, d2, d3) -> (d0 * 32 + d1 * 32 + d2, d3), <1x1>, memref<1x4x!ttcore.tile<32x32, f32>, #dram>, <interleaved>>

module attributes {} {
  func.func @embedding_bw_1d_indices(%arg0: tensor<32xf32, #layout_1d>, %arg1: tensor<512x128xf32, #layout_weight>, %arg2: tensor<1x1x32x128xf32, #layout_grad>) -> tensor<512x128xf32, #layout_weight> {
    // CHECK: "ttnn.reshape"
    // CHECK-SAME: shape = [1 : i32, 32 : i32]
    // CHECK: "ttnn.embedding_bw"
    %0 = "ttnn.embedding_bw"(%arg0, %arg1, %arg2) <{dtype = #ttcore.supportedDataTypes<f32>, memory_config = #ttnn.memory_config<#dram, <interleaved>>}> : (tensor<32xf32, #layout_1d>, tensor<512x128xf32, #layout_weight>, tensor<1x1x32x128xf32, #layout_grad>) -> tensor<512x128xf32, #layout_weight>
    return %0 : tensor<512x128xf32, #layout_weight>
  }
}
