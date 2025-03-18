// RUN: not ttmlir-opt %s 2>&1 | FileCheck %s
#dram = #ttnn.buffer_type<dram>
#ttnn_layout = #ttnn.ttnn_layout<(d0, d1) -> (d0, d1), <1x1>, memref<2x4x!tt.tile<32x32, f32>, #dram>, <interleaved>>
module attributes {} {
  func.func @add(%arg0: tensor<64x128xf32, #ttnn_layout>, %arg1: tensor<64x128xf32, #ttnn_layout>) -> tensor<64x128xf32, #ttnn_layout> {
    %0 = "ttnn.get_device"() <{mesh_shape = #ttnn<mesh_shape 1x1>}> : () -> !ttnn.device
    %1 = "ttnn.empty"(%0) <{dtype = #tt.supportedDataTypes<f32>, layout = #ttnn.layout<tile>, memory_config = #ttnn.memory_config<#dram, <<2x4>>, <interleaved>>, shape = #ttnn.shape<64x128>}> : (!ttnn.device) -> tensor<64x128xf32, #ttnn_layout>
    %2 = "ttnn.empty"(%0) <{dtype = #tt.supportedDataTypes<f32>, layout = #ttnn.layout<tile>, memory_config = #ttnn.memory_config<#dram, <<2x4>>, <interleaved>>, shape = #ttnn.shape<64x128>}> : (!ttnn.device) -> tensor<64x128xf32, #ttnn_layout>
    %3 = "ttnn.where"(%arg0, %arg1, %1, %2) : (tensor<64x128xf32, #ttnn_layout>, tensor<64x128xf32, #ttnn_layout>, tensor<64x128xf32, #ttnn_layout>, tensor<64x128xf32, #ttnn_layout>) -> tensor<64x128xf32, #ttnn_layout>
    // CHECK: error: 'ttnn.where' op Operation ttnn.where must have exactly 3 operands.
    return %2 : tensor<64x128xf32, #ttnn_layout>
  }
}
