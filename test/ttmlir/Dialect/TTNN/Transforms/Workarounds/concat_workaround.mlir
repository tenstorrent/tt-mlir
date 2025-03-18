// RUN: ttmlir-opt --ttnn-workaround --canonicalize %s | FileCheck %s

#dram = #ttnn.buffer_type<dram>
#ttnn_layout = #ttnn.ttnn_layout<(d0, d1) -> (d0, d1), <1x1>, memref<1x2x!tt.tile<32x32, u32>, #dram>, <interleaved>>
#ttnn_layout1 = #ttnn.ttnn_layout<(d0, d1) -> (d0, d1), <1x1>, memref<1x1x!tt.tile<32x32, u32>, #dram>, <interleaved>>
module attributes {} {
  func.func @test_concat_workaround(%arg0: tensor<1x53xui32, #ttnn_layout>, %arg1: tensor<1x1xui32, #ttnn_layout1>) -> tensor<1x54xui32, #ttnn_layout> {
    %0 = "ttnn.get_device"() <{mesh_shape = #ttnn<mesh_shape 1x1>}> : () -> !ttnn.device
    %1 = "ttnn.empty"(%0) <{dtype = #tt.supportedDataTypes<u32>, layout = #ttnn.layout<tile>, memory_config = #ttnn.memory_config<#dram, <<1x2>>, <interleaved>>, shape = #ttnn.shape<1x54>}> : (!ttnn.device) -> tensor<1x54xui32, #ttnn_layout>
    // CHECK: %[[ARG0:[0-9]+]] = "ttnn.to_layout"
    // CHECK-SAME: dtype = #tt.supportedDataTypes<bf16>
    // CHECK-SAME: tensor<1x53xui32
    // CHECK-SAME: -> tensor<1x53xbf16
    // CHECK: %[[ARG1:[0-9]+]] = "ttnn.to_layout"
    // CHECK-SAME: dtype = #tt.supportedDataTypes<bf16>
    // CHECK-SAME: tensor<1x1xui32
    // CHECK-SAME: -> tensor<1x1xbf16
    // CHECK: %[[ARG2:[0-9]+]] = "ttnn.to_layout"
    // CHECK-SAME: dtype = #tt.supportedDataTypes<bf16>
    // CHECK-SAME: tensor<1x54xui32
    // CHECK-SAME: -> tensor<1x54xbf16
    // CHECK: %[[CONCAT:[0-9]+]] = "ttnn.concat"(%[[ARG0]], %[[ARG1]], %[[ARG2]])
    // CHECK-SAME: dim = 1 : si32
    // CHECK-SAME: tensor<1x53xbf16
    // CHECK-SAME: tensor<1x1xbf16
    // CHECK-SAME: tensor<1x54xbf16
    // CHECK-SAME: -> tensor<1x54xbf16
    %2 = "ttnn.concat"(%arg0, %arg1, %1) <{dim = 1 : si32}> : (tensor<1x53xui32, #ttnn_layout>, tensor<1x1xui32, #ttnn_layout1>, tensor<1x54xui32, #ttnn_layout>) -> tensor<1x54xui32, #ttnn_layout>
    // CHECK: %6 = "ttnn.to_layout"(%[[CONCAT]]
    // CHECK-SAME: dtype = #tt.supportedDataTypes<u32>
    // CHECK-SAME: tensor<1x54xbf16
    // CHECK-SAME: -> tensor<1x54xui32
    return %2 : tensor<1x54xui32, #ttnn_layout>
  }
}
