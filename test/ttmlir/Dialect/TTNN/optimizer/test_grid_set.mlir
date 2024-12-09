// RUN: ttmlir-opt --ttir-load-system-desc --ttnn-optimizer %s | FileCheck %s
#device = #tt.device<workerGrid = #tt.grid<8x8, (d0, d1) -> (0, d0, d1)>, l1Map = (d0, d1)[s0, s1] -> (0, d0 floordiv s0, d1 floordiv s1, (d0 mod s0) * s1 + d1 mod s1), dramMap = (d0, d1)[s0, s1] -> (0, 0, ((((d0 floordiv s0) * 8 + d1 floordiv s1) * (s1 * s0) + (d0 mod s0) * s1 + d1 mod s1) floordiv 8192) mod 12, (((d0 floordiv s0) * 8 + d1 floordiv s1) * (s1 * s0) + (d0 mod s0) * s1 + d1 mod s1) floordiv 98304 + (((d0 floordiv s0) * 8 + d1 floordiv s1) * (s1 * s0) + (d0 mod s0) * s1 + d1 mod s1) mod 8192), meshShape = , chipIds = [0]>
#dram = #ttnn.buffer_type<dram>
#system_memory = #ttnn.buffer_type<system_memory>
#ttnn_layout = #ttnn.ttnn_layout<(d0, d1) -> (d0, d1), <1x1>, memref<64x128xf32, #system_memory>>
#ttnn_layout1 = #ttnn.ttnn_layout<(d0, d1) -> (d0, d1), <1x1>, memref<2x4x!tt.tile<32x32, f32>, #dram>, <interleaved>>
#ttnn_layout2 = #ttnn.ttnn_layout<(d0, d1) -> (d0, d1), <1x1>, memref<64x128xf32, #dram>, <interleaved>>
module attributes {tt.device = #device} {
  func.func @forward(%arg0: tensor<64x128xf32, #ttnn_layout>, %arg1: tensor<64x128xf32, #ttnn_layout>) -> tensor<64x128xf32, #ttnn_layout> {
    %0 = "ttnn.get_device"() <{mesh_shape = #ttnn<mesh_shape 1x1>}> : () -> !tt.device<#device>
    %1 = "ttnn.to_layout"(%arg0, %0) <{dtype = #tt.supportedDataTypes<f32>, layout = #ttnn.layout<tile>, memory_config = #ttnn.memory_config<<dram>, <<64x128>>, <interleaved>>}> : (tensor<64x128xf32, #ttnn_layout>, !tt.device<#device>) -> tensor<64x128xf32, #ttnn_layout1>
    %2 = "ttnn.to_layout"(%arg1, %0) <{dtype = #tt.supportedDataTypes<f32>, layout = #ttnn.layout<tile>, memory_config = #ttnn.memory_config<<dram>, <<64x128>>, <interleaved>>}> : (tensor<64x128xf32, #ttnn_layout>, !tt.device<#device>) -> tensor<64x128xf32, #ttnn_layout1>
    %3 = "ttnn.empty"(%0) <{dtype = #tt.supportedDataTypes<f32>, layout = #ttnn.layout<row_major>, memory_config = #ttnn.memory_config<<dram>, <<64x128>>, <interleaved>>, shape = #ttnn.shape<64x128>}> : (!tt.device<#device>) -> tensor<64x128xf32, #ttnn_layout2>
    %4 = "ttnn.multiply"(%1, %2, %3) <{operandSegmentSizes = array<i32: 2, 1>}> : (tensor<64x128xf32, #ttnn_layout1>, tensor<64x128xf32, #ttnn_layout1>, tensor<64x128xf32, #ttnn_layout2>) -> tensor<64x128xf32, #ttnn_layout2>
    // CHECK: #[[LAYOUT_2:.*]] = #ttnn.ttnn_layout<(d0, d1) -> (d0, d1), <8x8>, memref<8x16xf32, #dram>, <interleaved>>
    // CHECK: %{{.+}} = "ttnn.multiply"{{.+}} -> tensor<64x128xf32, #[[LAYOUT_2]]>
    %5 = "ttnn.to_layout"(%4) <{dtype = #tt.supportedDataTypes<f32>, layout = #ttnn.layout<row_major>, memory_config = #ttnn.memory_config<<system_memory>, <<64x128>>>}> : (tensor<64x128xf32, #ttnn_layout2>) -> tensor<64x128xf32, #ttnn_layout>
    return %5 : tensor<64x128xf32, #ttnn_layout>
  }
}
