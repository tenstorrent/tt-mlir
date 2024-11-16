// RUN: ttmlir-opt --ttir-load-system-desc --ttnn-optimizer %s | FileCheck %s
#device = #tt.device<workerGrid = #tt.grid<8x8, (d0, d1) -> (0, d0, d1)>, l1Map = (d0, d1)[s0, s1] -> (0, d0 floordiv s0, d1 floordiv s1, (d0 mod s0) * s1 + d1 mod s1), dramMap = (d0, d1)[s0, s1] -> (0, 0, ((((d0 floordiv s0) * 8 + d1 floordiv s1) * (s1 * s0) + (d0 mod s0) * s1 + d1 mod s1) floordiv 8192) mod 12, (((d0 floordiv s0) * 8 + d1 floordiv s1) * (s1 * s0) + (d0 mod s0) * s1 + d1 mod s1) floordiv 98304 + (((d0 floordiv s0) * 8 + d1 floordiv s1) * (s1 * s0) + (d0 mod s0) * s1 + d1 mod s1) mod 8192), meshShape = , chipIds = [0]>
#dram = #ttnn.buffer_type<dram>
#system_memory = #ttnn.buffer_type<system_memory>
#tensor_config = #ttnn.tensor_config<(d0, d1) -> (d0, d1), <1x1>, memref<64x128xf32, #system_memory>>
#tensor_config1 = #ttnn.tensor_config<(d0, d1) -> (d0, d1), <1x1>, memref<2x4x!tt.tile<32x32, f32>, #dram>, interleaved>
#tensor_config2 = #ttnn.tensor_config<(d0, d1) -> (d0, d1), <1x1>, memref<64x128xf32, #dram>, interleaved>
module attributes {tt.device = #device} {
  func.func @forward(%arg0: tensor<64x128xf32, #tensor_config>, %arg1: tensor<64x128xf32, #tensor_config>) -> tensor<64x128xf32, #tensor_config> {
    %0 = "ttnn.get_device"() <{mesh_shape = #ttnn<mesh_shape 1x1>}> : () -> !tt.device<#device>
    %1 = "ttnn.to_layout"(%arg0, %0) <{dtype = #tt.supportedDataTypes<f32>, layout = #ttnn.layout<tile>, memory_config = #ttnn.memory_config<<interleaved>, <dram>, <<64x128>>>}> : (tensor<64x128xf32, #tensor_config>, !tt.device<#device>) -> tensor<64x128xf32, #tensor_config1>
    %2 = "ttnn.to_layout"(%arg1, %0) <{dtype = #tt.supportedDataTypes<f32>, layout = #ttnn.layout<tile>, memory_config = #ttnn.memory_config<<interleaved>, <dram>, <<64x128>>>}> : (tensor<64x128xf32, #tensor_config>, !tt.device<#device>) -> tensor<64x128xf32, #tensor_config1>
    %3 = "ttnn.empty"(%0) <{dtype = #tt.supportedDataTypes<f32>, layout = #ttnn.layout<row_major>, memory_config = #ttnn.memory_config<<interleaved>, <dram>, <<64x128>>>, shape = #ttnn.shape<64x128>}> : (!tt.device<#device>) -> tensor<64x128xf32, #tensor_config2>
    %4 = "ttnn.multiply"(%1, %2, %3) <{operandSegmentSizes = array<i32: 2, 1>}> : (tensor<64x128xf32, #tensor_config1>, tensor<64x128xf32, #tensor_config1>, tensor<64x128xf32, #tensor_config2>) -> tensor<64x128xf32, #tensor_config2>
    // CHECK: #[[LAYOUT_2:.*]] = #ttnn.tensor_config<(d0, d1) -> (d0, d1), <8x8>, memref<8x16xf32, #dram>, interleaved>
    // CHECK: %{{.+}} = "ttnn.multiply"{{.+}} -> tensor<64x128xf32, #[[LAYOUT_2]]>
    %5 = "ttnn.to_layout"(%4) <{dtype = #tt.supportedDataTypes<f32>, layout = #ttnn.layout<row_major>, memory_config = #ttnn.memory_config<<none>, <system_memory>, <<64x128>>>}> : (tensor<64x128xf32, #tensor_config2>) -> tensor<64x128xf32, #tensor_config>
    return %5 : tensor<64x128xf32, #tensor_config>
  }
}
