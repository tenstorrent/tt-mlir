// RUN: ttmlir-opt --ttir-load-system-desc --ttnn-optimizer="memory-layout-analysis-enabled=true memreconfig-enabled=true insert-memreconfig=add_0_1_2=0 override-output-layout=add_1_2=1x1:dram:interleaved:row_major:f32" %s | FileCheck %s
#device = #tt.device<workerGrid = #tt.grid<8x8, (d0, d1) -> (0, d0, d1)>, l1Map = (d0, d1)[s0, s1] -> (0, d0 floordiv s0, d1 floordiv s1, (d0 mod s0) * s1 + d1 mod s1), dramMap = (d0, d1)[s0, s1] -> (0, 0, ((((d0 floordiv s0) * 8 + d1 floordiv s1) * (s1 * s0) + (d0 mod s0) * s1 + d1 mod s1) floordiv 8192) mod 12, (((d0 floordiv s0) * 8 + d1 floordiv s1) * (s1 * s0) + (d0 mod s0) * s1 + d1 mod s1) floordiv 98304 + (((d0 floordiv s0) * 8 + d1 floordiv s1) * (s1 * s0) + (d0 mod s0) * s1 + d1 mod s1) mod 8192), meshShape = , chipIds = [0]>
#dram = #ttnn.buffer_type<dram>
#system_memory = #ttnn.buffer_type<system_memory>
#tensor_config = #ttnn.tensor_config<(d0, d1, d2) -> (d0 * 32 + d1, d2), <1x1>, memref<32x32xf32, #system_memory>>
#tensor_config1 = #ttnn.tensor_config<(d0, d1, d2) -> (d0 * 32 + d1, d2), <1x1>, memref<32x32xf32, #dram>, interleaved>
module attributes {tt.device = #device} {
  func.func @main(%arg0: tensor<1x32x32xf32, #tensor_config>, %arg1: tensor<1x32x32xf32, #tensor_config>, %arg2: tensor<1x32x32xf32, #tensor_config>) -> tensor<1x32x32xf32, #tensor_config> {
    // CHECK: #[[LAYOUT_1:.*]] = #ttnn.tensor_config<(d0, d1, d2) -> (d0 * 32 + d1, d2), <1x1>, memref<32x32xf32, #dram>, interleaved>
    // CHECK: #[[LAYOUT_2:.*]] = #ttnn.tensor_config<(d0, d1, d2) -> (d0 * 32 + d1, d2), <1x1>, memref<32x32xf32, #l1_>, width_sharded>
    // CHECK: #[[LAYOUT_3:.*]] = #ttnn.tensor_config<(d0, d1, d2) -> (d0 * 32 + d1, d2), <8x8>, memref<4x4xf32, #dram>, interleaved>
    %0 = "ttnn.get_device"() <{mesh_shape = #ttnn<mesh_shape 1x1>}> : () -> !tt.device<#device>
    %1 = "ttnn.to_layout"(%arg0, %0) <{dtype = #tt.supportedDataTypes<f32>, layout = #ttnn.layout<tile>, memory_config = #ttnn.memory_config<<interleaved>, <dram>, <<32x32>>>}> : (tensor<1x32x32xf32, #tensor_config>, !tt.device<#device>) -> tensor<1x32x32xf32, #tensor_config1>
    %2 = "ttnn.to_layout"(%arg1, %0) <{dtype = #tt.supportedDataTypes<f32>, layout = #ttnn.layout<tile>, memory_config = #ttnn.memory_config<<interleaved>, <dram>, <<32x32>>>}> : (tensor<1x32x32xf32, #tensor_config>, !tt.device<#device>) -> tensor<1x32x32xf32, #tensor_config1>
    %3 = "ttnn.empty"(%0) <{dtype = #tt.supportedDataTypes<f32>, layout = #ttnn.layout<row_major>, memory_config = #ttnn.memory_config<<interleaved>, <dram>, <<32x32>>>, shape = #ttnn.shape<1x32x32>}> : (!tt.device<#device>) -> tensor<1x32x32xf32, #tensor_config1> loc(#loc1)
    // CHECK: %[[C:.*]] = "ttnn.add"{{.*}} -> tensor<1x32x32xf32, #[[LAYOUT_1]]>
    %4 = "ttnn.add"(%1, %2, %3) <{operandSegmentSizes = array<i32: 2, 1>}> : (tensor<1x32x32xf32, #tensor_config1>, tensor<1x32x32xf32, #tensor_config1>, tensor<1x32x32xf32, #tensor_config1>) -> tensor<1x32x32xf32, #tensor_config1> loc(#loc1)
    %5 = "ttnn.to_layout"(%arg0, %0) <{dtype = #tt.supportedDataTypes<f32>, layout = #ttnn.layout<tile>, memory_config = #ttnn.memory_config<<interleaved>, <dram>, <<32x32>>>}> : (tensor<1x32x32xf32, #tensor_config>, !tt.device<#device>) -> tensor<1x32x32xf32, #tensor_config1>
    %6 = "ttnn.empty"(%0) <{dtype = #tt.supportedDataTypes<f32>, layout = #ttnn.layout<row_major>, memory_config = #ttnn.memory_config<<interleaved>, <dram>, <<32x32>>>, shape = #ttnn.shape<1x32x32>}> : (!tt.device<#device>) -> tensor<1x32x32xf32, #tensor_config1> loc(#loc2)
    // CHECK: %{{.*}} = "ttnn.to_layout"(%[[C]], %0) {{.*}} -> tensor<1x32x32xf32, #[[LAYOUT_2]]>
    %7 = "ttnn.add"(%4, %6, %6) <{operandSegmentSizes = array<i32: 2, 1>}> : (tensor<1x32x32xf32, #tensor_config1>, tensor<1x32x32xf32, #tensor_config1>, tensor<1x32x32xf32, #tensor_config1>) -> tensor<1x32x32xf32, #tensor_config1> loc(#loc2)
    %8 = "ttnn.empty"(%0) <{dtype = #tt.supportedDataTypes<f32>, layout = #ttnn.layout<row_major>, memory_config = #ttnn.memory_config<<interleaved>, <dram>, <<32x32>>>, shape = #ttnn.shape<1x32x32>}> : (!tt.device<#device>) -> tensor<1x32x32xf32, #tensor_config1> loc(#loc3)
    %9 = "ttnn.relu"(%7, %8) <{operandSegmentSizes = array<i32: 1, 1>}> : (tensor<1x32x32xf32, #tensor_config1>, tensor<1x32x32xf32, #tensor_config1>) -> tensor<1x32x32xf32, #tensor_config1> loc(#loc3)
    %10 = "ttnn.to_layout"(%9) <{dtype = #tt.supportedDataTypes<f32>, layout = #ttnn.layout<row_major>, memory_config = #ttnn.memory_config<<none>, <system_memory>, <<32x32>>>}> : (tensor<1x32x32xf32, #tensor_config1>) -> tensor<1x32x32xf32, #tensor_config>
    return %10 : tensor<1x32x32xf32, #tensor_config>
  }
}
#loc1 = loc("add_1_2")
#loc2 = loc("add_0_1_2")
#loc3 = loc("relu")
