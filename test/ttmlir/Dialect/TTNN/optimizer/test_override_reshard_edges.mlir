// REQUIRES: opmodel
// RUN: ttmlir-opt --ttcore-register-device --ttnn-optimizer="memory-layout-analysis-enabled=true memreconfig-enabled=true insert-memreconfig=add_0_1_2=0 override-output-layout=add_1_2=1x1:dram:interleaved:row_major:f32" -o %t %s
// RUN: FileCheck %s --input-file=%t
// XFAIL: *
// TODO(rpavlovicTT): #https://github.com/tenstorrent/tt-metal/issues/21846 re-enable

#dram = #ttnn.buffer_type<dram>
#system_memory = #ttnn.buffer_type<system_memory>
#ttnn_layout = #ttnn.ttnn_layout<(d0, d1, d2) -> (d0 * 32 + d1, d2), <1x1>, memref<32x32xf32, #system_memory>>
#ttnn_layout1 = #ttnn.ttnn_layout<(d0, d1, d2) -> (d0 * 32 + d1, d2), <1x1>, memref<32x32xf32, #dram>, <interleaved>>
module attributes {} {
  func.func @main(%arg0: tensor<1x32x32xf32, #ttnn_layout>, %arg1: tensor<1x32x32xf32, #ttnn_layout>, %arg2: tensor<1x32x32xf32, #ttnn_layout>) -> tensor<1x32x32xf32, #ttnn_layout> {
    // CHECK: #[[LAYOUT_1:.*]] = #ttnn.ttnn_layout<(d0, d1, d2) -> (d0 * 32 + d1, d2), <1x1>, memref<32x32xf32, #dram>, <interleaved>>
    // CHECK: #[[LAYOUT_2:.*]] = #ttnn.ttnn_layout<(d0, d1, d2) -> (d0 * 32 + d1, d2), <1x1, (d0, d1) -> (0, d1 floordiv 8, d1 mod 8)>, memref<1x1x!ttcore.tile<32x32, f32>, #l1_>, <width_sharded>>
    // CHECK: #[[LAYOUT_3:.*]] = #ttnn.ttnn_layout<(d0, d1, d2) -> (d0 * 32 + d1, d2), <8x8, (d0, d1) -> (0, d0, d1)>, memref<1x1x!ttcore.tile<32x32, f32>, #dram>, <interleaved>>
    %0 = "ttnn.get_device"() <{mesh_shape = #ttnn<mesh_shape 1x1>}> : () -> !ttnn.device
    %1 = "ttnn.to_layout"(%arg0, %0) <{dtype = #ttcore.supportedDataTypes<f32>, layout = #ttnn.layout<tile>, memory_config = #ttnn.memory_config<<dram>, <interleaved>>}> : (tensor<1x32x32xf32, #ttnn_layout>, !ttnn.device) -> tensor<1x32x32xf32, #ttnn_layout1>
    %2 = "ttnn.to_layout"(%arg1, %0) <{dtype = #ttcore.supportedDataTypes<f32>, layout = #ttnn.layout<tile>, memory_config = #ttnn.memory_config<<dram>, <interleaved>>}> : (tensor<1x32x32xf32, #ttnn_layout>, !ttnn.device) -> tensor<1x32x32xf32, #ttnn_layout1>
    // CHECK: %[[C:.*]] = "ttnn.add"{{.*}} -> tensor<1x32x32xf32, #[[LAYOUT_1]]>
    %3 = "ttnn.add"(%1, %2) : (tensor<1x32x32xf32, #ttnn_layout1>, tensor<1x32x32xf32, #ttnn_layout1>) -> tensor<1x32x32xf32, #ttnn_layout1> loc(#loc1)
    %4 = "ttnn.to_layout"(%arg0, %0) <{dtype = #ttcore.supportedDataTypes<f32>, layout = #ttnn.layout<tile>, memory_config = #ttnn.memory_config<<dram>, <interleaved>>}> : (tensor<1x32x32xf32, #ttnn_layout>, !ttnn.device) -> tensor<1x32x32xf32, #ttnn_layout1>
    // CHECK: %{{.*}} = "ttnn.to_layout"(%[[C]], %0) {{.*}} -> tensor<1x32x32xf32, #[[LAYOUT_2]]>
    %5 = "ttnn.add"(%3, %3) : (tensor<1x32x32xf32, #ttnn_layout1>, tensor<1x32x32xf32, #ttnn_layout1>) -> tensor<1x32x32xf32, #ttnn_layout1> loc(#loc2)
    %6 = "ttnn.relu"(%5) : (tensor<1x32x32xf32, #ttnn_layout1>) -> tensor<1x32x32xf32, #ttnn_layout1> loc(#loc3)
    %7 = "ttnn.to_layout"(%6) <{dtype = #ttcore.supportedDataTypes<f32>, layout = #ttnn.layout<row_major>, memory_config = #ttnn.memory_config<<system_memory>>}> : (tensor<1x32x32xf32, #ttnn_layout1>) -> tensor<1x32x32xf32, #ttnn_layout>
    return %7 : tensor<1x32x32xf32, #ttnn_layout>
  }
}
#loc1 = loc("add_1_2")
#loc2 = loc("add_0_1_2")
#loc3 = loc("relu")
