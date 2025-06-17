// RUN: ttmlir-opt --ttcore-register-device --ttnn-workaround --canonicalize %s | FileCheck %s
#dram = #ttnn.buffer_type<dram>
#system_memory = #ttnn.buffer_type<system_memory>
#ttnn_layout = #ttnn.ttnn_layout<(d0, d1) -> (d0, d1), <1x1>, memref<1x32xf32, #system_memory>>
#ttnn_layout1 = #ttnn.ttnn_layout<(d0, d1) -> (d0, d1), <1x1>, memref<512x128xf32, #system_memory>>
#ttnn_layout2 = #ttnn.ttnn_layout<(d0, d1, d2) -> (d0 * 32 + d1, d2), <1x1>, memref<32x128xf32, #system_memory>>
#ttnn_layout3 = #ttnn.ttnn_layout<(d0, d1) -> (d0, d1), <1x1>, memref<1x1x!ttcore.tile<32x32, f32>, #dram>, <interleaved>>
#ttnn_layout4 = #ttnn.ttnn_layout<(d0, d1) -> (d0, d1), <1x1>, memref<16x4x!ttcore.tile<32x32, f32>, #dram>, <interleaved>>
#ttnn_layout5 = #ttnn.ttnn_layout<(d0, d1, d2) -> (d0 * 32 + d1, d2), <1x1>, memref<1x4x!ttcore.tile<32x32, f32>, #dram>, <interleaved>>
module attributes {} {
  func.func @backward(%arg0: tensor<1x32xf32, #ttnn_layout>, %arg1: tensor<512x128xf32, #ttnn_layout1>, %arg2: tensor<1x32x128xf32, #ttnn_layout2>) -> tensor<512x128xf32, #ttnn_layout1> {
    %0 = "ttnn.get_device"() <{mesh_shape = #ttnn<mesh_shape 1x1>}> : () -> !ttnn.device
    // CHECK: %[[DEVICE_OP:.*]] = "ttnn.get_device"
    %1 = "ttnn.to_layout"(%arg0, %0) <{dtype = #ttcore.supportedDataTypes<f32>, layout = #ttnn.layout<tile>, memory_config = #ttnn.memory_config<#dram, <interleaved>>}> : (tensor<1x32xf32, #ttnn_layout>, !ttnn.device) -> tensor<1x32xf32, #ttnn_layout3>
    %2 = "ttnn.to_layout"(%arg1, %0) <{dtype = #ttcore.supportedDataTypes<f32>, layout = #ttnn.layout<tile>, memory_config = #ttnn.memory_config<#dram, <interleaved>>}> : (tensor<512x128xf32, #ttnn_layout1>, !ttnn.device) -> tensor<512x128xf32, #ttnn_layout4>
    %3 = "ttnn.to_layout"(%arg2, %0) <{dtype = #ttcore.supportedDataTypes<f32>, layout = #ttnn.layout<tile>, memory_config = #ttnn.memory_config<#dram, <interleaved>>}> : (tensor<1x32x128xf32, #ttnn_layout2>, !ttnn.device) -> tensor<1x32x128xf32, #ttnn_layout5>
    // CHECK: "ttnn.reshape"
    %4 = "ttnn.reshape"(%3) <{shape = [1 : i32, 1 : i32, 32 : i32, 128 : i32]}> : (tensor<1x32x128xf32, #ttnn_layout5>) -> tensor<1x1x32x128xf32, #ttnn_layout5>
    // Check that the input operand is transformed into the row major layout.
    // CHECK-NEXT: %[[TO_LAYOUT_INPUT:.*]] = "ttnn.to_layout"
    // CHECK-SAME: dtype = #ttcore.supportedDataTypes<u32>
    // CHECK-SAME: layout = #ttnn.layout<row_major>
    // CHECK-SAME: memory_config = #ttnn.memory_config<#dram, <interleaved>>
    // CHECK-SAME: -> tensor<1x32xui32
    // Check that the data type of the weight operand is transformed in bf16.
    // CHECK-NEXT: %[[TO_LAYOUT_WEIGHTS:.*]] = "ttnn.to_layout"
    // CHECK-SAME: dtype = #ttcore.supportedDataTypes<bf16>
    // CHECK-SAME: layout = #ttnn.layout<tile>
    // CHECK-SAME: memory_config = #ttnn.memory_config<#dram, <interleaved>>
    // CHECK-SAME: -> tensor<512x128xbf16
    // Check that the data type of the in gradient operand is transformed in bf16.
    // CHECK-NEXT: %[[TO_LAYOUT_IN_GRADIENT:.*]] = "ttnn.to_layout"
    // CHECK-SAME: dtype = #ttcore.supportedDataTypes<bf16>
    // CHECK-SAME: layout = #ttnn.layout<tile>
    // CHECK-SAME: memory_config = #ttnn.memory_config<#dram, <interleaved>>
    // CHECK-SAME: -> tensor<1x1x32x128xbf16
    // Check that the data type of the output operand is transformed in bf16.
    %5 = "ttnn.embedding_bw"(%1, %2, %4) <{dtype = #ttcore.supportedDataTypes<f32>, memory_config = #ttnn.memory_config<#dram, <interleaved>>}> : (tensor<1x32xf32, #ttnn_layout3>, tensor<512x128xf32, #ttnn_layout4>, tensor<1x1x32x128xf32, #ttnn_layout5>) -> tensor<512x128xf32, #ttnn_layout4>
    // CHECK-NEXT: %[[EMBEDDING_BW_OP:.*]] = "ttnn.embedding_bw"(%[[TO_LAYOUT_INPUT]], %[[TO_LAYOUT_WEIGHTS]], %[[TO_LAYOUT_IN_GRADIENT]])
    // Check that the output operand is transformed back into the f32 data type.
    // CHECK-NEXT: "ttnn.to_layout"(%[[EMBEDDING_BW_OP]])
    // CHECK-SAME: dtype = #ttcore.supportedDataTypes<f32>
    // CHECK-SAME: layout = #ttnn.layout<row_major>
    // CHECK-SAME: memory_config = #ttnn.memory_config<#system_memory>
    %6 = "ttnn.to_layout"(%5) <{dtype = #ttcore.supportedDataTypes<f32>, layout = #ttnn.layout<row_major>, memory_config = #ttnn.memory_config<#system_memory>}> : (tensor<512x128xf32, #ttnn_layout4>) -> tensor<512x128xf32, #ttnn_layout1>
    return %6 : tensor<512x128xf32, #ttnn_layout1>
  }
}
