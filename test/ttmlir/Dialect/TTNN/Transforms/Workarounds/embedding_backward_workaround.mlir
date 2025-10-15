// RUN: ttmlir-opt --ttcore-register-device --ttnn-workaround --canonicalize -o %t %s
// RUN: FileCheck %s --input-file=%t
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
    %0 = "ttnn.to_layout"(%arg0) <{dtype = #ttcore.supportedDataTypes<f32>, layout = #ttnn.layout<tile>, memory_config = #ttnn.memory_config<#dram, <interleaved>>}> : (tensor<1x32xf32, #ttnn_layout>) -> tensor<1x32xf32, #ttnn_layout3>
    %1 = "ttnn.to_layout"(%arg1) <{dtype = #ttcore.supportedDataTypes<f32>, layout = #ttnn.layout<tile>, memory_config = #ttnn.memory_config<#dram, <interleaved>>}> : (tensor<512x128xf32, #ttnn_layout1>) -> tensor<512x128xf32, #ttnn_layout4>
    %2 = "ttnn.to_layout"(%arg2) <{dtype = #ttcore.supportedDataTypes<f32>, layout = #ttnn.layout<tile>, memory_config = #ttnn.memory_config<#dram, <interleaved>>}> : (tensor<1x32x128xf32, #ttnn_layout2>) -> tensor<1x32x128xf32, #ttnn_layout5>
    // CHECK: "ttnn.view"
    %3 = "ttnn.view"(%2) <{shape = [1 : i32, 1 : i32, 32 : i32, 128 : i32]}> : (tensor<1x32x128xf32, #ttnn_layout5>) -> tensor<1x1x32x128xf32, #ttnn_layout5>
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
    %4 = "ttnn.embedding_bw"(%0, %1, %3) <{dtype = #ttcore.supportedDataTypes<f32>, memory_config = #ttnn.memory_config<#dram, <interleaved>>}> : (tensor<1x32xf32, #ttnn_layout3>, tensor<512x128xf32, #ttnn_layout4>, tensor<1x1x32x128xf32, #ttnn_layout5>) -> tensor<512x128xf32, #ttnn_layout4>
    // CHECK-NEXT: %[[EMBEDDING_BW_OP:.*]] = "ttnn.embedding_bw"(%[[TO_LAYOUT_INPUT]], %[[TO_LAYOUT_WEIGHTS]], %[[TO_LAYOUT_IN_GRADIENT]])
    // Check that the output operand is transformed back into the f32 data type.
    // CHECK-NEXT: "ttnn.to_layout"(%[[EMBEDDING_BW_OP]])
    // CHECK-SAME: dtype = #ttcore.supportedDataTypes<f32>
    // CHECK-SAME: layout = #ttnn.layout<row_major>
    // CHECK-SAME: memory_config = #ttnn.memory_config<#system_memory>
    %5 = "ttnn.to_layout"(%4) <{dtype = #ttcore.supportedDataTypes<f32>, layout = #ttnn.layout<row_major>, memory_config = #ttnn.memory_config<#system_memory>}> : (tensor<512x128xf32, #ttnn_layout4>) -> tensor<512x128xf32, #ttnn_layout1>
    return %5 : tensor<512x128xf32, #ttnn_layout1>
  }
}
