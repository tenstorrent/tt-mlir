// RUN: ttmlir-opt --canonicalize %s | FileCheck %s
#dram = #ttnn.buffer_type<dram>
#system_memory = #ttnn.buffer_type<system_memory>
#ttnn_layout = #ttnn.ttnn_layout<(d0, d1) -> (d0, d1), <1x1>, memref<32x32xbf16, #system_memory>>
#ttnn_layout1 = #ttnn.ttnn_layout<(d0, d1) -> (d0, d1), <1x1>, memref<1x1x!ttcore.tile<32x32, bf16>, #dram>, <interleaved>>
#ttnn_layout2 = #ttnn.ttnn_layout<(d0, d1) -> (d0, d1), <1x1>, memref<32x32xbf16, #dram>, <interleaved>>
#ttnn_layout3 = #ttnn.ttnn_layout<(d0, d1) -> (d0, d1), <1x1>, memref<1x1x!ttcore.tile<32x32, f32>, #dram>, <interleaved>>
#ttnn_layout4 = #ttnn.ttnn_layout<(d0, d1) -> (d0, d1), <1x1>, memref<1x1x!ttcore.tile<32x32, f32>, #system_memory>>
#ttnn_layout5 = #ttnn.ttnn_layout<(d0, d1) -> (d0, d1), <1x1>, memref<32x32xf32, #system_memory>>
module attributes {} {
  func.func @merge_to_layout_op_layout(%arg0: tensor<32x32xbf16, #ttnn_layout>) -> tensor<32x32xbf16, #ttnn_layout2> {
    %0 = "ttnn.get_device"() <{mesh_shape = #ttnn<mesh_shape 1x1>}> : () -> !ttnn.device
    // Verify that the to_layout op is canonicalized to a single to_layout op and the attributes are merged.
    // CHECK: "ttnn.to_layout"(%arg0, %0)
    // CHECK-SAME: dtype = #ttcore.supportedDataTypes<bf16>
    // CHECK-SAME: layout = #ttnn.layout<row_major>
    // CHECK-SAME: memory_config = #ttnn.memory_config<#dram, <interleaved>>
    // CHECK-SAME: -> tensor<32x32xbf16, #ttnn_layout1>
    // CHECK-NEXT: return
    %1 = "ttnn.to_layout"(%arg0, %0) <{dtype = #ttcore.supportedDataTypes<bf16>, layout = #ttnn.layout<tile>, memory_config = #ttnn.memory_config<#dram, <interleaved>>}> : (tensor<32x32xbf16, #ttnn_layout>, !ttnn.device) -> tensor<32x32xbf16, #ttnn_layout1>
    %2 = "ttnn.to_layout"(%1, %0) <{dtype = #ttcore.supportedDataTypes<bf16>, layout = #ttnn.layout<row_major>, memory_config = #ttnn.memory_config<#dram, <interleaved>>}> : (tensor<32x32xbf16, #ttnn_layout1>, !ttnn.device) -> tensor<32x32xbf16, #ttnn_layout2>
    return %2 : tensor<32x32xbf16, #ttnn_layout2>
  }

  func.func @merge_to_layout_op_data_type(%arg0: tensor<32x32xbf16, #ttnn_layout>) -> tensor<32x32xf32, #ttnn_layout3> {
    // Verify that the to_layout op is canonicalized to a single to_layout op and the attributes are merged.
    // CHECK: "ttnn.to_layout"(%arg0, %0)
    // CHECK-SAME: dtype = #ttcore.supportedDataTypes<f32>
    // CHECK-SAME: layout = #ttnn.layout<tile>
    // CHECK-SAME: memory_config = #ttnn.memory_config<#dram, <interleaved>>
    // CHECK-SAME: -> tensor<32x32xf32, #ttnn_layout2>
    // CHECK-NEXT: return
    %0 = "ttnn.get_device"() <{mesh_shape = #ttnn<mesh_shape 1x1>}> : () -> !ttnn.device
    %1 = "ttnn.to_layout"(%arg0, %0) <{dtype = #ttcore.supportedDataTypes<bf16>, layout = #ttnn.layout<tile>, memory_config = #ttnn.memory_config<#dram, <interleaved>>}> : (tensor<32x32xbf16, #ttnn_layout>, !ttnn.device) -> tensor<32x32xbf16, #ttnn_layout1>
    %2 = "ttnn.to_layout"(%1, %0) <{dtype = #ttcore.supportedDataTypes<f32>, layout = #ttnn.layout<tile>, memory_config = #ttnn.memory_config<#dram, <interleaved>>}> : (tensor<32x32xbf16, #ttnn_layout1>, !ttnn.device) -> tensor<32x32xf32, #ttnn_layout3>
    return %2 : tensor<32x32xf32, #ttnn_layout3>
  }

  func.func @merge_to_layout_op_memory_config(%arg0: tensor<32x32xbf16, #ttnn_layout>) -> tensor<32x32xbf16, #ttnn_layout4> {
    // Verify that the to_layout op is canonicalized to a single to_layout op and the attributes are merged.
    // CHECK: "ttnn.to_layout"(%arg0, %0)
    // CHECK-SAME: dtype = #ttcore.supportedDataTypes<bf16>
    // CHECK-SAME: layout = #ttnn.layout<tile>
    // CHECK-SAME: memory_config = #ttnn.memory_config<#system_memory>
    // CHECK-SAME: -> tensor<32x32xbf16, #ttnn_layout3>
    // CHECK-NEXT: return
    %0 = "ttnn.get_device"() <{mesh_shape = #ttnn<mesh_shape 1x1>}> : () -> !ttnn.device
    %1 = "ttnn.to_layout"(%arg0, %0) <{dtype = #ttcore.supportedDataTypes<bf16>, layout = #ttnn.layout<tile>, memory_config = #ttnn.memory_config<#dram, <interleaved>>}> : (tensor<32x32xbf16, #ttnn_layout>, !ttnn.device) -> tensor<32x32xbf16, #ttnn_layout1>
    %2 = "ttnn.to_layout"(%1, %0) <{dtype = #ttcore.supportedDataTypes<bf16>, layout = #ttnn.layout<tile>, memory_config = #ttnn.memory_config<#system_memory>}> : (tensor<32x32xbf16, #ttnn_layout1>, !ttnn.device) -> tensor<32x32xbf16, #ttnn_layout4>
    return %2 : tensor<32x32xbf16, #ttnn_layout4>
  }

  func.func @merge_to_layout_op_all(%arg0: tensor<32x32xbf16, #ttnn_layout>) -> tensor<32x32xf32, #ttnn_layout5> {
    // Verify that the to_layout op is canonicalized to a single to_layout op and the attributes are merged.
    // CHECK: "ttnn.to_layout"(%arg0, %0)
    // CHECK-SAME: dtype = #ttcore.supportedDataTypes<f32>
    // CHECK-SAME: layout = #ttnn.layout<row_major>
    // CHECK-SAME: memory_config = #ttnn.memory_config<#system_memory>
    // CHECK-SAME: -> tensor<32x32xf32, #ttnn_layout4>
    // CHECK-NEXT: return
    %0 = "ttnn.get_device"() <{mesh_shape = #ttnn<mesh_shape 1x1>}> : () -> !ttnn.device
    %1 = "ttnn.to_layout"(%arg0, %0) <{dtype = #ttcore.supportedDataTypes<bf16>, layout = #ttnn.layout<tile>, memory_config = #ttnn.memory_config<#dram, <interleaved>>}> : (tensor<32x32xbf16, #ttnn_layout>, !ttnn.device) -> tensor<32x32xbf16, #ttnn_layout1>
    %2 = "ttnn.to_layout"(%1, %0) <{dtype = #ttcore.supportedDataTypes<f32>, layout = #ttnn.layout<row_major>, memory_config = #ttnn.memory_config<#system_memory>}> : (tensor<32x32xbf16, #ttnn_layout1>, !ttnn.device) -> tensor<32x32xf32, #ttnn_layout5>
    return %2 : tensor<32x32xf32, #ttnn_layout5>
  }

  func.func @merge_to_layout_op_4x(%arg0: tensor<32x32xbf16, #ttnn_layout>) -> tensor<32x32xf32, #ttnn_layout5> {
    // Verify that the to_layout op is canonicalized to a single to_layout op and the attributes are merged.
    // CHECK: "ttnn.to_layout"(%arg0, %0)
    // CHECK-SAME: dtype = #ttcore.supportedDataTypes<f32>
    // CHECK-SAME: layout = #ttnn.layout<row_major>
    // CHECK-SAME: memory_config = #ttnn.memory_config<#system_memory>
    // CHECK-SAME: -> tensor<32x32xf32, #ttnn_layout4>
    // CHECK-NEXT: return
    %0 = "ttnn.get_device"() <{mesh_shape = #ttnn<mesh_shape 1x1>}> : () -> !ttnn.device
    %1 = "ttnn.to_layout"(%arg0, %0) <{dtype = #ttcore.supportedDataTypes<bf16>, layout = #ttnn.layout<tile>, memory_config = #ttnn.memory_config<#dram, <interleaved>>}> : (tensor<32x32xbf16, #ttnn_layout>, !ttnn.device) -> tensor<32x32xbf16, #ttnn_layout1>
    %2 = "ttnn.to_layout"(%1, %0) <{dtype = #ttcore.supportedDataTypes<f32>, layout = #ttnn.layout<row_major>, memory_config = #ttnn.memory_config<#system_memory>}> : (tensor<32x32xbf16, #ttnn_layout1>, !ttnn.device) -> tensor<32x32xf32, #ttnn_layout5>
    %3 = "ttnn.to_layout"(%2, %0) <{dtype = #ttcore.supportedDataTypes<f32>, layout = #ttnn.layout<tile>, memory_config = #ttnn.memory_config<#dram, <interleaved>>}> : (tensor<32x32xf32, #ttnn_layout5>, !ttnn.device) -> tensor<32x32xf32, #ttnn_layout3>
    %4 = "ttnn.to_layout"(%3, %0) <{dtype = #ttcore.supportedDataTypes<f32>, layout = #ttnn.layout<row_major>, memory_config = #ttnn.memory_config<#system_memory>}> : (tensor<32x32xf32, #ttnn_layout3>, !ttnn.device) -> tensor<32x32xf32, #ttnn_layout5>
    return %4 : tensor<32x32xf32, #ttnn_layout5>
  }

  func.func @fold_to_layout_op(%arg0: tensor<32x32xbf16, #ttnn_layout>) -> tensor<32x32xbf16, #ttnn_layout> {
    // Verify folding of to_layout_op.
    %0 = "ttnn.to_layout"(%arg0) <{dtype = #ttcore.supportedDataTypes<bf16>, layout = #ttnn.layout<row_major>, memory_config = #ttnn.memory_config<#system_memory>}> : (tensor<32x32xbf16, #ttnn_layout>) -> tensor<32x32xbf16, #ttnn_layout>
    // CHECK-NOT: "ttnn.to_layout"
    return %0 : tensor<32x32xbf16, #ttnn_layout>
    // CHECK: return %arg0 : tensor<32x32xbf16
  }
}
