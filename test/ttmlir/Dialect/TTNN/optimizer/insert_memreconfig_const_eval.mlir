// REQUIRES: opmodel
// RUN: ttmlir-opt --ttcore-register-device --const-eval-hoist-transform --ttnn-optimizer="insert-memreconfig=add_1_2=0" --ttnn-decompose-layouts -o %t %s
// RUN: FileCheck %s --input-file=%t

#dram = #ttnn.buffer_type<dram>
#system_memory = #ttnn.buffer_type<system_memory>
#ttnn_layout = #ttnn.ttnn_layout<(d0, d1, d2) -> (d0 * 32 + d1, d2), <1x1>, memref<32x32xf32, #system_memory>>
#ttnn_layout1 = #ttnn.ttnn_layout<(d0, d1, d2) -> (d0 * 32 + d1, d2), <1x1>, memref<32x32xf32, #dram>, <interleaved>>
module attributes {} {
  // CHECK-LABEL: func.func @main_const_eval_0

  // CHECK-LABEL: func.func @main(
  func.func @main(%arg0: tensor<1x32x32xf32, #ttnn_layout>, %arg1: tensor<1x32x32xf32, #ttnn_layout> {ttcore.argument_type = #ttcore.argument_type<constant>}) -> tensor<1x32x32xf32, #ttnn_layout1> {
    // CHECK: "ttnn.to_layout"
    %0 = "ttnn.to_layout"(%arg1) <{dtype = #ttcore.supportedDataTypes<f32>, layout = #ttnn.layout<tile>, memory_config = #ttnn.memory_config<<dram>, <interleaved>>}> : (tensor<1x32x32xf32, #ttnn_layout>) -> tensor<1x32x32xf32, #ttnn_layout1>
    %1 = "ttnn.add"(%arg0, %0) <{dtype = #ttcore.supportedDataTypes<f32>}> : (tensor<1x32x32xf32, #ttnn_layout>, tensor<1x32x32xf32, #ttnn_layout1>) -> tensor<1x32x32xf32, #ttnn_layout1> loc(#loc1)
    return %1 : tensor<1x32x32xf32, #ttnn_layout1>
  }
}
#loc1 = loc("add_1_2")
