// RUN: ttmlir-opt --ttnn-create-input-gens %s | FileCheck %s

#dram = #ttnn.buffer_type<dram>
#system_memory = #ttnn.buffer_type<system_memory>
#ttnn_layout = #ttnn.ttnn_layout<(d0, d1) -> (d0, d1), <1x1>, memref<32x32xbf16, #system_memory>>
#ttnn_layout1 = #ttnn.ttnn_layout<(d0, d1) -> (d0, d1), <1x1>, memref<32x32xbf16, #dram>, <interleaved>>
#ttnn_layout2 = #ttnn.ttnn_layout<(d0, d1) -> (d0, d1), <1x1>, memref<1x1x!tt.tile<32x32, bf16>, #dram>, <interleaved>>
#ttnn_layout3 = #ttnn.ttnn_layout<(d0, d1) -> (d0, d1), <1x1>, memref<1x1x!tt.tile<32x32, bf16>, #system_memory>>
module attributes {} {
  // CHECK: func.func @add(%arg0: [[TENSOR_A:.*]], %arg1: [[TENSOR_B:.*]]) -> [[TENSOR_OUT:.*]] {
  func.func @add(%arg0: tensor<32x32xbf16, #ttnn_layout>, %arg1: tensor<32x32xbf16, #ttnn_layout>) -> tensor<32x32xbf16, #ttnn_layout> {
    %0 = "ttnn.get_device"() <{mesh_shape = #ttnn<mesh_shape 1x1>}> : () -> !ttnn.device
    %1 = "ttnn.to_device"(%arg0, %0) <{memory_config = #ttnn.memory_config<#dram, <<1x1>>, <interleaved>>}> : (tensor<32x32xbf16, #ttnn_layout>, !ttnn.device) -> tensor<32x32xbf16, #ttnn_layout1>
    %2 = "ttnn.to_layout"(%1) <{layout = #ttnn.layout<tile>}> : (tensor<32x32xbf16, #ttnn_layout1>) -> tensor<32x32xbf16, #ttnn_layout2>
    %3 = "ttnn.to_device"(%arg1, %0) <{memory_config = #ttnn.memory_config<#dram, <<1x1>>, <interleaved>>}> : (tensor<32x32xbf16, #ttnn_layout>, !ttnn.device) -> tensor<32x32xbf16, #ttnn_layout1>
    %4 = "ttnn.to_layout"(%3) <{layout = #ttnn.layout<tile>}> : (tensor<32x32xbf16, #ttnn_layout1>) -> tensor<32x32xbf16, #ttnn_layout2>
    %5 = "ttnn.add"(%2, %4) : (tensor<32x32xbf16, #ttnn_layout2>, tensor<32x32xbf16, #ttnn_layout2>) -> tensor<32x32xbf16, #ttnn_layout2>
    %6 = "ttnn.from_device"(%5) : (tensor<32x32xbf16, #ttnn_layout2>) -> tensor<32x32xbf16, #ttnn_layout3>
    %7 = "ttnn.to_layout"(%6) <{layout = #ttnn.layout<row_major>}> : (tensor<32x32xbf16, #ttnn_layout3>) -> tensor<32x32xbf16, #ttnn_layout>
    return %7 : tensor<32x32xbf16, #ttnn_layout>
  }

// Confirm that the generator func is generated, and that the tensor attrs match:
//
// CHECK: func.func @createInputsFor_add() -> ([[TENSOR_A]], [[TENSOR_B]]) {
// CHECK: {{.*}} -> [[TENSOR_A]]
// CHECK: {{.*}} -> [[TENSOR_B]]
// CHECK: return %0, %1 : [[TENSOR_A]], [[TENSOR_B]]

// Confirm that the main func is generated, and that the tensor attrs match:
//
// CHECK: func.func @main() -> i32 {
// CHECK: %0:2 = call @createInputsFor_add() : () -> ([[TENSOR_A]], [[TENSOR_B]])
// CHECK: %1 = call @add(%0#0, %0#1) : ([[TENSOR_A]], [[TENSOR_B]]) -> [[TENSOR_OUT]]
}
