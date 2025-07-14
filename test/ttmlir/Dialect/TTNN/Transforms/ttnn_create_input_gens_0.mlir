// RUN: ttmlir-opt --ttnn-create-input-gens %s | FileCheck %s

#dram = #ttnn.buffer_type<dram>
#system_desc = #ttnn.buffer_type<system_memory>
#ttnn_layout = #ttnn.ttnn_layout<(d0, d1) -> (d0, d1), <1x1>, memref<2x4x!ttcore.tile<32x32, f32>, #dram>, <interleaved>>
module attributes {ttcore.system_desc = #system_desc} {
  ttcore.device @default_device = <workerGrid = #ttcore.grid<8x8, (d0, d1) -> (0, d0, d1)>, l1Map = (d0, d1, d2)[s0] -> (0, d0, d1, d2 + s0), dramMap = (d0, d1, d2)[s0, s1, s2, s3, s4, s5] -> (0, 0, (((d0 * s1) * (s2 * s3) + d1 * (s2 * s3) + d2) floordiv s4) mod 12, ((d0 * s1) * (s2 * s3) + d1 * (s2 * s3) + d2) floordiv (s4 * 12) + ((d0 * s1) * (s2 * s3) + d1 * (s2 * s3) + d2) mod s4 + s5), meshShape = , chipIds = [0]>
  // CHECK: func.func @add(%arg0: tuple<[[TENSOR_A:.*>]], [[TENSOR_B:.*]]>) -> tuple<[[TENSOR_OUT:.*]]> {
  func.func @add(%arg0: tuple<tensor<64x128xf32, #ttnn_layout>, tensor<64x128xf32, #ttnn_layout>>) -> tuple<tensor<64x128xf32, #ttnn_layout>> {
    %0 = ttcore.get_tuple_element %arg0[0] : (tuple<tensor<64x128xf32, #ttnn_layout>, tensor<64x128xf32, #ttnn_layout>>) -> tensor<64x128xf32, #ttnn_layout>
    %1 = ttcore.get_tuple_element %arg0[1] : (tuple<tensor<64x128xf32, #ttnn_layout>, tensor<64x128xf32, #ttnn_layout>>) -> tensor<64x128xf32, #ttnn_layout>
    %2 = "ttnn.add"(%0, %1) <{output_dtype = #ttcore.supportedDataTypes<f32>}> : (tensor<64x128xf32, #ttnn_layout>, tensor<64x128xf32, #ttnn_layout>) -> tensor<64x128xf32, #ttnn_layout>
    "ttnn.deallocate"(%1) <{force = false}> : (tensor<64x128xf32, #ttnn_layout>) -> ()
    "ttnn.deallocate"(%0) <{force = false}> : (tensor<64x128xf32, #ttnn_layout>) -> ()
    %3 = ttcore.tuple %2 : tuple<tensor<64x128xf32, #ttnn_layout>>
    return %3 : tuple<tensor<64x128xf32, #ttnn_layout>>
  }
// Confirm that the generator func is generated, and that the tensor attrs match:
//
// CHECK: func.func @create_inputs_for_add() -> tuple<[[TENSOR_A]], [[TENSOR_B]]> {
// CHECK: {{.*}} -> [[TENSOR_A]]
// CHECK: {{.*}} -> [[TENSOR_B]]
// CHECK: return %{{[0-9]+}} : tuple<[[TENSOR_A]], [[TENSOR_B]]>

// Confirm that the main func is generated, and that the tensor attrs match:
//
// CHECK: func.func @main() -> i32 {
// CHECK: %[[ARG:[0-9]+]] = call @create_inputs_for_add() : () -> tuple<[[TENSOR_A]], [[TENSOR_B]]>
// CHECK: %{{[0-9]+}} = call @add(%[[ARG]]) : (tuple<[[TENSOR_A]], [[TENSOR_B]]>) -> tuple<[[TENSOR_OUT]]>
}
