// RUN: not ttmlir-opt --split-input-file --ttcore-register-device="system-desc-path=%system_desc_path%" %s 2>&1 | FileCheck %s
// Unit tests for ttnn point_to_point op

#dram = #ttnn.buffer_type<dram>
#ttnn_layout1 = #ttnn.ttnn_layout<(d0, d1) -> (d0, d1), <1x1>, memref<128x512x!ttcore.tile<32x32, f32>, #dram>, <interleaved>>
#ttnn_layout2 = #ttnn.ttnn_layout<(d0, d1) -> (d0, d1), <1x1>, memref<128x256x!ttcore.tile<32x32, f32>, #dram>, <interleaved>>
module @point_to_point_provided_output_tensor_different_shape attributes {} {
  func.func public @main(%arg0: tensor<4096x16384xf32, #ttnn_layout1>) -> (tensor<4096x8192xf32, #ttnn_layout2> {}) {
    %0 = "ttnn.get_device"() <{mesh_shape = #ttnn<mesh_shape 2x4>}> : () -> !ttnn.device
    %1 = "ttnn.empty"(%0) <{dtype = #ttcore.supportedDataTypes<f32>, layout = #ttnn.layout<tile>, memory_config = #ttnn.memory_config<#dram, <interleaved>>, shape = #ttnn.shape<4096x8192>}> : (!ttnn.device) -> tensor<4096x8192xf32, #ttnn_layout2>
    %2 = "ttnn.point_to_point"(%arg0, %1) <{receive_coord = array<i64: 0, 1>, send_coord = array<i64: 0, 0>}> : (tensor<4096x16384xf32, #ttnn_layout1>, tensor<4096x8192xf32, #ttnn_layout2>) -> tensor<4096x8192xf32, #ttnn_layout2>
    return %2 : tensor<4096x8192xf32, #ttnn_layout2>
  }
}
// CHECK: error: 'ttnn.point_to_point' op Accum tensor must match input tensor in shape and element type.


// -----

#dram = #ttnn.buffer_type<dram>
#ttnn_layout1 = #ttnn.ttnn_layout<(d0, d1) -> (d0, d1), <1x1>, memref<128x512x!ttcore.tile<32x32, f32>, #dram>, <interleaved>>
#ttnn_layout2 = #ttnn.ttnn_layout<(d0, d1) -> (d0, d1), <1x1>, memref<128x512x!ttcore.tile<32x32, f16>, #dram>, <interleaved>>
module @point_to_point_provided_output_tensor_different_element_type attributes {} {
  func.func public @main(%arg0: tensor<4096x16384xf32, #ttnn_layout1>) -> (tensor<4096x16384xi32, #ttnn_layout2> {}) {
    %0 = "ttnn.get_device"() <{mesh_shape = #ttnn<mesh_shape 2x4>}> : () -> !ttnn.device
    %1 = "ttnn.empty"(%0) <{dtype = #ttcore.supportedDataTypes<f16>, layout = #ttnn.layout<tile>, memory_config = #ttnn.memory_config<#dram, <interleaved>>, shape = #ttnn.shape<4096x16384>}> : (!ttnn.device) -> tensor<4096x16384xf16, #ttnn_layout2>
    %2 = "ttnn.point_to_point"(%arg0, %1) <{receive_coord = array<i64: 0, 1>, send_coord = array<i64: 0, 0>}> : (tensor<4096x16384xf32, #ttnn_layout1>, tensor<4096x16384xf16, #ttnn_layout2>) -> tensor<4096x16384xf16, #ttnn_layout2>
    return %2 : tensor<4096x16384xf16, #ttnn_layout2>
  }
}
// CHECK: error: 'ttnn.point_to_point' op Accum tensor must match input tensor in shape and element type.
