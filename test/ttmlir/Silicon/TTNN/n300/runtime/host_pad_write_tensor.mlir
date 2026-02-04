// RUN: ttmlir-opt --ttcore-register-device="system-desc-path=%system_desc_path%" -o %t.mlir %s
// RUN: FileCheck %s --input-file=%t.mlir
// RUN: ttmlir-translate --ttnn-to-flatbuffer -o %t.ttnn %t.mlir

// After ttnn.pad on a host tensor, the logical shape may not match the padded
// shape. The write_tensor runtime aligns them before writing to device.

#system_memory = #ttnn.buffer_type<system_memory>
#dram = #ttnn.buffer_type<dram>
#layout_host = #ttnn.ttnn_layout<(d0, d1, d2, d3) -> (d0 * 96 + d1 * 3 + d2, d3), <1x1>, memref<6144x3xf32, #system_memory>>
#layout_host_padded = #ttnn.ttnn_layout<(d0, d1, d2, d3) -> (d0 * 96 + d1 * 3 + d2, d3), <1x1>, memref<6144x32xf32, #system_memory>>
#layout_device = #ttnn.ttnn_layout<(d0, d1, d2, d3) -> (d0 * 96 + d1 * 3 + d2, d3), <1x1>, memref<6144x32xf32, #dram>, <interleaved>>

func.func @test_pad_host_to_device(%arg0: tensor<64x32x3x3xf32, #layout_host>) -> tensor<64x32x3x32xf32, #layout_device> {
  %device = "ttnn.get_device"() <{mesh_shape = #ttnn<mesh_shape 1x2>}> : () -> !ttnn.device

  // CHECK: ttnn.pad
  %padded = "ttnn.pad"(%arg0) <{padding = array<i32: 0, 0, 0, 0, 0, 0, 0, 29>, value = 0.0 : f32, use_multicore = false}> : (tensor<64x32x3x3xf32, #layout_host>) -> tensor<64x32x3x32xf32, #layout_host_padded>

  // CHECK: ttnn.empty
  %empty = "ttnn.empty"(%device) <{dtype = #ttcore.supportedDataTypes<f32>, layout = #ttnn.layout<row_major>, memory_config = #ttnn.memory_config<#dram, <interleaved>>, shape = #ttnn.shape<64x32x3x32>}> : (!ttnn.device) -> tensor<64x32x3x32xf32, #layout_device>

  // CHECK: ttnn.write_tensor
  "ttnn.write_tensor"(%padded, %empty) <{blocking = false, cq_id = 0 : ui32}> : (tensor<64x32x3x32xf32, #layout_host_padded>, tensor<64x32x3x32xf32, #layout_device>) -> ()

  return %empty : tensor<64x32x3x32xf32, #layout_device>
}
