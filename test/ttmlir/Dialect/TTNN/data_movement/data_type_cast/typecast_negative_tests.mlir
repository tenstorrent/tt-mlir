// RUN: not ttmlir-opt %s 2>&1 | FileCheck %s
// Negative tests for typecast operation
#dram = #ttnn.buffer_type<dram>
#ttnn_layout_device_tile_f32 = #ttnn.ttnn_layout<(d0, d1) -> (d0, d1), <1x1>, memref<2x4x!ttcore.tile<32x32, f32>, #dram>, <interleaved>>
#ttnn_layout_device_tile_bf16 = #ttnn.ttnn_layout<(d0, d1) -> (d0, d1), <1x1>, memref<2x4x!ttcore.tile<32x32, bf16>, #dram>, <interleaved>>
// Verify that verification fails when dtype attribute is empty.
module {
  func.func @to_dtype_dtype_attribute_missmatch(%arg0: tensor<2x32x32xf32, #ttnn_layout_device_tile_f32>) -> tensor<2x32x32xbf16, #ttnn_layout_device_tile_bf16> {
    %0 = "ttnn.typecast"(%arg0) <{dtype = #ttcore.supportedDataTypes<f32>}> : (tensor<2x32x32xf32, #ttnn_layout_device_tile_f32>) -> tensor<2x32x32xbf16, #ttnn_layout_device_tile_bf16>
    // CHECK: error: 'ttnn.typecast' op Output tensor data type bf16 must match the data type of dtype attribute f32.
    return %0 : tensor<2x32x32xbf16, #ttnn_layout_device_tile_bf16>
  }
}
