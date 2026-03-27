// RUN: not ttmlir-opt %s 2>&1 | FileCheck %s
// Negative tests for bitcast_convert operation
#dram = #ttnn.buffer_type<dram>
#ttnn_layout_device_tile_u32 = #ttnn.ttnn_layout<(d0, d1) -> (d0, d1), <1x1>, memref<2x4x!ttcore.tile<32x32, u32>, #dram>, <interleaved>>
#ttnn_layout_device_tile_f32 = #ttnn.ttnn_layout<(d0, d1) -> (d0, d1), <1x1>, memref<2x4x!ttcore.tile<32x32, f32>, #dram>, <interleaved>>

// Verify that verification fails when dtype attribute does not match the output tensor dtype.
module {
  func.func @bitcast_convert_dtype_mismatch(%arg0: tensor<2x32x32xui32, #ttnn_layout_device_tile_u32>) -> tensor<2x32x32xf32, #ttnn_layout_device_tile_f32> {
    %0 = "ttnn.bitcast_convert"(%arg0) <{dtype = #ttcore.supportedDataTypes<u32>}> : (tensor<2x32x32xui32, #ttnn_layout_device_tile_u32>) -> tensor<2x32x32xf32, #ttnn_layout_device_tile_f32>
    // CHECK: error: 'ttnn.bitcast_convert' op Output tensor data type f32 must match the data type of dtype attribute u32.
    return %0 : tensor<2x32x32xf32, #ttnn_layout_device_tile_f32>
  }
}

