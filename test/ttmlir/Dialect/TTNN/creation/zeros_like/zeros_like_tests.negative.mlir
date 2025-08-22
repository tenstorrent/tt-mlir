// RUN: not ttmlir-opt --ttcore-register-device --split-input-file %s 2>&1 | FileCheck %s
// Negative tests for zeros_like operation

// Verify that the parsing fails if input and output shapes do not match.
module {
  func.func @zeros_like_shape_mismatch(%arg0: tensor<64x64xbf16>) -> tensor<64x128xbf16> {
    // CHECK: error: 'ttnn.zeros_like' op output tensor shape must be (64, 64), but got (64, 128)
    %0 = ttir.empty() : tensor<64x128xbf16>
    %1 = "ttnn.zeros_like"(%arg0) : (tensor<64x64xbf16>) -> tensor<64x128xbf16>
    return %1 : tensor<64x128xbf16>
  }
}

// -----
// Verify that parsing fails in case if input and output data types do not match.
#dram = #ttnn.buffer_type<dram>
#ttnn_layout = #ttnn.ttnn_layout<(d0, d1) -> (d0, d1), <1x1>, memref<2x4x!ttcore.tile<32x32, f32>, #dram>, <interleaved>>
module {
  func.func @zeros_like_data_type_mismatch(%arg0: tensor<64x128xf32, #ttnn_layout>) -> tensor<64x128xf32, #ttnn_layout> {
    // CHECK: error: 'ttnn.zeros_like' op output tensor layout data type f32 must match output data type attribute bf16
    %0 = "ttnn.get_device"() <{mesh_shape = #ttnn<mesh_shape 1x1>}> : () -> !ttnn.device
    %1 = "ttnn.zeros_like"(%arg0, %0) <{ output_dtype = #ttcore.supportedDataTypes<bf16>, layout = #ttnn.layout<tile>, memory_config = #ttnn.memory_config<#dram, <interleaved>>}> : (tensor<64x128xf32, #ttnn_layout>, !ttnn.device) -> tensor<64x128xf32, #ttnn_layout>
    return %1 : tensor<64x128xf32, #ttnn_layout>
  }
}

// -----
// Verify that parsing fails in case if input and output tensor layout do not match.
#dram = #ttnn.buffer_type<dram>
#ttnn_layout = #ttnn.ttnn_layout<(d0, d1) -> (d0, d1), <1x1>, memref<2x4x!ttcore.tile<32x32, f32>, #dram>, <interleaved>>
module {
  func.func @zeros_like_layout_mismatch(%arg0: tensor<64x128xf32, #ttnn_layout>) -> tensor<64x128xf32, #ttnn_layout> {
    // CHECK: error: 'ttnn.zeros_like' op output tensor layout tile must match layout attribute row_major
    %0 = "ttnn.get_device"() <{mesh_shape = #ttnn<mesh_shape 1x1>}> : () -> !ttnn.device
    %1 = "ttnn.zeros_like"(%arg0, %0) <{ output_dtype = #ttcore.supportedDataTypes<f32>, layout = #ttnn.layout<row_major>, memory_config = #ttnn.memory_config<#dram, <interleaved>>}> : (tensor<64x128xf32, #ttnn_layout>, !ttnn.device) -> tensor<64x128xf32, #ttnn_layout>
    return %1 : tensor<64x128xf32, #ttnn_layout>
  }
}

// -----
// Verify that parsing fails in case if input and output tensor memory config do not match.
#dram = #ttnn.buffer_type<dram>
#l1 = #ttnn.buffer_type<l1>
#ttnn_layout = #ttnn.ttnn_layout<(d0, d1) -> (d0, d1), <1x1>, memref<2x4x!ttcore.tile<32x32, f32>, #dram>, <interleaved>>
module {
  func.func @zeros_like_memory_config_mismatch(%arg0: tensor<64x128xf32, #ttnn_layout>) -> tensor<64x128xf32, #ttnn_layout> {
    // CHECK: error: 'ttnn.zeros_like' op Output tensor buffer type dram must match memory config buffer type l1
    %0 = "ttnn.get_device"() <{mesh_shape = #ttnn<mesh_shape 1x1>}> : () -> !ttnn.device
    %1 = "ttnn.zeros_like"(%arg0, %0) <{ output_dtype = #ttcore.supportedDataTypes<f32>, layout = #ttnn.layout<tile>, memory_config = #ttnn.memory_config<#l1, <interleaved>>}> : (tensor<64x128xf32, #ttnn_layout>, !ttnn.device) -> tensor<64x128xf32, #ttnn_layout>
    return %1 : tensor<64x128xf32, #ttnn_layout>
  }
}
