// RUN: not ttmlir-opt --split-input-file %s 2>&1 | FileCheck %s
// Negative tests for assign operation

// Verify that verification fails when the output tensor's data type does not match the input tensor's data type.
#dram = #ttnn.buffer_type<dram>
#ttnn_layout1 = #ttnn.ttnn_layout<(d0, d1) -> (d0, d1), <1x1>, memref<1x1x!ttcore.tile<32x32, f32>, #dram>, <interleaved>>
#ttnn_layout2 = #ttnn.ttnn_layout<(d0, d1) -> (d0, d1), <1x1>, memref<1x1x!ttcore.tile<32x32, f16>, #dram>, <interleaved>>
module {
  func.func @forward(%arg0: tensor<32x32xf32, #ttnn_layout1>) -> tensor<32x32xf32, #ttnn_layout2> {
    // CHECK: error: 'ttnn.assign' op output tensor layout data type f16 must match output data type attribute f32
    %1 = "ttnn.assign"(%arg0) <{memory_config = #ttnn.memory_config<#dram, <interleaved>>, dtype = #ttcore.supportedDataTypes<f32>}> : (tensor<32x32xf32, #ttnn_layout1>) -> tensor<32x32xf16, #ttnn_layout2>
    return %1 : tensor<32x32xf16, #ttnn_layout2>
  }
}

// -----

// Verify that verification fails when the output tensor's shape does not match the input tensor's shape.
#dram = #ttnn.buffer_type<dram>
#ttnn_layout1 = #ttnn.ttnn_layout<(d0, d1) -> (d0, d1), <1x1>, memref<1x1x!ttcore.tile<32x32, f32>, #dram>, <interleaved>>
#ttnn_layout2 = #ttnn.ttnn_layout<(d0, d1) -> (d0, d1), <1x1>, memref<1x2x!ttcore.tile<32x32, f32>, #dram>, <interleaved>>
module {
  func.func @forward(%arg0: tensor<32x32xf32, #ttnn_layout1>) -> tensor<32x32xf32, #ttnn_layout2> {
    // CHECK: error: 'ttnn.assign' op input and output tensor must have the same shape
    %1 = "ttnn.assign"(%arg0) <{memory_config = #ttnn.memory_config<#dram, <interleaved>>, dtype = #ttcore.supportedDataTypes<f32>}> : (tensor<32x32xf32, #ttnn_layout1>) -> tensor<32x64xf32, #ttnn_layout2>
    return %1 : tensor<32x64xf32, #ttnn_layout2>
  }
}

// -----

// Verify that verification fails when the output tensor's buffer type does not match the type specified in the memory_config attribute.
#dram = #ttnn.buffer_type<dram>
#l1 = #ttnn.buffer_type<l1>
#ttnn_layout1 = #ttnn.ttnn_layout<(d0, d1) -> (d0, d1), <1x1>, memref<1x1x!ttcore.tile<32x32, f32>, #dram>, <interleaved>>
#ttnn_layout2 = #ttnn.ttnn_layout<(d0, d1) -> (d0, d1), <1x1>, memref<1x1x!ttcore.tile<32x32, f32>, #dram>, <interleaved>>
module {
  func.func @forward(%arg0: tensor<32x32xf32, #ttnn_layout1>) -> tensor<32x32xf32, #ttnn_layout2> {
    // CHECK: error: 'ttnn.assign' op Output tensor buffer type dram must match memory config buffer type l1
    %1 = "ttnn.assign"(%arg0) <{memory_config = #ttnn.memory_config<#l1, <interleaved>>, dtype = #ttcore.supportedDataTypes<f32>}> : (tensor<32x32xf32, #ttnn_layout1>) -> tensor<32x32xf32, #ttnn_layout2>
    return %1 : tensor<32x32xf32, #ttnn_layout2>
  }
}

// -----

// Verify that verification fails when the output tensor's memory layout does not match the layout specified in the memory_config attribute.
#dram = #ttnn.buffer_type<dram>
#l1 = #ttnn.buffer_type<l1>
#ttnn_layout1 = #ttnn.ttnn_layout<(d0, d1) -> (d0, d1), <1x1>, memref<1x1x!ttcore.tile<32x32, f32>, #dram>, <interleaved>>
#ttnn_layout2 = #ttnn.ttnn_layout<(d0, d1) -> (d0, d1), <1x1>, memref<1x1x!ttcore.tile<32x32, f32>, #l1>, <interleaved>>
module {
  func.func @forward(%arg0: tensor<32x32xf32, #ttnn_layout1>) -> tensor<32x32xf32, #ttnn_layout2> {
    // CHECK: error: 'ttnn.assign' op Output tensor layout memory space interleaved must match memory config memory space height_sharded
    %1 = "ttnn.assign"(%arg0) <{memory_config = #ttnn.memory_config<#l1, <height_sharded>>, dtype = #ttcore.supportedDataTypes<f32>}> : (tensor<32x32xf32, #ttnn_layout1>) -> tensor<32x32xf32, #ttnn_layout2>
    return %1 : tensor<32x32xf32, #ttnn_layout2>
  }
}

// -----

// Verify that verification fails when the output tensor's data type does not match the data type described by the dtype attribute.
#dram = #ttnn.buffer_type<dram>
#ttnn_layout1 = #ttnn.ttnn_layout<(d0, d1) -> (d0, d1), <1x1>, memref<1x1x!ttcore.tile<32x32, f32>, #dram>, <interleaved>>
module {
  func.func @forward(%arg0: tensor<32x32xf32, #ttnn_layout1>) -> tensor<32x32xf32, #ttnn_layout1> {
    // CHECK: error: 'ttnn.assign' op output tensor layout data type f32 must match output data type attribute f16
    %1 = "ttnn.assign"(%arg0) <{memory_config = #ttnn.memory_config<#dram, <interleaved>>, dtype = #ttcore.supportedDataTypes<f16>}> : (tensor<32x32xf32, #ttnn_layout1>) -> tensor<32x32xf32, #ttnn_layout1>
    return %1 : tensor<32x32xf32, #ttnn_layout1>
  }
}
