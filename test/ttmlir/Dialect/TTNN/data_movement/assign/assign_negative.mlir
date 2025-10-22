// RUN: not ttmlir-opt --split-input-file %s 2>&1 | FileCheck %s
// Negative tests for assign operation

// Verify that verification fails when the output tensor's data type does not match the input tensor's data type.
#dram = #ttnn.buffer_type<dram>
#ttnn_layout1 = #ttnn.ttnn_layout<(d0, d1) -> (d0, d1), <1x1>, memref<1x1x!ttcore.tile<32x32, f32>, #dram>, <interleaved>>
#ttnn_layout2 = #ttnn.ttnn_layout<(d0, d1) -> (d0, d1), <1x1>, memref<1x1x!ttcore.tile<32x32, f16>, #dram>, <interleaved>>
module {
  func.func @forward(%arg0: tensor<32x32xf32, #ttnn_layout1>) -> tensor<32x32xf32, #ttnn_layout2> {
    // CHECK: error: 'ttnn.assign' op output tensor data type does not match expected output data type
    %1 = "ttnn.assign"(%arg0) <{output_mem_config = #ttnn.memory_config<#dram, <interleaved>>}> : (tensor<32x32xf32, #ttnn_layout1>) -> tensor<32x32xf16, #ttnn_layout2>
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
    %1 = "ttnn.assign"(%arg0) <{output_mem_config = #ttnn.memory_config<#dram, <interleaved>>}> : (tensor<32x32xf32, #ttnn_layout1>) -> tensor<32x64xf32, #ttnn_layout2>
    return %1 : tensor<32x64xf32, #ttnn_layout2>
  }
}

// -----

// Verify that verification fails when the output tensor's buffer type does not match the type specified in the output_mem_config attribute.
#dram = #ttnn.buffer_type<dram>
#l1 = #ttnn.buffer_type<l1>
#ttnn_layout1 = #ttnn.ttnn_layout<(d0, d1) -> (d0, d1), <1x1>, memref<1x1x!ttcore.tile<32x32, f32>, #dram>, <interleaved>>
#ttnn_layout2 = #ttnn.ttnn_layout<(d0, d1) -> (d0, d1), <1x1>, memref<1x1x!ttcore.tile<32x32, f32>, #dram>, <interleaved>>
module {
  func.func @forward(%arg0: tensor<32x32xf32, #ttnn_layout1>) -> tensor<32x32xf32, #ttnn_layout2> {
    // CHECK: error: 'ttnn.assign' op Buffer type mismatch between op and output memory config
    %1 = "ttnn.assign"(%arg0) <{output_mem_config = #ttnn.memory_config<#l1, <interleaved>>}> : (tensor<32x32xf32, #ttnn_layout1>) -> tensor<32x32xf32, #ttnn_layout2>
    return %1 : tensor<32x32xf32, #ttnn_layout2>
  }
}

// -----

// Verify that verification fails when the output tensor's memory layout does not match the layout specified in the output_mem_config attribute.
#dram = #ttnn.buffer_type<dram>
#l1 = #ttnn.buffer_type<l1>
#ttnn_layout1 = #ttnn.ttnn_layout<(d0, d1) -> (d0, d1), <1x1>, memref<1x1x!ttcore.tile<32x32, f32>, #dram>, <interleaved>>
#ttnn_layout2 = #ttnn.ttnn_layout<(d0, d1) -> (d0, d1), <1x1>, memref<1x1x!ttcore.tile<32x32, f32>, #l1>, <interleaved>>
module {
  func.func @forward(%arg0: tensor<32x32xf32, #ttnn_layout1>) -> tensor<32x32xf32, #ttnn_layout2> {
    // CHECK: error: 'ttnn.assign' op Tensor memory layout mismatch between op and output memory config
    %1 = "ttnn.assign"(%arg0) <{output_mem_config = #ttnn.memory_config<#l1, <height_sharded>>}> : (tensor<32x32xf32, #ttnn_layout1>) -> tensor<32x32xf32, #ttnn_layout2>
    return %1 : tensor<32x32xf32, #ttnn_layout2>
  }
}

// -----

// Verify that verification fails when the output tensor's data type does not match the data type described by the dtype attribute.
#dram = #ttnn.buffer_type<dram>
#ttnn_layout1 = #ttnn.ttnn_layout<(d0, d1) -> (d0, d1), <1x1>, memref<1x1x!ttcore.tile<32x32, f32>, #dram>, <interleaved>>
module {
  func.func @forward(%arg0: tensor<32x32xf32, #ttnn_layout1>) -> tensor<32x32xf32, #ttnn_layout1> {
    // CHECK: error: 'ttnn.assign' op output tensor data type does not match expected output data type
    %1 = "ttnn.assign"(%arg0) <{output_mem_config = #ttnn.memory_config<#dram, <interleaved>>, output_dtype = #ttcore.supportedDataTypes<f16>}> : (tensor<32x32xf32, #ttnn_layout1>) -> tensor<32x32xf32, #ttnn_layout1>
    return %1 : tensor<32x32xf32, #ttnn_layout1>
  }
}

// -----

// Verify that verification fails when the output tensor's shape does not match the provided optional output tensor's shape.
#dram = #ttnn.buffer_type<dram>
#ttnn_layout1 = #ttnn.ttnn_layout<(d0, d1) -> (d0, d1), <1x1>, memref<1x1x!ttcore.tile<32x32, f32>, #dram>, <interleaved>>
#ttnn_layout2 = #ttnn.ttnn_layout<(d0, d1) -> (d0, d1), <1x1>, memref<1x2x!ttcore.tile<32x32, f32>, #dram>, <interleaved>>
module {
  func.func @forward(%arg0: tensor<32x32xf32, #ttnn_layout1>) -> tensor<32x64xf32, #ttnn_layout2> {
    %0 = "ttnn.get_device"() <{mesh_shape = #ttnn<mesh_shape 1x1>}> : () -> !ttnn.device
    %1 = "ttnn.empty"(%0) <{dtype = #ttcore.supportedDataTypes<f32>, layout = #ttnn.layout<tile>, memory_config = #ttnn.memory_config<#dram, <interleaved>>, shape = #ttnn.shape<32x32>}> : (!ttnn.device) -> tensor<32x32xf32, #ttnn_layout1>
    // CHECK: error: 'ttnn.assign' op input and output tensor must have the same shape
    %2 = "ttnn.assign"(%arg0, %1) <{output_mem_config = #ttnn.memory_config<#dram, <interleaved>>}> : (tensor<32x32xf32, #ttnn_layout1>, tensor<32x32xf32, #ttnn_layout1>) -> tensor<32x64xf32, #ttnn_layout2>
    return %2 : tensor<32x64xf32, #ttnn_layout2>
  }
}

// -----

// Verify that verification fails when the output tensor's data type does not match the provided optional output tensor's data type.
#dram = #ttnn.buffer_type<dram>
#ttnn_layout1 = #ttnn.ttnn_layout<(d0, d1) -> (d0, d1), <1x1>, memref<1x1x!ttcore.tile<32x32, f32>, #dram>, <interleaved>>
#ttnn_layout2 = #ttnn.ttnn_layout<(d0, d1) -> (d0, d1), <1x1>, memref<1x1x!ttcore.tile<32x32, f16>, #dram>, <interleaved>>
module {
  func.func @forward(%arg0: tensor<32x32xf32, #ttnn_layout1>) -> tensor<32x32xf16, #ttnn_layout2> {
    %0 = "ttnn.get_device"() <{mesh_shape = #ttnn<mesh_shape 1x1>}> : () -> !ttnn.device
    %1 = "ttnn.empty"(%0) <{dtype = #ttcore.supportedDataTypes<f32>, layout = #ttnn.layout<tile>, memory_config = #ttnn.memory_config<#dram, <interleaved>>, shape = #ttnn.shape<32x32>}> : (!ttnn.device) -> tensor<32x32xf32, #ttnn_layout1>
    // CHECK: error: 'ttnn.assign' op optional output tensor data type does not match output tensor data type
    %2 = "ttnn.assign"(%arg0, %1) <{output_mem_config = #ttnn.memory_config<#dram, <interleaved>>}> : (tensor<32x32xf32, #ttnn_layout1>, tensor<32x32xf32, #ttnn_layout1>) -> tensor<32x32xf16, #ttnn_layout2>
    return %2 : tensor<32x32xf16, #ttnn_layout2>
  }
}

// -----

// Verify that verification fails when the output tensor's data type does not match the provided optional output tensor's data type.
#dram = #ttnn.buffer_type<dram>
#ttnn_layout1 = #ttnn.ttnn_layout<(d0, d1) -> (d0, d1), <1x1>, memref<1x1x!ttcore.tile<32x32, f32>, #dram>, <interleaved>>
#ttnn_layout2 = #ttnn.ttnn_layout<(d0, d1) -> (d0, d1), <1x1>, memref<1x1x!ttcore.tile<32x32, f16>, #dram>, <interleaved>>
module {
  func.func @forward(%arg0: tensor<32x32xf32, #ttnn_layout1>) -> tensor<32x32xf32, #ttnn_layout1> {
    %0 = "ttnn.get_device"() <{mesh_shape = #ttnn<mesh_shape 1x1>}> : () -> !ttnn.device
    %1 = "ttnn.empty"(%0) <{dtype = #ttcore.supportedDataTypes<f16>, layout = #ttnn.layout<tile>, memory_config = #ttnn.memory_config<#dram, <interleaved>>, shape = #ttnn.shape<32x32>}> : (!ttnn.device) -> tensor<32x32xf16, #ttnn_layout2>
    // CHECK: error: 'ttnn.assign' op optional output tensor data type does not match output tensor data type
    %2 = "ttnn.assign"(%arg0, %1) <{output_mem_config = #ttnn.memory_config<#dram, <interleaved>>}> : (tensor<32x32xf32, #ttnn_layout1>, tensor<32x32xf16, #ttnn_layout2>) -> tensor<32x32xf32, #ttnn_layout1>
    return %2 : tensor<32x32xf32, #ttnn_layout1>
  }
}

// -----

// Verify that verification fails when the output tensor's data type doesn't match the provided optional output tensor's data type, even if an output data type is specified.
#dram = #ttnn.buffer_type<dram>
#ttnn_layout1 = #ttnn.ttnn_layout<(d0, d1) -> (d0, d1), <1x1>, memref<1x1x!ttcore.tile<32x32, f32>, #dram>, <interleaved>>
#ttnn_layout2 = #ttnn.ttnn_layout<(d0, d1) -> (d0, d1), <1x1>, memref<1x1x!ttcore.tile<32x32, f16>, #dram>, <interleaved>>
module {
  func.func @forward(%arg0: tensor<32x32xf32, #ttnn_layout1>) -> tensor<32x32xf32, #ttnn_layout1> {
    %0 = "ttnn.get_device"() <{mesh_shape = #ttnn<mesh_shape 1x1>}> : () -> !ttnn.device
    %1 = "ttnn.empty"(%0) <{dtype = #ttcore.supportedDataTypes<f16>, layout = #ttnn.layout<tile>, memory_config = #ttnn.memory_config<#dram, <interleaved>>, shape = #ttnn.shape<32x32>}> : (!ttnn.device) -> tensor<32x32xf16, #ttnn_layout2>
    // CHECK: error: 'ttnn.assign' op optional output tensor data type does not match output tensor data type
    %2 = "ttnn.assign"(%arg0, %1) <{output_mem_config = #ttnn.memory_config<#dram, <interleaved>>, output_dtype = #ttcore.supportedDataTypes<f32>}> : (tensor<32x32xf32, #ttnn_layout1>, tensor<32x32xf16, #ttnn_layout2>) -> tensor<32x32xf32, #ttnn_layout1>
    return %2 : tensor<32x32xf32, #ttnn_layout1>
  }
}
