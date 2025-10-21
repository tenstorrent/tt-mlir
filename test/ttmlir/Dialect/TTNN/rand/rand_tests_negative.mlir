// RUN: not ttmlir-opt --split-input-file %s 2>&1 | FileCheck %s
// Negative tests for rand operation

#dram = #ttnn.buffer_type<dram>
#ttnn_layout = #ttnn.ttnn_layout<(d0, d1) -> (d0, d1), <1x1>, memref<1x1x!ttcore.tile<32x32, bf16>, #dram>, <interleaved>>
module {
  func.func @test_rand_dtype() -> tensor<32x32xbf16, #ttnn_layout> {
    // CHECK: error: 'ttnn.rand' op output tensor layout data type bf16 must match output data type attribute f32
    %0 = "ttnn.get_device"() <{mesh_offset = #ttnn<mesh_offset 0x0>, mesh_shape = #ttnn<mesh_shape 1x1>}> : () -> !ttnn.device
    %1 = "ttnn.rand"(%0) <{dtype = #ttcore.supportedDataTypes<f32>, high = 1.000000e+00 : f32, layout = #ttnn.layout<tile>, low = 0.000000e+00 : f32, memory_config = #ttnn.memory_config<#dram, <interleaved>>, seed = 0 : ui32, size = #ttnn.shape<32x32>}> : (!ttnn.device) -> tensor<32x32xbf16, #ttnn_layout>
    return %1 : tensor<32x32xbf16, #ttnn_layout>
  }
}

// -----
#dram = #ttnn.buffer_type<dram>
#ttnn_layout = #ttnn.ttnn_layout<(d0, d1) -> (d0, d1), <1x1>, memref<1x1x!ttcore.tile<32x32, bf16>, #dram>, <interleaved>>
module {
  func.func @test_rand_interval() -> tensor<32x32xbf16, #ttnn_layout> {
    // CHECK: error: 'ttnn.rand' op 'low' value must be < 'high' value
    %0 = "ttnn.get_device"() <{mesh_offset = #ttnn<mesh_offset 0x0>, mesh_shape = #ttnn<mesh_shape 1x1>}> : () -> !ttnn.device
    %1 = "ttnn.rand"(%0) <{dtype = #ttcore.supportedDataTypes<bf16>, high = 1.000000e+00 : f32, layout = #ttnn.layout<tile>, low = 1.000000e+00 : f32, memory_config = #ttnn.memory_config<#dram, <interleaved>>, seed = 0 : ui32, size = #ttnn.shape<32x32>}> : (!ttnn.device) -> tensor<32x32xbf16, #ttnn_layout>
    return %1 : tensor<32x32xbf16, #ttnn_layout>
  }
}

// -----
#dram = #ttnn.buffer_type<dram>
#ttnn_layout = #ttnn.ttnn_layout<(d0, d1) -> (d0, d1), <1x1>, memref<1x1x!ttcore.tile<32x32, bf16>, #dram>, <interleaved>>
module {
  func.func @test_rand_size() -> tensor<32x32xbf16, #ttnn_layout> {
    // CHECK: error: 'ttnn.rand' op size argument does not match with output tensor shape. [Size = (64, 64), output tensor shape = (32, 32)]
    %0 = "ttnn.get_device"() <{mesh_offset = #ttnn<mesh_offset 0x0>, mesh_shape = #ttnn<mesh_shape 1x1>}> : () -> !ttnn.device
    %1 = "ttnn.rand"(%0) <{dtype = #ttcore.supportedDataTypes<bf16>, high = 1.000000e+00 : f32, layout = #ttnn.layout<tile>, low = 0.000000e+00 : f32, memory_config = #ttnn.memory_config<#dram, <interleaved>>, seed = 0 : ui32, size = #ttnn.shape<64x64>}> : (!ttnn.device) -> tensor<32x32xbf16, #ttnn_layout>
    return %1 : tensor<32x32xbf16, #ttnn_layout>
  }
}

// -----
#dram = #ttnn.buffer_type<dram>
#ttnn_layout = #ttnn.ttnn_layout<(d0, d1) -> (d0, d1), <1x1>, memref<1x1x!ttcore.tile<32x32, bf16>, #dram>, <interleaved>>
module {
  func.func @test_rand_layout() -> tensor<32x32xbf16, #ttnn_layout> {
    // CHECK: error: 'ttnn.rand' op output tensor layout tile must match output layout attribute row_major
    %0 = "ttnn.get_device"() <{mesh_offset = #ttnn<mesh_offset 0x0>, mesh_shape = #ttnn<mesh_shape 1x1>}> : () -> !ttnn.device
    %1 = "ttnn.rand"(%0) <{dtype = #ttcore.supportedDataTypes<bf16>, high = 1.000000e+00 : f32, layout = #ttnn.layout<row_major>, low = 0.000000e+00 : f32, memory_config = #ttnn.memory_config<#dram, <interleaved>>, seed = 0 : ui32, size = #ttnn.shape<32x32>}> : (!ttnn.device) -> tensor<32x32xbf16, #ttnn_layout>
    return %1 : tensor<32x32xbf16, #ttnn_layout>
  }
}
