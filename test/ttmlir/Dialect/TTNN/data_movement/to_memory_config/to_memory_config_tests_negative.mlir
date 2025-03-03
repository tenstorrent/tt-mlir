// RUN: not ttmlir-opt --split-input-file %s 2>&1 | FileCheck %s
// Negative tests for to_memory_config operation

// Verify that verification fails if the output tensor buffer type is not the same as the memory_config one.
#dram = #ttnn.buffer_type<dram>
#system_memory = #ttnn.buffer_type<system_memory>
#device_tile_layout1 = #ttnn.ttnn_layout<(d0, d1) -> (d0, d1), <1x1>, memref<1x1x!tt.tile<32x32, f32>, #dram>, <interleaved>>
#device_tile_layout2 = #ttnn.ttnn_layout<(d0, d1) -> (d0, d1), <1x1>, memref<1x3x!tt.tile<32x32, f32>, #system_memory>>
module {
  func.func @forward(%arg0: tensor<32x32xf32, #device_tile_layout1>) -> tensor<32x96xf32, #device_tile_layout2> {
    // CHECK: error: 'ttnn.to_memory_config' op Output tensor buffer type must match memory config buffer type.
    %1 = "ttnn.to_memory_config"(%arg0) <{memory_config = #ttnn.memory_config<#dram, <<1x3>>, <interleaved>>}> : (tensor<32x32xf32, #device_tile_layout1>) -> tensor<32x96xf32, #device_tile_layout2>
    return %1 : tensor<32x96xf32, #device_tile_layout2>
  }
}

// -----

// Verify that verification fails if the output tensor buffer memory layout is not the same as the memory_config one.
#dram = #ttnn.buffer_type<dram>
#device_tile_layout1 = #ttnn.ttnn_layout<(d0, d1) -> (d0, d1), <1x1>, memref<1x1x!tt.tile<32x32, f32>, #dram>, <interleaved>>
#device_tile_layout2 = #ttnn.ttnn_layout<(d0, d1) -> (d0, d1), <1x1>, memref<1x3x!tt.tile<32x32, f32>, #dram>, <interleaved>>
module{
  func.func @forward(%arg0: tensor<32x32xf32, #device_tile_layout1>) -> tensor<32x96xf32, #device_tile_layout2> {
    // CHECK: error: DRAM buffer type must have Interleaved memory layout.
    %1 = "ttnn.to_memory_config"(%arg0) <{memory_config = #ttnn.memory_config<#dram, <<1x3>>, <single_bank>>}> : (tensor<32x32xf32, #device_tile_layout1>) -> tensor<32x96xf32, #device_tile_layout2>
    return %1 : tensor<32x96xf32, #device_tile_layout2>
  }
}

// -----

// Verify that verification fails if the output tensor sharding is not the same as the memory_config one.
#dram = #ttnn.buffer_type<dram>
#l1 = #ttnn.buffer_type<l1>
#device_tile_layout1 = #ttnn.ttnn_layout<(d0, d1) -> (d0, d1), <1x1>, memref<1x1x!tt.tile<32x32, f32>, #dram>, <interleaved>>
#device_tile_layout2 = #ttnn.ttnn_layout<(d0, d1) -> (d0, d1), <1x1>, memref<1x3x!tt.tile<32x32, f32>, #l1>, <interleaved>>
module {
  func.func @forward(%arg0: tensor<32x32xf32, #device_tile_layout1>) -> tensor<32x96xf32, #device_tile_layout2> {
    // CHECK: error: 'ttnn.to_memory_config' op Output tensor shard spec must match memory config shard spec.
    %1 = "ttnn.to_memory_config"(%arg0) <{memory_config = #ttnn.memory_config<#l1, <<1x4>>, <interleaved>>}> : (tensor<32x32xf32, #device_tile_layout1>) -> tensor<32x96xf32, #device_tile_layout2>
    return %1 : tensor<32x96xf32, #device_tile_layout2>
  }
}

// -----

// Verify that memory_config attribute can't have system memory buffer type and tensor memory layout
#dram = #ttnn.buffer_type<dram>
#system_memory = #ttnn.buffer_type<system_memory>
#device_tile_layout1 = #ttnn.ttnn_layout<(d0, d1) -> (d0, d1), <1x1>, memref<1x1x!tt.tile<32x32, f32>, #dram>, <interleaved>>
#device_tile_layout2 = #ttnn.ttnn_layout<(d0, d1) -> (d0, d1), <1x1>, memref<1x3x!tt.tile<32x32, f32>, #system_memory>>
module {
  func.func @forward(%arg0: tensor<32x32xf32, #device_tile_layout1>) -> tensor<32x96xf32, #device_tile_layout2> {
    // CHECK: error: Memory layout is not allowed for SystemMemory buffer type.
    %1 = "ttnn.to_memory_config"(%arg0) <{memory_config = #ttnn.memory_config<#system_memory, <<1x4>>, <interleaved>>}> : (tensor<32x32xf32, #device_tile_layout1>) -> tensor<32x96xf32, #device_tile_layout2>
    return %1 : tensor<32x96xf32, #device_tile_layout2>
  }
}
