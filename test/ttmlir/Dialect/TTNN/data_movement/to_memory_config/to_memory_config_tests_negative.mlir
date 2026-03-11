// RUN: ttmlir-opt --split-input-file %s > /dev/null
// Negative tests for to_memory_config operation

// Verify that verification fails if the output tensor buffer type is not the same as the memory_config one.
#dram = #ttnn.buffer_type<dram>
#system_memory = #ttnn.buffer_type<system_memory>
#device_tile_layout1 = #ttnn.ttnn_layout<(d0, d1) -> (d0, d1), <1x1>, memref<1x1x!ttcore.tile<32x32, f32>, #dram>, <interleaved>>
#device_tile_layout2 = #ttnn.ttnn_layout<(d0, d1) -> (d0, d1), <1x1>, memref<1x3x!ttcore.tile<32x32, f32>, #system_memory>>
module {
  func.func @forward(%arg0: tensor<32x32xf32, #device_tile_layout1>) -> tensor<32x96xf32, #device_tile_layout2> {
    %1 = "ttnn.to_memory_config"(%arg0) : (tensor<32x32xf32, #device_tile_layout1>) -> tensor<32x96xf32, #device_tile_layout2>
    return %1 : tensor<32x96xf32, #device_tile_layout2>
  }
}

// -----

// Verify that verification fails if the output tensor memory layout is not the same as the memory_config one.
#dram = #ttnn.buffer_type<dram>
#l1 = #ttnn.buffer_type<l1>
#device_tile_layout1 = #ttnn.ttnn_layout<(d0, d1) -> (d0, d1), <1x1>, memref<1x1x!ttcore.tile<32x32, f32>, #dram>, <interleaved>>
#device_tile_layout2 = #ttnn.ttnn_layout<(d0, d1) -> (d0, d1), <1x1>, memref<1x3x!ttcore.tile<32x32, f32>, #l1>, <block_sharded>>
module {
  func.func @forward(%arg0: tensor<32x32xf32, #device_tile_layout1>) -> tensor<32x96xf32, #device_tile_layout2> {
    %1 = "ttnn.to_memory_config"(%arg0) : (tensor<32x32xf32, #device_tile_layout1>) -> tensor<32x96xf32, #device_tile_layout2>
    return %1 : tensor<32x96xf32, #device_tile_layout2>
  }
}

// -----

// Verify that verification fails if the output tensor sharding is not the same as the memory_config one.
#dram = #ttnn.buffer_type<dram>
#l1 = #ttnn.buffer_type<l1>
#device_tile_dram_interleaved_layout = #ttnn.ttnn_layout<(d0, d1) -> (d0, d1), <1x1>, memref<3x3x!ttcore.tile<32x32, f32>, #dram>, <interleaved>>
#device_tile_l1_sharded_layout = #ttnn.ttnn_layout<(d0, d1) -> (d0, d1), <3x3>, memref<1x1x!ttcore.tile<32x32, f32>, #l1>, <block_sharded>>
module {
  func.func @forward(%arg0: tensor<96x96xf32, #device_tile_dram_interleaved_layout>) -> tensor<96x96xf32, #device_tile_l1_sharded_layout> {
    %1 = "ttnn.to_memory_config"(%arg0) : (tensor<96x96xf32, #device_tile_dram_interleaved_layout>) -> tensor<96x96xf32, #device_tile_l1_sharded_layout>
    return %1 : tensor<96x96xf32, #device_tile_l1_sharded_layout>
  }
}

// -----

// Verify that memory config attribute verification fails if it has system memory buffer type and any tensor memory layout.
#dram = #ttnn.buffer_type<dram>
#system_memory = #ttnn.buffer_type<system_memory>
#device_tile_layout1 = #ttnn.ttnn_layout<(d0, d1) -> (d0, d1), <1x1>, memref<1x1x!ttcore.tile<32x32, f32>, #dram>, <interleaved>>
#device_tile_layout2 = #ttnn.ttnn_layout<(d0, d1) -> (d0, d1), <1x1>, memref<1x3x!ttcore.tile<32x32, f32>, #system_memory>>
module {
  func.func @forward(%arg0: tensor<32x32xf32, #device_tile_layout1>) -> tensor<32x96xf32, #device_tile_layout2> {
    %1 = "ttnn.to_memory_config"(%arg0) : (tensor<32x32xf32, #device_tile_layout1>) -> tensor<32x96xf32, #device_tile_layout2>
    return %1 : tensor<32x96xf32, #device_tile_layout2>
  }
}

// -----

// Verify that memory config attribute verification fails if it has dram buffer type and doesn't have interleaved memory layout.
#dram = #ttnn.buffer_type<dram>
#device_tile_layout1 = #ttnn.ttnn_layout<(d0, d1) -> (d0, d1), <1x1>, memref<1x1x!ttcore.tile<32x32, f32>, #dram>, <interleaved>>
#device_tile_layout2 = #ttnn.ttnn_layout<(d0, d1) -> (d0, d1), <1x1>, memref<1x3x!ttcore.tile<32x32, f32>, #dram>, <interleaved>>
module{
  func.func @forward(%arg0: tensor<32x32xf32, #device_tile_layout1>) -> tensor<32x96xf32, #device_tile_layout2> {
    %1 = "ttnn.to_memory_config"(%arg0) : (tensor<32x32xf32, #device_tile_layout1>) -> tensor<32x96xf32, #device_tile_layout2>
    return %1 : tensor<32x96xf32, #device_tile_layout2>
  }
}

// -----

// Verify that memory config attribute verification fails if it has dram buffer type and doesn't have tensor memory layout.
#dram = #ttnn.buffer_type<dram>
#device_tile_layout1 = #ttnn.ttnn_layout<(d0, d1) -> (d0, d1), <1x1>, memref<1x1x!ttcore.tile<32x32, f32>, #dram>, <interleaved>>
#device_tile_layout2 = #ttnn.ttnn_layout<(d0, d1) -> (d0, d1), <1x1>, memref<1x3x!ttcore.tile<32x32, f32>, #dram>, <interleaved>>
module{
  func.func @forward(%arg0: tensor<32x32xf32, #device_tile_layout1>) -> tensor<32x96xf32, #device_tile_layout2> {
    %1 = "ttnn.to_memory_config"(%arg0) : (tensor<32x32xf32, #device_tile_layout1>) -> tensor<32x96xf32, #device_tile_layout2>
    return %1 : tensor<32x96xf32, #device_tile_layout2>
  }
}

// -----

// Verify that memory config attribute verification fails if it has sharded tensor memory layout and doesn't have l1 buffer type.
#dram = #ttnn.buffer_type<dram>
#l1_small = #ttnn.buffer_type<l1_small>
#device_tile_layout1 = #ttnn.ttnn_layout<(d0, d1) -> (d0, d1), <1x1>, memref<3x3x!ttcore.tile<32x32, f32>, #dram>, <interleaved>>
#device_tile_layout2 = #ttnn.ttnn_layout<(d0, d1) -> (d0, d1), <3x3>, memref<1x1x!ttcore.tile<32x32, f32>, #l1_small>, <block_sharded>>
module{
  func.func @forward(%arg0: tensor<96x96xf32, #device_tile_layout1>) -> tensor<96x96xf32, #device_tile_layout2> {
    %1 = "ttnn.to_memory_config"(%arg0) : (tensor<96x96xf32, #device_tile_layout1>) -> tensor<96x96xf32, #device_tile_layout2>
    return %1 : tensor<96x96xf32, #device_tile_layout2>
  }
}

// -----

// Verify that memory config attribute verification fails if it has sharded layout and doesn't have a sharded tensor memory layout.
#dram = #ttnn.buffer_type<dram>
#l1 = #ttnn.buffer_type<l1>
#device_tile_layout1 = #ttnn.ttnn_layout<(d0, d1) -> (d0, d1), <1x1>, memref<3x3x!ttcore.tile<32x32, f32>, #dram>, <interleaved>>
#device_tile_layout2 = #ttnn.ttnn_layout<(d0, d1) -> (d0, d1), <3x3>, memref<1x1x!ttcore.tile<32x32, f32>, #l1>, <block_sharded>>
module{
  func.func @forward(%arg0: tensor<96x96xf32, #device_tile_layout1>) -> tensor<96x96xf32, #device_tile_layout2> {
    %1 = "ttnn.to_memory_config"(%arg0) : (tensor<96x96xf32, #device_tile_layout1>) -> tensor<96x96xf32, #device_tile_layout2>
    return %1 : tensor<96x96xf32, #device_tile_layout2>
  }
}
