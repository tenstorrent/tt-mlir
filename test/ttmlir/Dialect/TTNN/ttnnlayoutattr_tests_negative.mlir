// RUN: not ttmlir-opt --split-input-file %s 2>&1 | FileCheck %s
// Negative tests for to_memory_config operation

// Verify that verification fails when buffer type is DRAM and grid shape is not 1x1.
#dram = #ttnn.buffer_type<dram>
#dram_layout = #ttnn.ttnn_layout<(d0, d1) -> (d0, d1), <8x8, (d0, d1) -> (0, d0, d1)>, memref<1x1x!ttcore.tile<32x32, f32>, #dram>, <interleaved>>
#system_memory = #ttnn.buffer_type<system_memory>
#system_memory_layout = #ttnn.ttnn_layout<(d0, d1) -> (d0, d1), <1x1>, memref<1x3x!ttcore.tile<32x32, f32>, #system_memory>>
module {
  func.func @forward(%arg0: tensor<32x32xf32, #dram_layout>) -> tensor<32x32xf32, #dram_layout> {
    // CHECK: error: expected (1, 1) grid shape for non-L1 buffer type, got (8, 8) for dram buffer type
    %1 = "ttnn.relu"(%arg0) : (tensor<32x32xf32, #dram_layout>) -> tensor<32x32xf32, #dram_layout>
    return %1 : tensor<32x32xf32, #dram_layout>
  }
}

// -----

// Verify that verification fails when buffer type is system memory and grid shape is not 1x1.
#system_memory = #ttnn.buffer_type<system_memory>
#system_memory_layout = #ttnn.ttnn_layout<(d0, d1) -> (d0, d1), <3x4, (d0, d1) -> (0, d0, d1)>, memref<1x1x!ttcore.tile<32x32, f32>, #system_memory>>
module {
  func.func @forward(%arg0: tensor<32x32xf32, #system_memory_layout>) -> tensor<32x32xf32, #system_memory_layout> {
    // CHECK: error: expected (1, 1) grid shape for non-L1 buffer type, got (3, 4) for system_memory buffer type
    %1 = "ttnn.relu"(%arg0) : (tensor<32x32xf32, #system_memory_layout>) -> tensor<32x32xf32, #system_memory_layout>
    return %1 : tensor<32x32xf32, #system_memory_layout>
  }
}
