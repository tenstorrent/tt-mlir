// RUN: not ttmlir-opt --split-input-file %s 2>&1 | FileCheck %s
// Negative tests for TTNNLayoutAttr

// Verify that verification fails if tensor memory layout is set for SystemMemory buffer type.
#system_memory = #ttnn.buffer_type<system_memory>
#ttnn_layout = #ttnn.ttnn_layout<(d0, d1) -> (d0, d1), <1x1>, memref<32x32xf32, #system_memory>, <interleaved>>
#ttnn_layout1 = #ttnn.ttnn_layout<(d0, d1) -> (d0, d1), <1x1>, memref<1x1x!ttcore.tile<32x32, f32>, #system_memory>, <interleaved>>
module {
  func.func @forward(%arg0: tensor<32x32xf32, #ttnn_layout>) -> tensor<32x96xf32, #ttnn_layout1> {
    // CHECK: error: Memory layout is not allowed for SystemMemory buffer type.
    %1 = "ttnn.to_layout"(%arg0)   : (tensor<32x32xf32, #ttnn_layout>) -> tensor<32x32xf32, #ttnn_layout1>
    return %1 : tensor<32x32xf32, #ttnn_layout1>
  }
}

// -----

// Verify that verification fails if a sharded TTNN layout is missing its
// core_range_set.
#dram = #ttnn.buffer_type<dram>
#ttnn_layout = #ttnn.ttnn_layout<(d0, d1) -> (d0, d1), <1x8>, memref<32x4xf32, #dram>, <width_sharded>>
#ttnn_layout1 = #ttnn.ttnn_layout<(d0, d1) -> (d0, d1), <1x8>, memref<1x1x!ttcore.tile<32x32, f32>, #dram>, <width_sharded>>
module {
  func.func @forward(%arg0: tensor<32x32xf32, #ttnn_layout>) -> tensor<32x96xf32, #ttnn_layout1> {
    // CHECK: error: sharded TTNN layout (width_sharded) must carry a core_range_set
    %1 = "ttnn.to_layout"(%arg0)   : (tensor<32x32xf32, #ttnn_layout>) -> tensor<32x32xf32, #ttnn_layout1>
    return %1 : tensor<32x32xf32, #ttnn_layout1>
  }
}

// -----

// Verify that verification fails if tensor memory layout is not set for DRAM/L1 buffer type.
#l1 = #ttnn.buffer_type<l1>
#ttnn_layout = #ttnn.ttnn_layout<(d0, d1) -> (d0, d1), <1x1>, memref<32x32xf32, #l1>>
#ttnn_layout1 = #ttnn.ttnn_layout<(d0, d1) -> (d0, d1), <1x1>, memref<1x1x!ttcore.tile<32x32, f32>, #l1>>
module {
  func.func @forward(%arg0: tensor<32x32xf32, #ttnn_layout>) -> tensor<32x96xf32, #ttnn_layout1> {
    // CHECK: error: Memory layout is required for non-SystemMemory buffer type.
    %1 = "ttnn.to_layout"(%arg0)   : (tensor<32x32xf32, #ttnn_layout>) -> tensor<32x32xf32, #ttnn_layout1>
    return %1 : tensor<32x32xf32, #ttnn_layout1>
  }
}
