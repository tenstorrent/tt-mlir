// RUN: not ttmlir-opt --split-input-file %s 2>&1 | FileCheck %s
// Negative tests for TTNNLayoutAttr

// Verify that verification fails if tensor memory layout is set for SystemMemory buffer type.
#system_memory = #ttnn.buffer_type<system_memory>
#ttnn_layout = #ttnn.ttnn_layout<(d0, d1) -> (d0, d1), <1x1>, memref<32x32xf32, #system_memory>, <interleaved>>
#ttnn_layout1 = #ttnn.ttnn_layout<(d0, d1) -> (d0, d1), <1x1>, memref<1x1x!tt.tile<32x32, f32>, #system_memory>, <interleaved>>
module {
  func.func @forward(%arg0: tensor<32x32xf32, #ttnn_layout>) -> tensor<32x96xf32, #ttnn_layout1> {
    // CHECK: error: Memory layout is not allowed for SystemMemory buffer type.
    %1 = "ttnn.to_layout"(%arg0) <{layout = #ttnn.layout<tile>}>  : (tensor<32x32xf32, #ttnn_layout>) -> tensor<32x32xf32, #ttnn_layout1>
    return %1 : tensor<32x32xf32, #ttnn_layout1>
  }
}

// -----

// Verify that verification fails if tensor memory layout is set to anything other than interleaved for DRAM buffer type.
#dram = #ttnn.buffer_type<dram>
#ttnn_layout = #ttnn.ttnn_layout<(d0, d1) -> (d0, d1), <1x1>, memref<32x32xf32, #dram>, <block_sharded>>
#ttnn_layout1 = #ttnn.ttnn_layout<(d0, d1) -> (d0, d1), <1x1>, memref<1x1x!tt.tile<32x32, f32>, #dram>, <block_sharded>>
module {
  func.func @forward(%arg0: tensor<32x32xf32, #ttnn_layout>) -> tensor<32x96xf32, #ttnn_layout1> {
    // CHECK: error: DRAM buffer type must have Interleaved memory layout.
    %1 = "ttnn.to_layout"(%arg0) <{layout = #ttnn.layout<tile>}>  : (tensor<32x32xf32, #ttnn_layout>) -> tensor<32x32xf32, #ttnn_layout1>
    return %1 : tensor<32x32xf32, #ttnn_layout1>
  }
}

// -----

// Verify that verification fails if tensor memory layout is not set for DRAM/L1 buffer type.
#l1 = #ttnn.buffer_type<l1>
#ttnn_layout = #ttnn.ttnn_layout<(d0, d1) -> (d0, d1), <1x1>, memref<32x32xf32, #l1>>
#ttnn_layout1 = #ttnn.ttnn_layout<(d0, d1) -> (d0, d1), <1x1>, memref<1x1x!tt.tile<32x32, f32>, #l1>>
module {
  func.func @forward(%arg0: tensor<32x32xf32, #ttnn_layout>) -> tensor<32x96xf32, #ttnn_layout1> {
    // CHECK: error: Memory layout is required for non-SystemMemory buffer type.
    %1 = "ttnn.to_layout"(%arg0) <{layout = #ttnn.layout<tile>}>  : (tensor<32x32xf32, #ttnn_layout>) -> tensor<32x32xf32, #ttnn_layout1>
    return %1 : tensor<32x32xf32, #ttnn_layout1>
  }
}
