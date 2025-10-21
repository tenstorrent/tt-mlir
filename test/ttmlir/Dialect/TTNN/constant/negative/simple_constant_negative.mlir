// RUN: not ttmlir-opt --split-input-file %s 2>&1 | FileCheck %s

// Verify that device must exist for non-system memory buffer types
#dram = #ttnn.buffer_type<dram>
#ttnn_layout_dram_rm = #ttnn.ttnn_layout<(d0, d1) -> (d0, d1), <1x1>, memref<1x2xbf16, #ttnn.buffer_type<dram>>, <interleaved>>
module {
  func.func @test_dram_no_device() -> tensor<1x2xbf16, #ttnn_layout_dram_rm> {
    // CHECK: error: 'ttnn.constant' op device operand must be specified for non-system memory buffer type
    %0 = "ttnn.constant"() <{ value = dense<[[0.0, 0.0]]> : tensor<1x2xbf16>, dtype = #ttcore.supportedDataTypes<bf16>, layout = #ttnn.layout<row_major>, memory_config = #ttnn.memory_config<#dram, <interleaved>>}> : () -> tensor<1x2xbf16, #ttnn_layout_dram_rm>
    return %0 : tensor<1x2xbf16, #ttnn_layout_dram_rm>
  }
}

// -----
// Verify that device must not be specified for system memory buffer types
#system_memory = #ttnn.buffer_type<system_memory>
#ttnn_layout_host_rm = #ttnn.ttnn_layout<(d0, d1) -> (d0, d1), <1x1>, memref<1x2xbf16, #ttnn.buffer_type<system_memory>>>
module {
  func.func @test_host_device() -> tensor<1x2xbf16, #ttnn_layout_host_rm> {
    // CHECK: error: 'ttnn.constant' op device operand must not be specified for system memory buffer type
    %0 = "ttnn.get_device"() : () -> !ttnn.device
    %1 = "ttnn.constant"(%0) <{ value = dense<[[0.0, 0.0]]> : tensor<1x2xbf16>, dtype = #ttcore.supportedDataTypes<bf16>, layout = #ttnn.layout<row_major>, memory_config = #ttnn.memory_config<#system_memory>}> : (!ttnn.device) -> tensor<1x2xbf16, #ttnn_layout_host_rm>
    return %1 : tensor<1x2xbf16, #ttnn_layout_host_rm>
  }
}
