// REQUIRES: opmodel
// RUN: ttmlir-opt --ttcore-register-device --ttnn-optimizer="post-optimizer-validation-enabled=true" %s -o %t.mlir
// RUN: FileCheck %s --input-file %t.mlir

// Test that the post-optimizer validation analysis automatically detects
// that embedding op requires BFloat16 for weight tensor.
// The input (indices) can remain in tile layout.

#dram = #ttnn.buffer_type<dram>
#ttnn_layout_tile_input = #ttnn.ttnn_layout<(d0, d1) -> (d0 * 32 + d1), <1x1>, memref<1x1x!ttcore.tile<32x32, u32>, #dram>, <interleaved>>
#ttnn_layout_tile_weight = #ttnn.ttnn_layout<(d0, d1) -> (d0 * 32 + d1), <1x1>, memref<32x1x!ttcore.tile<32x32, f32>, #dram>, <interleaved>>
#ttnn_layout_tile_output = #ttnn.ttnn_layout<(d0, d1, d2) -> (d0 * 32 + d1, d2), <1x1>, memref<32x1x!ttcore.tile<32x32, bf16>, #dram>, <interleaved>>

module attributes {} {
  func.func @embedding_with_tile_input(%arg0: tensor<32x32xui32, #ttnn_layout_tile_input>, %arg1: tensor<1000x32xf32, #ttnn_layout_tile_weight>) -> tensor<32x32x32xbf16, #ttnn_layout_tile_output> {
    %0 = "ttnn.get_device"() <{mesh_shape = #ttnn<mesh_shape 1x1>}> : () -> !ttnn.device

    // The post-optimizer validation should detect that embedding requires:
    // BFloat16 data type for weight tensor
    // Input (indices) can remain in tile layout

    // CHECK: "ttnn.to_layout"
    // CHECK-SAME: -> tensor<1000x32xbf16,
    // CHECK-NEXT: "ttnn.embedding"
    // CHECK: "ttnn.to_layout"
    // CHECK-SAME: -> tensor<32x32x32xbf16,

    %1 = "ttnn.embedding"(%arg0, %arg1) : (tensor<32x32xui32, #ttnn_layout_tile_input>, tensor<1000x32xf32, #ttnn_layout_tile_weight>) -> tensor<32x32x32xbf16, #ttnn_layout_tile_output>

    return %1 : tensor<32x32x32xbf16, #ttnn_layout_tile_output>
  }
}