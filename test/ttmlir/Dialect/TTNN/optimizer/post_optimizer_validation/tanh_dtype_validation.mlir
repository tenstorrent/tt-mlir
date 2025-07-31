// REQUIRES: opmodel
// RUN: ttmlir-opt --ttcore-register-device --ttnn-optimizer="post-optimizer-validation-enabled=true" %s -o %t.mlir
// RUN: FileCheck %s --input-file %t.mlir

// Test that the post-optimizer validation analysis automatically detects
// that tanh op requires BFloat16 data type and inserts the necessary type
// conversion transformations for non-BFloat16 inputs.

#dram = #ttnn.buffer_type<dram>
#ttnn_layout_tile_f32 = #ttnn.ttnn_layout<(d0, d1, d2, d3) -> (d0 * 1024 + d1 * 32 + d2, d3), <1x1>, memref<32x1x!ttcore.tile<32x32, f32>, #dram>, <interleaved>>
#ttnn_layout_tile_bf16 = #ttnn.ttnn_layout<(d0, d1, d2, d3) -> (d0 * 1024 + d1 * 32 + d2, d3), <1x1>, memref<32x1x!ttcore.tile<32x32, bf16>, #dram>, <interleaved>>

module attributes {} {
  func.func @tanh_with_f32_input(%arg0: tensor<1x1x32x32xf32, #ttnn_layout_tile_f32>) -> tensor<1x1x32x32xbf16, #ttnn_layout_tile_bf16> {
    %0 = "ttnn.get_device"() <{mesh_shape = #ttnn<mesh_shape 1x1>}> : () -> !ttnn.device

    // The post-optimizer validation should detect that tanh requires BFloat16
    // and automatically insert type conversion for float32 input
    
    // CHECK: "ttnn.to_layout"
    // CHECK-SAME: -> tensor<1x1x32x32xbf16,
    // CHECK-NEXT: "ttnn.tanh"

    %1 = "ttnn.tanh"(%arg0) : (tensor<1x1x32x32xf32, #ttnn_layout_tile_f32>) -> tensor<1x1x32x32xbf16, #ttnn_layout_tile_bf16>

    return %1 : tensor<1x1x32x32xbf16, #ttnn_layout_tile_bf16>
  }
}