// REQUIRES: opmodel
// RUN: ttmlir-opt --ttcore-register-device --ttnn-operation-validation-and-fallback %s -o %t.mlir
// RUN: FileCheck %s --input-file %t.mlir

// Test that the post-optimizer validation analysis automatically detects
// that pool2d ops require row-major layout and BFloat16 data type.

#dram = #ttnn.buffer_type<dram>
#ttnn_layout_tile_f32 = #ttnn.ttnn_layout<(d0, d1, d2, d3) -> (d0 * 1024 + d1 * 32 + d2, d3), <1x1>, memref<32x1x!ttcore.tile<32x32, f32>, #dram>, <interleaved>>
#ttnn_layout_tile_bf16 = #ttnn.ttnn_layout<(d0, d1, d2, d3) -> (d0, d1, d2, d3), <1x1>, memref<128x1x!ttcore.tile<32x32, bf16>, #dram>, <interleaved>>

module attributes {} {
  func.func @max_pool2d_with_tile_f32_input(%arg0: tensor<1x1x16384x32xf32, #ttnn_layout_tile_f32>) -> tensor<1x1x4096x32xbf16, #ttnn_layout_tile_bf16> {
    %0 = "ttnn.get_device"() <{mesh_shape = #ttnn<mesh_shape 1x1>}> : () -> !ttnn.device

    // The op validation pass should detect that max_pool2d requires:
    // 1. BFloat16 data type for input
    // 2. Returns row-major layout
    // Therefore, the pass will insert the necessary type conversion and revert output layout ops.

    // CHECK: "ttnn.to_layout"
    // CHECK-SAME: dtype = #ttcore.supportedDataTypes<bf16>
    // CHECK-SAME: -> tensor<1x1x16384x32xbf16,
    // CHECK-NEXT: "ttnn.max_pool2d"
    // CHECK-SAME: -> tensor<1x1x4096x32xbf16,
    // CHECK-NEXT: "ttnn.to_layout"
    // CHECK-SAME: layout = #ttnn.layout<tile>

    %1 = "ttnn.max_pool2d"(%arg0) <{
      batch_size = 1 : si32,
      ceil_mode = false,
      channels = 32 : si32,
      input_height = 128 : si32,
      input_width = 128 : si32,
      kernel_size = array<i32: 2, 2>,
      stride = array<i32: 2, 2>,
      dilation = array<i32: 1, 1>,
      padding = array<i32: 0, 0>,
      in_place_halo = false
    }> : (tensor<1x1x16384x32xf32, #ttnn_layout_tile_f32>) -> tensor<1x1x4096x32xbf16, #ttnn_layout_tile_bf16>

    return %1 : tensor<1x1x4096x32xbf16, #ttnn_layout_tile_bf16>
  }
}
