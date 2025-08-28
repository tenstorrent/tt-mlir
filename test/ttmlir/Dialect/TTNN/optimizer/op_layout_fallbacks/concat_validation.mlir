// REQUIRES: opmodel
// RUN: ttmlir-opt --ttcore-register-device --ttnn-optimizer --ttnn-operation-validation-and-fallback %s -o %t.mlir
// RUN: FileCheck %s --input-file %t.mlir

// Test that the post-optimizer validation analysis automatically detects
// that concat ops require BFloat16 data type when using tile layout with shapes not divisible by tile size.

#dram = #ttnn.buffer_type<dram>
#ttnn_layout_tile_si32 = #ttnn.ttnn_layout<(d0, d1, d2, d3) -> (d0 * 1024 + d1 * 32 + d2, d3), <1x1>, memref<32x1x!ttcore.tile<32x32, si32>, #dram>, <interleaved>>
#ttnn_layout_tile_bf16 = #ttnn.ttnn_layout<(d0, d1, d2, d3) -> (d0 * 1024 + d1 * 32 + d2, d3), <1x1>, memref<32x1x!ttcore.tile<32x32, bf16>, #dram>, <interleaved>>

module attributes {} {
  func.func @concat_with_tile_i32_non_divisible_shapes(%arg0: tensor<1x1x17x32xsi32, #ttnn_layout_tile_si32>, %arg1: tensor<1x1x15x32xsi32, #ttnn_layout_tile_si32>) -> tensor<1x1x32x32xbf16, #ttnn_layout_tile_bf16> {
    %0 = "ttnn.get_device"() <{mesh_shape = #ttnn<mesh_shape 1x1>}> : () -> !ttnn.device

    // The post-optimizer validation should detect that concat requires:
    // BFloat16 data type when using tile layout with shapes not divisible by tile size (32)
    // Input shapes 17 and 15 are not divisible by 32, so padding is needed, which requires BFloat16

    // CHECK: "ttnn.to_layout"
    // CHECK: "ttnn.concat"

    %1 = "ttnn.concat"(%arg0, %arg1) <{
      dim = 2 : si32
    }> : (tensor<1x1x17x32xsi32, #ttnn_layout_tile_si32>, tensor<1x1x15x32xsi32, #ttnn_layout_tile_si32>) -> tensor<1x1x32x32xbf16, #ttnn_layout_tile_bf16>

    return %1 : tensor<1x1x32x32xbf16, #ttnn_layout_tile_bf16>
  }
}
