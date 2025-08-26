// REQUIRES: opmodel
// RUN: ttmlir-opt --ttcore-register-device --ttnn-optimizer --ttnn-operation-validation-and-fallback %s -o %t.mlir
// RUN: FileCheck %s --input-file %t.mlir

// Test that the post-optimizer validation analysis automatically detects
// that sum reduction ops require BFloat16 data type for integer inputs.

#dram = #ttnn.buffer_type<dram>
#ttnn_layout_tile_si32 = #ttnn.ttnn_layout<(d0, d1, d2, d3) -> (d0 * 1024 + d1 * 32 + d2, d3), <1x1>, memref<32x1x!ttcore.tile<32x32, si32>, #dram>, <interleaved>>
#ttnn_layout_tile_bf16 = #ttnn.ttnn_layout<(d0, d1, d2, d3) -> (d0 * 1024 + d1 * 32 + d2, d3), <1x1>, memref<32x1x!ttcore.tile<32x32, bf16>, #dram>, <interleaved>>

module attributes {} {
  func.func @sum_with_integer_input(%arg0: tensor<1x1x32x32xsi32, #ttnn_layout_tile_si32>) -> tensor<1x1x1x1xbf16, #ttnn_layout_tile_bf16> {
    %0 = "ttnn.get_device"() <{mesh_shape = #ttnn<mesh_shape 1x1>}> : () -> !ttnn.device

    // The post-optimizer validation should detect that sum requires:
    // BFloat16 data type for integer inputs to avoid incorrect results

    // CHECK: "ttnn.to_layout"
    // CHECK: "ttnn.sum"

    %1 = "ttnn.sum"(%arg0) <{
      dim_list = [2 : i32, 3 : i32],
      keep_dim = true
    }> : (tensor<1x1x32x32xsi32, #ttnn_layout_tile_si32>) -> tensor<1x1x1x1xbf16, #ttnn_layout_tile_bf16>

    return %1 : tensor<1x1x1x1xbf16, #ttnn_layout_tile_bf16>
  }
}
