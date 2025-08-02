// REQUIRES: opmodel
// RUN: ttmlir-opt --ttcore-register-device --ttnn-optimizer="post-optimizer-validation-enabled=true" %s -o %t.mlir
// RUN: FileCheck %s --input-file %t.mlir

// Test that the post-optimizer validation analysis automatically detects
// that slice ops require workarounds for strided slice operations.

#dram = #ttnn.buffer_type<dram>
#ttnn_layout_tile_si32 = #ttnn.ttnn_layout<(d0, d1, d2, d3) -> (d0 * 1024 + d1 * 32 + d2, d3), <1x1>, memref<32x1x!ttcore.tile<32x32, si32>, #dram>, <interleaved>>

module attributes {} {
  func.func @slice_with_stride(%arg0: tensor<1x1x64x64xsi32, #ttnn_layout_tile_si32>) -> tensor<1x1x32x32xsi32, #ttnn_layout_tile_si32> {
    %0 = "ttnn.get_device"() <{mesh_shape = #ttnn<mesh_shape 1x1>}> : () -> !ttnn.device

    // The post-optimizer validation should detect that slice requires:
    // BFloat16 data type for strided slice (step > 1)
    // This will trigger the data type workaround through to_layout conversions

    // CHECK: "ttnn.to_layout"
    // CHECK-SAME: dtype = #ttcore.supportedDataTypes<bf16>
    // CHECK: "ttnn.slice"
    // CHECK: "ttnn.to_layout"

    %1 = "ttnn.slice"(%arg0) <{
      begins = [0 : i32, 0 : i32, 0 : i32, 0 : i32],
      ends = [1 : i32, 1 : i32, 64 : i32, 64 : i32],
      step = [1 : i32, 1 : i32, 2 : i32, 2 : i32]
    }> : (tensor<1x1x64x64xsi32, #ttnn_layout_tile_si32>) -> tensor<1x1x32x32xsi32, #ttnn_layout_tile_si32>

    return %1 : tensor<1x1x32x32xsi32, #ttnn_layout_tile_si32>
  }
}
