// REQUIRES: opmodel
// RUN: ttmlir-opt --ttcore-register-device --ttcore-mark-functions-as-forward --ttnn-greedy-memory-layout-propagation %s --mlir-print-local-scope -o %t.mlir
// RUN: FileCheck %s --input-file %t.mlir

// Test that the DRAM interleaved fallback derives a layout per result instead
// of reusing result 0's layout for every result.
//
// The layernorm_fw kernel only accepts bfloat16, so an f32 operand makes every
// candidate config fail validation and the op falls back to DRAM interleaved.
// It returns the full-size output alongside the per-row mean/rstd statistics,
// so stamping result 0's encoding onto results 1 and 2 would leave their
// tensor types describing a 128x256 memref while their shape is 1x1x128x1.

#dram = #ttnn.buffer_type<dram>
#ttnn_layout_out = #ttnn.ttnn_layout<(d0, d1, d2, d3) -> (d0 * 128 + d1 * 128 + d2, d3), <1x1>, memref<4x8x!ttcore.tile<32x32, f32>, #dram>, <interleaved>>
#ttnn_layout_w = #ttnn.ttnn_layout<(d0, d1, d2, d3) -> (d0 + d1 + d2, d3), <1x1>, memref<1x8x!ttcore.tile<32x32, f32>, #dram>, <interleaved>>
#ttnn_layout_stat = #ttnn.ttnn_layout<(d0, d1, d2, d3) -> (d0 * 128 + d1 * 128 + d2, d3), <1x1>, memref<4x1x!ttcore.tile<32x32, f32>, #dram>, <interleaved>>

module attributes {} {
  func.func @layernorm_fw_mean_rstd_dram_fallback(
      %arg0: tensor<1x1x128x256xf32, #ttnn_layout_out>,
      %arg1: tensor<1x1x1x256xf32, #ttnn_layout_w>,
      %arg2: tensor<1x1x1x256xf32, #ttnn_layout_w>)
      -> (tensor<1x1x128x256xf32, #ttnn_layout_out>,
          tensor<1x1x128x1xf32, #ttnn_layout_stat>,
          tensor<1x1x128x1xf32, #ttnn_layout_stat>) {
    %0 = "ttnn.get_device"() <{mesh_shape = #ttnn<mesh_shape 1x1>}> : () -> !ttnn.device

    // The mean/rstd results keep a 4x1 tile memref matching their 1x1x128x1
    // shape rather than inheriting the output's 4x8 memref.
    // CHECK: "ttnn.layernorm_fw"
    // CHECK-SAME: tensor<1x1x128x1xf32
    // CHECK-SAME: memref<4x1x!ttcore.tile<32x32, f32>
    %1, %2, %3 = "ttnn.layernorm_fw"(%arg0, %arg1, %arg2) <{
        epsilon = 1.000000e-05 : f32,
        return_mean_rstd = true}>
        : (tensor<1x1x128x256xf32, #ttnn_layout_out>,
           tensor<1x1x1x256xf32, #ttnn_layout_w>,
           tensor<1x1x1x256xf32, #ttnn_layout_w>)
          -> (tensor<1x1x128x256xf32, #ttnn_layout_out>,
              tensor<1x1x128x1xf32, #ttnn_layout_stat>,
              tensor<1x1x128x1xf32, #ttnn_layout_stat>)

    return %1, %2, %3 : tensor<1x1x128x256xf32, #ttnn_layout_out>,
                        tensor<1x1x128x1xf32, #ttnn_layout_stat>,
                        tensor<1x1x128x1xf32, #ttnn_layout_stat>
  }
}
