// REQUIRES: opmodel
// RUN: ttmlir-opt --ttcore-register-device --ttcore-mark-functions-as-forward --ttnn-operation-validation-and-fallback="max-fallback-attempts=50" %s -o %t.mlir
// RUN: FileCheck %s --input-file %t.mlir

// Test that config fallbacks are tried for non-OOM errors like slice window
// misalignment. This conv2d configuration can fail with index overflow errors
// that are fixed by trying different slice configurations.

#dram = #ttnn.buffer_type<dram>
#system_memory = #ttnn.buffer_type<system_memory>
#ttnn_layout_input = #ttnn.ttnn_layout<(d0, d1, d2, d3) -> (d0 * 524288 + d1 * 524288 + d2, d3), <1x1>, memref<16384x8x!ttcore.tile<32x32, bf16>, #dram>, <interleaved>>
#ttnn_layout_weight = #ttnn.ttnn_layout<(d0, d1, d2, d3) -> (d0 * 768 + d1 * 3 + d2, d3), <1x1>, memref<196608x3xbf16, #system_memory>>
#ttnn_layout_output = #ttnn.ttnn_layout<(d0, d1, d2, d3) -> (d0 * 524288 + d1 * 524288 + d2, d3), <1x1>, memref<16384x8x!ttcore.tile<32x32, bf16>, #dram>, <interleaved>>

module attributes {} {
  func.func @conv2d_non_oom_config_fallback(%arg0: tensor<1x1x524288x256xbf16, #ttnn_layout_input>,
                                            %arg1: tensor<256x256x3x3xbf16, #ttnn_layout_weight>) -> tensor<1x1x524288x256xbf16, #ttnn_layout_output> {
    %0 = "ttnn.get_device"() <{mesh_shape = #ttnn<mesh_shape 1x1>}> : () -> !ttnn.device

    // This conv2d may fail with slice window misalignment (index overflow).
    // The validation pass should try config fallbacks for non-OOM errors.

    // CHECK: %[[RES:.*]] = "ttnn.conv2d"
    // CHECK-SAME: conv2d_slice_config = #ttnn.conv2d_slice_config<dram_width

    %1 = "ttnn.conv2d"(%arg0, %arg1, %0) <{
      batch_size = 2 : i32,
      compute_config = #ttnn.device_compute_kernel_config<math_fidelity = hifi4, fp32_dest_acc_en = true>,
      conv2d_config = #ttnn.conv2d_config<weights_dtype = bf16, deallocate_activation = true, act_block_h_override = 0, enable_kernel_stride_folding = false, config_tensors_in_dram = true>,
      conv2d_slice_config = #ttnn.conv2d_slice_config<l1_full, 0>,
      dilation = array<i32: 1, 1>,
      dtype = #ttcore.supportedDataTypes<bf16>,
      groups = 1 : i32,
      in_channels = 256 : i32,
      input_height = 512 : i32,
      input_width = 512 : i32,
      kernel_size = array<i32: 3, 3>,
      out_channels = 256 : i32,
      padding = array<i32: 1, 1, 1, 1>,
      stride = array<i32: 1, 1>
    }> : (tensor<1x1x524288x256xbf16, #ttnn_layout_input>,
         tensor<256x256x3x3xbf16, #ttnn_layout_weight>,
         !ttnn.device) -> tensor<1x1x524288x256xbf16, #ttnn_layout_output>

    return %1 : tensor<1x1x524288x256xbf16, #ttnn_layout_output>
  }
}
