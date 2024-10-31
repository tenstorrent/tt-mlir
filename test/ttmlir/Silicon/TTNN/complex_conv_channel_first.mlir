// RUN: ttmlir-opt --ttir-to-ttnn-backend-pipeline="system-desc-path=%system_desc_path%" %s > %t.mlir
// RUN: FileCheck %s --input-file=%t.mlir
// RUN: ttmlir-translate --ttnn-to-flatbuffer %t.mlir > %t.ttnn
#any_device_tile = #tt.operand_constraint<dram|l1|tile|any_device_tile>
module @jit_convolution {
  func.func public @test_NCHW_IOHW_to_NHWC_OIHW_conv2d(%arg0: tensor<1x3x100x100xbf16>, %arg1: tensor<7x3x3x3xbf16>) -> tensor<1x7x100x100xbf16> {
    %0 = tensor.empty() : tensor<1x7x100x100xbf16>
    // CHECK: %[[C:.*]] = "ttnn.transpose"[[C:.*]]
    // CHECK: %[[C:.*]] = "ttnn.transpose"[[C:.*]]
    // CHECK: %[[C:.*]] = "ttnn.conv2d"[[C:.*]]
    %1 = "ttir.convolution"(%arg0, %arg1, %0) <{
      batch_group_count = 1 : i64,
      convolution_layout = #ttir<convolution_layout
        input_batch = 0,
        input_feature = 1,
        input_spatial_dimensions = 2x3,
        kernel_output_feature = 0,
        kernel_input_feature = 1,
        kernel_spatial_dimensions = 2x3,
        output_batch = 0,
        output_feature = 1,
        output_spatial_dimensions = 2x3
      >,
      feature_group_count = 1 : i64,
      input_dilation = array<i64: 1, 1>,
      operand_constraints = [#any_device_tile, #any_device_tile, #any_device_tile],
      padding = dense<1> : tensor<2x2xi64>,
      weight_dilation = array<i64: 1, 1>,
      window_reversal = array<i1: false, false>,
      window_strides = array<i64: 1, 1>
    }> : (tensor<1x3x100x100xbf16>, tensor<7x3x3x3xbf16>, tensor<1x7x100x100xbf16>) -> tensor<1x7x100x100xbf16>
    // CHECK: %[[C:.*]] = "ttnn.transpose"[[C:.*]]
    // CHECK: %[[C:.*]] = "ttnn.transpose"[[C:.*]]
    return %1 : tensor<1x7x100x100xbf16>
  }
}
