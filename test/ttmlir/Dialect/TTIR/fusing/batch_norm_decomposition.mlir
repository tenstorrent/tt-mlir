// REQUIRES: opmodel
// RUN: ttmlir-opt --ttir-to-ttnn-backend-pipeline="system-desc-path=%system_desc_path% enable-optimizer=true memory-layout-analysis-enabled=true" %s -o %t.mlir
// RUN: FileCheck %s --input-file=%t.mlir
// RUN: ttmlir-translate --ttnn-to-flatbuffer -o %t.ttnn %t.mlir

func.func @convolution_batch_norm(%arg0: tensor<1x3x224x224xbf16>, %arg1: tensor<64x3x7x7xbf16>, %arg2: tensor<64xbf16>, %arg3: tensor<64xbf16>, %arg4: tensor<64xbf16>, %arg5: tensor<64xbf16>) -> tensor<1x64x112x112xbf16> {
    %0 = ttir.empty() : tensor<1x64x112x112xbf16>
    // CHECK: ttnn.conv2d
    %1 = "ttir.convolution"(%arg0, %arg1, %0) <{batch_group_count = 1 : i64, convolution_layout = #ttir<convolution_layout input_batch = 0, input_feature = 1, input_spatial_dimensions = 2x3, kernel_output_feature = 0, kernel_input_feature = 1, kernel_spatial_dimensions = 2x3, output_batch = 0, output_feature = 1, output_spatial_dimensions = 2x3>, feature_group_count = 1 : i64, input_dilation = array<i64: 1, 1>, padding = array<i64: 3, 3, 3, 3>, weight_dilation = array<i64: 1, 1>, window_reversal = array<i1: false, false>, window_strides = array<i64: 2, 2>}> : (tensor<1x3x224x224xbf16>, tensor<64x3x7x7xbf16>, tensor<1x64x112x112xbf16>) -> tensor<1x64x112x112xbf16>
    %2 = ttir.empty() : tensor<1x64x112x112xbf16>
    // CHECK-NOT: ttnn.batch_norm
    %3 = "ttir.batch_norm"(%1, %arg2, %arg3, %arg4, %arg5, %2) <{dimension = 1 : i32, epsilon = 9.99999974E-6 : f32, training = false}> : (tensor<1x64x112x112xbf16>, tensor<64xbf16>, tensor<64xbf16>, tensor<64xbf16>, tensor<64xbf16>, tensor<1x64x112x112xbf16>) -> tensor<1x64x112x112xbf16>
    return %3 : tensor<1x64x112x112xbf16>
}
