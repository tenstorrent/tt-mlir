// RUN: ttmlir-opt --ttir-fusing="ttnn-enable-conv2d-with-multiply-pattern=true" -o %t %s
// RUN: FileCheck %s --input-file=%t

module {
    func.func @test_bottleneck_block(%arg0: tensor<1x64x56x56xbf16>, 
                                                                     %arg1: tensor<64x64x1x1xbf16> {ttcore.argument_type = #ttcore.argument_type<constant>},
                                                                     %arg2: tensor<64xbf16> {ttcore.argument_type = #ttcore.argument_type<constant>},
                                                                     %arg3: tensor<64xbf16> {ttcore.argument_type = #ttcore.argument_type<constant>},
                                                                     %arg4: tensor<64xbf16> {ttcore.argument_type = #ttcore.argument_type<constant>},
                                                                     %arg5: tensor<64xbf16> {ttcore.argument_type = #ttcore.argument_type<constant>},
                                                                     %arg6: tensor<64x64x3x3xbf16> {ttcore.argument_type = #ttcore.argument_type<constant>},
                                                                     %arg7: tensor<64xbf16> {ttcore.argument_type = #ttcore.argument_type<constant>},
                                                                     %arg8: tensor<64xbf16> {ttcore.argument_type = #ttcore.argument_type<constant>},
                                                                     %arg9: tensor<64xbf16> {ttcore.argument_type = #ttcore.argument_type<constant>},
                                                                     %arg10: tensor<64xbf16> {ttcore.argument_type = #ttcore.argument_type<constant>},
                                                                     %arg11: tensor<256x64x1x1xbf16> {ttcore.argument_type = #ttcore.argument_type<constant>},
                                                                     %arg12: tensor<256xbf16> {ttcore.argument_type = #ttcore.argument_type<constant>},
                                                                     %arg13: tensor<256xbf16> {ttcore.argument_type = #ttcore.argument_type<constant>},
                                                                     %arg14: tensor<256xbf16> {ttcore.argument_type = #ttcore.argument_type<constant>},
                                                                     %arg15: tensor<256xbf16> {ttcore.argument_type = #ttcore.argument_type<constant>},
                                                                     %arg16: tensor<256x64x1x1xbf16> {ttcore.argument_type = #ttcore.argument_type<constant>},
                                                                     %arg17: tensor<256xbf16> {ttcore.argument_type = #ttcore.argument_type<constant>},
                                                                     %arg18: tensor<256xbf16> {ttcore.argument_type = #ttcore.argument_type<constant>},
                                                                     %arg19: tensor<256xbf16> {ttcore.argument_type = #ttcore.argument_type<constant>},
                                                                     %arg20: tensor<256xbf16> {ttcore.argument_type = #ttcore.argument_type<constant>}) 
                                                                     -> tensor<1x256x56x56xbf16> {
        
        %0 = ttir.empty() : tensor<1x64x56x56xbf16>
        %1 = "ttir.convolution"(%arg0, %arg1, %0) <{batch_group_count = 1 : i64, convolution_layout = #ttir<convolution_layout input_batch = 0, input_feature = 1, input_spatial_dimensions = 2x3, kernel_output_feature = 0, kernel_input_feature = 1, kernel_spatial_dimensions = 2x3, output_batch = 0, output_feature = 1, output_spatial_dimensions = 2x3>, feature_group_count = 1 : i64, input_dilation = array<i64: 1, 1>, padding = array<i64: 0, 0, 0, 0>, weight_dilation = array<i64: 1, 1>, window_reversal = array<i1: false, false>, window_strides = array<i64: 1, 1>}> : (tensor<1x64x56x56xbf16>, tensor<64x64x1x1xbf16>, tensor<1x64x56x56xbf16>) -> tensor<1x64x56x56xbf16>
        
        %2 = ttir.empty() : tensor<1x64x56x56xbf16>
        %3 = "ttir.batch_norm"(%1, %arg2, %arg3, %arg4, %arg5, %2) <{dimension = 1 : i32, epsilon = 9.99999974E-6 : f32, training = false}> : (tensor<1x64x56x56xbf16>, tensor<64xbf16>, tensor<64xbf16>, tensor<64xbf16>, tensor<64xbf16>, tensor<1x64x56x56xbf16>) -> tensor<1x64x56x56xbf16>
        
        %4 = ttir.empty() : tensor<1x64x56x56xbf16>
        %5 = "ttir.relu"(%3, %4) : (tensor<1x64x56x56xbf16>, tensor<1x64x56x56xbf16>) -> tensor<1x64x56x56xbf16>
        
        %6 = ttir.empty() : tensor<1x64x56x56xbf16>
        %7 = "ttir.convolution"(%5, %arg6, %6) <{batch_group_count = 1 : i64, convolution_layout = #ttir<convolution_layout input_batch = 0, input_feature = 1, input_spatial_dimensions = 2x3, kernel_output_feature = 0, kernel_input_feature = 1, kernel_spatial_dimensions = 2x3, output_batch = 0, output_feature = 1, output_spatial_dimensions = 2x3>, feature_group_count = 1 : i64, input_dilation = array<i64: 1, 1>, padding = array<i64: 1, 1, 1, 1>, weight_dilation = array<i64: 1, 1>, window_reversal = array<i1: false, false>, window_strides = array<i64: 1, 1>}> : (tensor<1x64x56x56xbf16>, tensor<64x64x3x3xbf16>, tensor<1x64x56x56xbf16>) -> tensor<1x64x56x56xbf16>
        
        %8 = ttir.empty() : tensor<1x64x56x56xbf16>
        %9 = "ttir.batch_norm"(%7, %arg7, %arg8, %arg9, %arg10, %8) <{dimension = 1 : i32, epsilon = 9.99999974E-6 : f32, training = false}> : (tensor<1x64x56x56xbf16>, tensor<64xbf16>, tensor<64xbf16>, tensor<64xbf16>, tensor<64xbf16>, tensor<1x64x56x56xbf16>) -> tensor<1x64x56x56xbf16>
        
        %10 = ttir.empty() : tensor<1x64x56x56xbf16>
        %11 = "ttir.relu"(%9, %10) : (tensor<1x64x56x56xbf16>, tensor<1x64x56x56xbf16>) -> tensor<1x64x56x56xbf16>

        %12 = ttir.empty() : tensor<1x256x56x56xbf16>
        %13 = "ttir.convolution"(%11, %arg11, %12) <{batch_group_count = 1 : i64, convolution_layout = #ttir<convolution_layout input_batch = 0, input_feature = 1, input_spatial_dimensions = 2x3, kernel_output_feature = 0, kernel_input_feature = 1, kernel_spatial_dimensions = 2x3, output_batch = 0, output_feature = 1, output_spatial_dimensions = 2x3>, feature_group_count = 1 : i64, input_dilation = array<i64: 1, 1>, padding = array<i64: 0, 0, 0, 0>, weight_dilation = array<i64: 1, 1>, window_reversal = array<i1: false, false>, window_strides = array<i64: 1, 1>}> : (tensor<1x64x56x56xbf16>, tensor<256x64x1x1xbf16>, tensor<1x256x56x56xbf16>) -> tensor<1x256x56x56xbf16>
        
        %14 = ttir.empty() : tensor<1x256x56x56xbf16>
        %15 = "ttir.batch_norm"(%13, %arg12, %arg13, %arg14, %arg15, %14) <{dimension = 1 : i32, epsilon = 9.99999974E-6 : f32, training = false}> : (tensor<1x256x56x56xbf16>, tensor<256xbf16>, tensor<256xbf16>, tensor<256xbf16>, tensor<256xbf16>, tensor<1x256x56x56xbf16>) -> tensor<1x256x56x56xbf16>
        
        %16 = ttir.empty() : tensor<1x256x56x56xbf16>
        %17 = "ttir.convolution"(%arg0, %arg16, %16) <{batch_group_count = 1 : i64, convolution_layout = #ttir<convolution_layout input_batch = 0, input_feature = 1, input_spatial_dimensions = 2x3, kernel_output_feature = 0, kernel_input_feature = 1, kernel_spatial_dimensions = 2x3, output_batch = 0, output_feature = 1, output_spatial_dimensions = 2x3>, feature_group_count = 1 : i64, input_dilation = array<i64: 1, 1>, padding = array<i64: 0, 0, 0, 0>, weight_dilation = array<i64: 1, 1>, window_reversal = array<i1: false, false>, window_strides = array<i64: 1, 1>}> : (tensor<1x64x56x56xbf16>, tensor<256x64x1x1xbf16>, tensor<1x256x56x56xbf16>) -> tensor<1x256x56x56xbf16>
        
        %18 = ttir.empty() : tensor<1x256x56x56xbf16>
        %19 = "ttir.batch_norm"(%17, %arg17, %arg18, %arg19, %arg20, %18) <{dimension = 1 : i32, epsilon = 9.99999974E-6 : f32, training = false}> : (tensor<1x256x56x56xbf16>, tensor<256xbf16>, tensor<256xbf16>, tensor<256xbf16>, tensor<256xbf16>, tensor<1x256x56x56xbf16>) -> tensor<1x256x56x56xbf16>

        %20 = ttir.empty() : tensor<1x256x56x56xbf16>
        %21 = "ttir.add"(%15, %19, %20) : (tensor<1x256x56x56xbf16>, tensor<1x256x56x56xbf16>, tensor<1x256x56x56xbf16>) -> tensor<1x256x56x56xbf16>

        %22 = ttir.empty() : tensor<1x256x56x56xbf16>
        %23 = "ttir.relu"(%21, %22) : (tensor<1x256x56x56xbf16>, tensor<1x256x56x56xbf16>) -> tensor<1x256x56x56xbf16>
        
        // CHECK: "ttir.add"
        // CHECK: "ttir.multiply"
        // CHECK: "ttir.convolution"
        // CHECK-NOT: "ttir.batch_norm"
        
        return %23 : tensor<1x256x56x56xbf16>
    }
}

