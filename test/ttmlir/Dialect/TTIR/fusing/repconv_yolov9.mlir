    func.func @repconv(
        %arg679: tensor<128xbf16> {ttcore.argument_type = #ttcore.argument_type<constant>},
        %arg680: tensor<128xbf16> {ttcore.argument_type = #ttcore.argument_type<constant>},
        %arg681: tensor<128xbf16> {ttcore.argument_type = #ttcore.argument_type<constant>},
        %arg682: tensor<128x256x1x1xbf16> {ttcore.argument_type = #ttcore.argument_type<constant>},
        %arg684: tensor<128xbf16> {ttcore.argument_type = #ttcore.argument_type<constant>},
        %arg685: tensor<128xbf16> {ttcore.argument_type = #ttcore.argument_type<constant>},
        %arg686: tensor<128xbf16> {ttcore.argument_type = #ttcore.argument_type<constant>},
        %arg687: tensor<128x128x3x3xbf16> {ttcore.argument_type = #ttcore.argument_type<constant>},
        %arg677: tensor<128x128x1x1xbf16>   {ttcore.argument_type = #ttcore.argument_type<constant>},
        %arg676: tensor<128xbf16> {ttcore.argument_type = #ttcore.argument_type<constant>},
        %arg675: tensor<128xbf16> {ttcore.argument_type = #ttcore.argument_type<constant>},
        %arg674: tensor<128xbf16> {ttcore.argument_type = #ttcore.argument_type<constant>},
        %3643: tensor<1x256x20x20xbf16>,
        %3257: tensor<128xbf16> {ttcore.argument_type = #ttcore.argument_type<constant>},
        %3261: tensor<128xbf16> {ttcore.argument_type = #ttcore.argument_type<constant>},
        %3265: tensor<128xbf16> {ttcore.argument_type = #ttcore.argument_type<constant>}
    ) -> tensor<1x128x20x20xbf16> {

        %3786 = ttir.empty() : tensor<1x128x20x20xbf16> 
        %3787 = "ttir.convolution"(%3643, %arg682, %3786) <{batch_group_count = 1 : i64, convolution_layout = #ttir<convolution_layout input_batch = 0, input_feature = 1, input_spatial_dimensions = 2x3, kernel_output_feature = 0, kernel_input_feature = 1, kernel_spatial_dimensions = 2x3, output_batch = 0, output_feature = 1, output_spatial_dimensions = 2x3>, feature_group_count = 1 : i64, input_dilation = array<i64: 1, 1>, padding = array<i64: 0, 0, 0, 0>, weight_dilation = array<i64: 1, 1>, window_reversal = array<i1: false, false>, window_strides = array<i64: 1, 1>}> : (tensor<1x256x20x20xbf16>, tensor<128x256x1x1xbf16>, tensor<1x128x20x20xbf16>) -> tensor<1x128x20x20xbf16> 
        %3788 = ttir.empty() : tensor<1x1x128xbf16> 
        %3789 = "ttir.reshape"(%arg681, %3788) <{shape = [1 : i32, 1 : i32, 128 : i32]}> : (tensor<128xbf16>, tensor<1x1x128xbf16>) -> tensor<1x1x128xbf16> 
        %3790 = ttir.empty() : tensor<128xbf16> 
        %3791 = "ttir.reshape"(%3789, %3790) <{shape = [128 : i32]}> : (tensor<1x1x128xbf16>, tensor<128xbf16>) -> tensor<128xbf16> 
        %3792 = ttir.empty() : tensor<1x1x128xbf16> 
        %3793 = "ttir.reshape"(%arg680, %3792) <{shape = [1 : i32, 1 : i32, 128 : i32]}> : (tensor<128xbf16>, tensor<1x1x128xbf16>) -> tensor<1x1x128xbf16> 
        %3794 = ttir.empty() : tensor<128xbf16> 
        %3795 = "ttir.reshape"(%3793, %3794) <{shape = [128 : i32]}> : (tensor<1x1x128xbf16>, tensor<128xbf16>) -> tensor<128xbf16> 
        %3796 = ttir.empty() : tensor<1x1x128xbf16> 
        %3797 = "ttir.reshape"(%arg679, %3796) <{shape = [1 : i32, 1 : i32, 128 : i32]}> : (tensor<128xbf16>, tensor<1x1x128xbf16>) -> tensor<1x1x128xbf16> 
        %3798 = ttir.empty() : tensor<128xbf16> 
        %3799 = "ttir.reshape"(%3797, %3798) <{shape = [128 : i32]}> : (tensor<1x1x128xbf16>, tensor<128xbf16>) -> tensor<128xbf16> 
        %3800 = ttir.empty() : tensor<1x128x20x20xbf16> 
        %3801 = "ttir.batch_norm_inference"(%3787, %3791, %3795, %3799, %3257, %3800) <{dimension = 1 : i32, epsilon = 1.000000e-03 : f32}> : (tensor<1x128x20x20xbf16>, tensor<128xbf16>, tensor<128xbf16>, tensor<128xbf16>, tensor<128xbf16>, tensor<1x128x20x20xbf16>) -> tensor<1x128x20x20xbf16> 
        %3802 = ttir.empty() : tensor<1x128x20x20xbf16> 
        %3803 = "ttir.sigmoid"(%3801, %3802) : (tensor<1x128x20x20xbf16>, tensor<1x128x20x20xbf16>) -> tensor<1x128x20x20xbf16> 
        %3804 = ttir.empty() : tensor<1x128x20x20xbf16> 
        %3805 = "ttir.multiply"(%3801, %3803, %3804) : (tensor<1x128x20x20xbf16>, tensor<1x128x20x20xbf16>, tensor<1x128x20x20xbf16>) -> tensor<1x128x20x20xbf16> 
        %3806 = ttir.empty() : tensor<1x128x20x20xbf16>     
        %3807 = "ttir.convolution"(%3805, %arg687, %3806) <{batch_group_count = 1 : i64, convolution_layout = #ttir<convolution_layout input_batch = 0, input_feature = 1, input_spatial_dimensions = 2x3, kernel_output_feature = 0, kernel_input_feature = 1, kernel_spatial_dimensions = 2x3, output_batch = 0, output_feature = 1, output_spatial_dimensions = 2x3>, feature_group_count = 1 : i64, input_dilation = array<i64: 1, 1>, padding = array<i64: 1, 1, 1, 1>, weight_dilation = array<i64: 1, 1>, window_reversal = array<i1: false, false>, window_strides = array<i64: 1, 1>}> : (tensor<1x128x20x20xbf16>, tensor<128x128x3x3xbf16>, tensor<1x128x20x20xbf16>) -> tensor<1x128x20x20xbf16> 
        %3808 = ttir.empty() : tensor<1x1x128xbf16> 
        %3809 = "ttir.reshape"(%arg686, %3808) <{shape = [1 : i32, 1 : i32, 128 : i32]}> : (tensor<128xbf16>, tensor<1x1x128xbf16>) -> tensor<1x1x128xbf16> 
        %3810 = ttir.empty() : tensor<128xbf16> 
        %3811 = "ttir.reshape"(%3809, %3810) <{shape = [128 : i32]}> : (tensor<1x1x128xbf16>, tensor<128xbf16>) -> tensor<128xbf16> 
        %3812 = ttir.empty() : tensor<1x1x128xbf16> 
        %3813 = "ttir.reshape"(%arg685, %3812) <{shape = [1 : i32, 1 : i32, 128 : i32]}> : (tensor<128xbf16>, tensor<1x1x128xbf16>) -> tensor<1x1x128xbf16> 
        %3814 = ttir.empty() : tensor<128xbf16> 
        %3815 = "ttir.reshape"(%3813, %3814) <{shape = [128 : i32]}> : (tensor<1x1x128xbf16>, tensor<128xbf16>) -> tensor<128xbf16> 
        %3816 = ttir.empty() : tensor<1x1x128xbf16> 
        %3817 = "ttir.reshape"(%arg684, %3816) <{shape = [1 : i32, 1 : i32, 128 : i32]}> : (tensor<128xbf16>, tensor<1x1x128xbf16>) -> tensor<1x1x128xbf16> 
        %3818 = ttir.empty() : tensor<128xbf16> 
        %3819 = "ttir.reshape"(%3817, %3818) <{shape = [128 : i32]}> : (tensor<1x1x128xbf16>, tensor<128xbf16>) -> tensor<128xbf16> 
        %3820 = ttir.empty() : tensor<1x128x20x20xbf16> 
        %3821 = "ttir.batch_norm_inference"(%3807, %3811, %3815, %3819, %3265, %3820) <{dimension = 1 : i32, epsilon = 1.000000e-03 : f32}> : (tensor<1x128x20x20xbf16>, tensor<128xbf16>, tensor<128xbf16>, tensor<128xbf16>, tensor<128xbf16>, tensor<1x128x20x20xbf16>) -> tensor<1x128x20x20xbf16> 
        %3822 = ttir.empty() : tensor<1x128x20x20xbf16> 
        %3823 = "ttir.convolution"(%3805, %arg677, %3822) <{batch_group_count = 1 : i64, convolution_layout = #ttir<convolution_layout input_batch = 0, input_feature = 1, input_spatial_dimensions = 2x3, kernel_output_feature = 0, kernel_input_feature = 1, kernel_spatial_dimensions = 2x3, output_batch = 0, output_feature = 1, output_spatial_dimensions = 2x3>, feature_group_count = 1 : i64, input_dilation = array<i64: 1, 1>, padding = array<i64: 0, 0, 0, 0>, weight_dilation = array<i64: 1, 1>, window_reversal = array<i1: false, false>, window_strides = array<i64: 1, 1>}> : (tensor<1x128x20x20xbf16>, tensor<128x128x1x1xbf16>, tensor<1x128x20x20xbf16>) -> tensor<1x128x20x20xbf16> 
        %3824 = ttir.empty() : tensor<1x1x128xbf16> 
        %3825 = "ttir.reshape"(%arg676, %3824) <{shape = [1 : i32, 1 : i32, 128 : i32]}> : (tensor<128xbf16>, tensor<1x1x128xbf16>) -> tensor<1x1x128xbf16> 
        %3826 = ttir.empty() : tensor<128xbf16> 
        %3827 = "ttir.reshape"(%3825, %3826) <{shape = [128 : i32]}> : (tensor<1x1x128xbf16>, tensor<128xbf16>) -> tensor<128xbf16> 
        %3828 = ttir.empty() : tensor<1x1x128xbf16> 
        %3829 = "ttir.reshape"(%arg675, %3828) <{shape = [1 : i32, 1 : i32, 128 : i32]}> : (tensor<128xbf16>, tensor<1x1x128xbf16>) -> tensor<1x1x128xbf16> 
        %3830 = ttir.empty() : tensor<128xbf16> 
        %3831 = "ttir.reshape"(%3829, %3830) <{shape = [128 : i32]}> : (tensor<1x1x128xbf16>, tensor<128xbf16>) -> tensor<128xbf16> 
        %3832 = ttir.empty() : tensor<1x1x128xbf16> 
        %3833 = "ttir.reshape"(%arg674, %3832) <{shape = [1 : i32, 1 : i32, 128 : i32]}> : (tensor<128xbf16>, tensor<1x1x128xbf16>) -> tensor<1x1x128xbf16> 
        %3834 = ttir.empty() : tensor<128xbf16> 
        %3835 = "ttir.reshape"(%3833, %3834) <{shape = [128 : i32]}> : (tensor<1x1x128xbf16>, tensor<128xbf16>) -> tensor<128xbf16> 
        %3836 = ttir.empty() : tensor<1x128x20x20xbf16> 
        %3837 = "ttir.batch_norm_inference"(%3823, %3827, %3831, %3835, %3261, %3836) <{dimension = 1 : i32, epsilon = 1.000000e-03 : f32}> : (tensor<1x128x20x20xbf16>, tensor<128xbf16>, tensor<128xbf16>, tensor<128xbf16>, tensor<128xbf16>, tensor<1x128x20x20xbf16>) -> tensor<1x128x20x20xbf16> 
        %3838 = ttir.empty() : tensor<1x128x20x20xbf16> 
        %3839 = "ttir.add"(%3821, %3837, %3838) : (tensor<1x128x20x20xbf16>, tensor<1x128x20x20xbf16>, tensor<1x128x20x20xbf16>) -> tensor<1x128x20x20xbf16> 
        %3840 = ttir.empty() : tensor<1x128x20x20xbf16> 
        %3841 = "ttir.sigmoid"(%3839, %3840) : (tensor<1x128x20x20xbf16>, tensor<1x128x20x20xbf16>) -> tensor<1x128x20x20xbf16> 
        %3842 = ttir.empty() : tensor<1x128x20x20xbf16> 
        return %3841 : tensor<1x128x20x20xbf16>
    }