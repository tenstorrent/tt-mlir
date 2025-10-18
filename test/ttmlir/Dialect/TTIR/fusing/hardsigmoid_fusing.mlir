// RUN: ttmlir-opt --canonicalize --ttir-fusing -o %t %s
// RUN: FileCheck %s --input-file=%t
module {
    func.func @main(%arg0: tensor<12x256x1x1xbf16>, %arg1: tensor<256x256x1x1xbf16>) -> tensor<12x256x1x1xbf16> {
        %0 = "ttir.constant"() <{value = dense<0.000000e+00> : tensor<12x256x1x1xbf16>}> : () -> tensor<12x256x1x1xbf16>
        %1 = "ttir.constant"() <{value = dense<3.000000e+00> : tensor<12x256x1x1xbf16>}> : () -> tensor<12x256x1x1xbf16>
        %2 = "ttir.constant"() <{value = dense<6.000000e+00> : tensor<12x256x1x1xbf16>}> : () -> tensor<12x256x1x1xbf16>
        %3 = ttir.empty() : tensor<12x256x1x1xbf16>
        // CHECK: "ttir.convolution"
        %4 = "ttir.convolution"(%arg0, %arg1, %3) <{batch_group_count = 1 : i64, convolution_layout = #ttir<convolution_layout input_batch = 0, input_feature = 1, input_spatial_dimensions = 2x3, kernel_output_feature = 0, kernel_input_feature = 1, kernel_spatial_dimensions = 2x3, output_batch = 0, output_feature = 1, output_spatial_dimensions = 2x3>, feature_group_count = 1 : i64, input_dilation = array<i64: 1, 1>, padding = array<i64: 0, 0, 0, 0>, weight_dilation = array<i64: 1, 1>, window_reversal = array<i1: false, false>, window_strides = array<i64: 1, 1>}> : (tensor<12x256x1x1xbf16>, tensor<256x256x1x1xbf16>, tensor<12x256x1x1xbf16>) -> tensor<12x256x1x1xbf16>
        %5 = ttir.empty() : tensor<12x256x1x1xbf16>
        // CHECK-NOT: "ttir.add"
        // CHECK-NOT: "ttir.clamp_tensor"
        // CHECK-NOT: "ttir.div"
        // CHECK: "ttir.hardsigmoid"
        %6 = "ttir.add"(%4, %1, %5) : (tensor<12x256x1x1xbf16>, tensor<12x256x1x1xbf16>, tensor<12x256x1x1xbf16>) -> tensor<12x256x1x1xbf16>
        %7 = ttir.empty() : tensor<12x256x1x1xbf16>
        %8 = "ttir.clamp_tensor"(%6, %0, %2, %7) : (tensor<12x256x1x1xbf16>, tensor<12x256x1x1xbf16>, tensor<12x256x1x1xbf16>, tensor<12x256x1x1xbf16>) -> tensor<12x256x1x1xbf16>
        %9 = ttir.empty() : tensor<12x256x1x1xbf16>
        %10 = "ttir.clamp_tensor"(%8, %0, %2, %9) : (tensor<12x256x1x1xbf16>, tensor<12x256x1x1xbf16>, tensor<12x256x1x1xbf16>, tensor<12x256x1x1xbf16>) -> tensor<12x256x1x1xbf16>
        %11 = ttir.empty() : tensor<12x256x1x1xbf16>
        %12 = "ttir.div"(%10, %2, %11) : (tensor<12x256x1x1xbf16>, tensor<12x256x1x1xbf16>, tensor<12x256x1x1xbf16>) -> tensor<12x256x1x1xbf16>
        return %12 : tensor<12x256x1x1xbf16>
    }
}
