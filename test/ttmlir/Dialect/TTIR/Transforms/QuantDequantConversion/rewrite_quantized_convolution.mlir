// RUN: ttmlir-opt --ttir-quant-dequant-conversion %s | FileCheck %s
module {
    func.func @test_quantized_conv(%arg0: tensor<1x3x224x224xf32>, %arg1: tensor<64x3x7x7xf32>) -> tensor<1x64x112x112xf32> {
        %0 = ttir.empty() : tensor<1x3x224x224x!quant.uniform<i8:f32, 0.1>>
        %1 = "ttir.quantize"(%arg0, %0) : (tensor<1x3x224x224xf32>, tensor<1x3x224x224x!quant.uniform<i8:f32, 0.1>>) -> tensor<1x3x224x224x!quant.uniform<i8:f32, 0.1>>
        %2 = ttir.empty() : tensor<1x3x224x224xf32>
        %3 = "ttir.dequantize"(%1, %2) : (tensor<1x3x224x224x!quant.uniform<i8:f32, 0.1>>, tensor<1x3x224x224xf32>) -> tensor<1x3x224x224xf32>
        %4 = ttir.empty() : tensor<64x3x7x7x!quant.uniform<i8:f32, 0.1>>
        %5 = "ttir.quantize"(%arg1, %4) : (tensor<64x3x7x7xf32>, tensor<64x3x7x7x!quant.uniform<i8:f32, 0.1>>) -> tensor<64x3x7x7x!quant.uniform<i8:f32, 0.1>>
        %6 = ttir.empty() : tensor<64x3x7x7xf32>
        %7 = "ttir.dequantize"(%5, %6) : (tensor<64x3x7x7x!quant.uniform<i8:f32, 0.1>>, tensor<64x3x7x7xf32>) -> tensor<64x3x7x7xf32>
        %8 = ttir.empty() : tensor<1x64x112x112xf32>
        %9 = "ttir.convolution"(%3, %7, %8) <{batch_group_count = 1 : i64, convolution_layout = #ttir<convolution_layout input_batch = 0, input_feature = 1, input_spatial_dimensions = 2x3, kernel_output_feature = 0, kernel_input_feature = 1, kernel_spatial_dimensions = 2x3, output_batch = 0, output_feature = 1, output_spatial_dimensions = 2x3>, feature_group_count = 1 : i64, input_dilation = array<i64: 1, 1>, padding = array<i64: 3, 3, 3, 3>, weight_dilation = array<i64: 1, 1>, window_reversal = array<i1: false, false>, window_strides = array<i64: 2, 2>}> : (tensor<1x3x224x224xf32>, tensor<64x3x7x7xf32>, tensor<1x64x112x112xf32>) -> tensor<1x64x112x112xf32>
        return %9 : tensor<1x64x112x112xf32>
    }
}
