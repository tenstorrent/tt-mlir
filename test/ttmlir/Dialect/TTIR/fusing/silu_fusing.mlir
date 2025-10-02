// RUN: ttmlir-opt --ttir-fusing %s -o %t.mlir
// RUN: FileCheck %s --input-file=%t.mlir

module {
    // CHECK-LABEL: func.func @silu_fusing
    func.func @silu_fusing(%arg0: tensor<32x3x3x3xbf16> {ttcore.argument_type = #ttcore.argument_type<constant>}, %arg1: tensor<1x3x224x224xbf16>) -> tensor<1x32x112x112xbf16> {
        %0 = ttir.empty() : tensor<1x32x112x112xbf16>
        // CHECK: "ttir.convolution"
        %1 = "ttir.convolution"(%arg1, %arg0, %0) <{batch_group_count = 1 : i64, convolution_layout = #ttir<convolution_layout input_batch = 0, input_feature = 1, input_spatial_dimensions = 2x3, kernel_output_feature = 0, kernel_input_feature = 1, kernel_spatial_dimensions = 2x3, output_batch = 0, output_feature = 1, output_spatial_dimensions = 2x3>, feature_group_count = 1 : i64, input_dilation = array<i64: 1, 1>, padding = array<i64: 1, 1, 1, 1>, weight_dilation = array<i64: 1, 1>, window_reversal = array<i1: false, false>, window_strides = array<i64: 2, 2>}> : (tensor<1x3x224x224xbf16>, tensor<32x3x3x3xbf16>, tensor<1x32x112x112xbf16>) -> tensor<1x32x112x112xbf16>
        %2 = ttir.empty() : tensor<1x32x112x112xbf16>
        // CHECK-NOT: "ttir.sigmoid"
        %3 = "ttir.sigmoid"(%1, %2) : (tensor<1x32x112x112xbf16>, tensor<1x32x112x112xbf16>) -> tensor<1x32x112x112xbf16>
        %4 = ttir.empty() : tensor<1x32x112x112xbf16>
        // CHECK-NOT: "ttir.multiply"
        %5 = "ttir.multiply"(%1, %3, %4) : (tensor<1x32x112x112xbf16>, tensor<1x32x112x112xbf16>, tensor<1x32x112x112xbf16>) -> tensor<1x32x112x112xbf16>
        // CHECK: "ttir.silu"
        return %5 : tensor<1x32x112x112xbf16>
    }
}
