// REQUIRES: opmodel
// RUN: ttmlir-opt --ttir-to-ttnn-backend-pipeline="enable-optimizer=true" -o %t %s
// RUN: FileCheck %s --input-file=%t

module {
  func.func @conv2d_prepare_weight_and_bias(%arg0: tensor<1x32x32x64xbf16>, %arg1: tensor<64x64x3x3xf32>, %arg2: tensor<1x1x1x64xbf16>) -> tensor<1x30x30x64xbf16> {
    %0 = ttir.empty() : tensor<1x30x30x64xbf16>
    // CHECK: "ttnn.prepare_conv2d_weights"
    // CHECK-SAME: weights_dtype = bf16
    // CHECK-SAME: input_dtype = #ttcore.supportedDataTypes<bf16>
    // CHECK-SAME: output_dtype = #ttcore.supportedDataTypes<bf16>

    // CHECK: "ttnn.prepare_conv2d_bias"
    // CHECK-SAME: weights_dtype = bf16
    // CHECK-SAME: input_dtype = #ttcore.supportedDataTypes<bf16>
    // CHECK-SAME: output_dtype = #ttcore.supportedDataTypes<bf16>
    %1 = "ttir.conv2d"(%arg0, %arg1, %0)
            <{
              stride = 1: i32,
              padding = 0: i32,
              dilation = 1: i32,
              groups = 1: i32
            }> : (tensor<1x32x32x64xbf16>, tensor<64x64x3x3xf32>, tensor<1x30x30x64xbf16>) -> tensor<1x30x30x64xbf16>
    %2 = ttir.empty() : tensor<1x30x30x64xbf16>
    %4 = "ttir.add"(%1, %arg2, %2) : (tensor<1x30x30x64xbf16>, tensor<1x1x1x64xbf16>, tensor<1x30x30x64xbf16>) -> tensor<1x30x30x64xbf16>
    return %4: tensor<1x30x30x64xbf16>
  }
}
